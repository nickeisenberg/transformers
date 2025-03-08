import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int, 
                 num_heads: int, num_encoder_layers: int, num_decoder_layers: int,
                 dim_feedforward: int, max_len: int = 5000, 
                 dropout: float | None = 0.1):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size, embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
            max_len=max_len, dropout=dropout
        )

        self.decoder = TransformerDecoder(
            tgt_vocab_size, embed_dim, num_heads, num_decoder_layers,
            dim_feedforward, max_len, dropout
        )

        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)


    def forward(self, input_tokens: torch.Tensor,
                target_tokens: torch.Tensor,
                src_padding_mask: torch.Tensor | None = None,
                src_padding_token: float = 0,
                tgt_look_ahead_mask: torch.Tensor | None = None):
        encoder_output = self.encoder(
            input_tokens=input_tokens, padding_mask=src_padding_mask,
            padding_value=src_padding_token
        )

        decoder_output = self.decoder(
            target_tokens=target_tokens, encoder_output=encoder_output,
            look_ahead_mask=tgt_look_ahead_mask, padding_mask=src_padding_mask,
            padding_value=src_padding_token
        )

        #output shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        output = self.fc_out(decoder_output)  

        return output


    def inference(self, input_tokens: torch.Tensor, max_len: int, 
                  sos_token: float, eos_token: float, 
                  src_padding_mask: torch.Tensor | None = None):
        """
        Inference method that generates the target sequence autoregressively.

        Args:
            input_tokens: (batch_size, src_seq_len) Source sequence.
            max_len: Maximum length of the generated sequence.
            sos_token: The start-of-sequence token (special token to begin the generation).
            eos_token: The end-of-sequence token (special token to end the generation).
            src_padding_mask: Optional padding mask for the source.

        Returns:
            Generated target sequence of shape (batch_size, generated_seq_len).
        """

        device = input_tokens.device

        encoder_output = self.encoder(input_tokens=input_tokens,
                                      padding_mask=src_padding_mask)

        batch_size = input_tokens.size(0)
        tgt_tokens = torch.full(
            (batch_size, 1), sos_token, dtype=torch.long, device=input_tokens.device
        )

        # Iteratively generate tokens
        for _ in range(max_len):
            tgt_look_ahead_mask = create_look_ahead_mask(
                tgt_tokens.size(1), device
            )
            decoder_output = self.decoder(
                target_tokens=tgt_tokens, encoder_output=encoder_output,
                look_ahead_mask=tgt_look_ahead_mask
            )

            # (batch_size, tgt_seq_len, tgt_vocab_size)
            logits = self.fc_out(decoder_output)  
            
            # (batch_size, tgt_vocab_size)
            next_token_logits = logits[:, -1, :]  
            
            # (batch_size, 1)
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            # (batch_size, seq_len + 1)
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)  

            if torch.all(next_token == eos_token):
                break

        return tgt_tokens


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        if not embed_dim % num_heads == 0:
            raise Exception("Embedding dimension must be divisible by the number of heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension of each attention head

        # Linear layers for query, key, and value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final linear layer to combine all the heads' outputs
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                padding_mask: torch.Tensor | None = None,
                padding_value: int | float = 0):
        # batch_size = query.size(0)
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)  # Sequence length for K (which could differ from Q)

        # Linear projection to compute Q, K, and V
        Q = self.q_proj(query)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(key)    # (batch_size, seq_len, embed_dim)
        V = self.v_proj(value)  # (batch_size, seq_len, embed_dim)

        # Reshape for multi-head attention (split into multiple heads)
        # New shape: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(
            batch_size, seq_len_q, self.num_heads, self.head_dim
        ).transpose(1, 2).contiguous()

        K = K.view(
            batch_size, seq_len_k, self.num_heads, self.head_dim
        ).transpose(1, 2).contiguous()

        V = V.view(
            batch_size, seq_len_k, self.num_heads, self.head_dim
        ).transpose(1, 2).contiguous()

        # Perform scaled dot-product attention for each head, with optional padding mask
        attn_output, attn_weights = self.scaled_dot_product_attention(
            Q, K, V, padding_mask, padding_value
        )

        # Concatenate the attention outputs from all heads
        # New shape: (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len_q, self.embed_dim
        ).contiguous()

        # Apply the final linear layer to combine the outputs from all heads
        output = self.out_proj(attn_output)

        return output, attn_weights

    def scaled_dot_product_attention(self, query: torch.Tensor, key: torch.Tensor,
                                     value: torch.Tensor,
                                     padding_mask: torch.Tensor | None = None,
                                     padding_value: float | int = 0):
        # Compute the attention scores
        # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores /= torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply the padding mask if provided
        if padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                padding_mask == padding_value, float('-inf')
            )

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout to the attention weights
        attn_weights = self.dropout(attn_weights)

        # Compute the weighted sum of the values
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()

        # Create a matrix of shape (max_len, embed_dim) for positional encodings
        pe = torch.zeros(max_len, embed_dim)  # This should not be self.pe yet

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)

        # Now register it as a buffer
        self.register_buffer('pe', pe)


    def forward(self, x: torch.Tensor):
        if isinstance(self.pe, torch.Tensor):
            x = x + self.pe[:, :x.size(1), :]
            return x
        else:
            raise Exception("pe is not a Tensor")


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int, 
                 dropout: float | None = 0.1):
        super().__init__()

        self.self_attention = SelfAttention(embed_dim, num_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None,
                padding_value: float = 0):
        attn_output, _ = self.self_attention(
            x, x, x, padding_mask, padding_value
        )
        x += self.dropout(attn_output) if self.dropout else attn_output
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x += self.dropout(ff_output) if self.dropout else ff_output
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, 
                 num_layers: int, dim_feedforward, max_len=5000, 
                 dropout: float | None = 0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input_tokens, padding_mask=None, padding_value=0):
        x = self.embedding(input_tokens)  # (batch_size, seq_len, embed_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x) if self.dropout else x
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask, padding_value=padding_value)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int,
                 dropout: float | None = 0.1):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.cross_attention = SelfAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                look_ahead_mask: torch. Tensor | None = None,
                padding_mask: torch. Tensor | None =None,
                padding_value: float = 0):
        attn_output, _ = self.self_attention(x, x, x, look_ahead_mask)
        x += self.dropout(attn_output) if self.dropout else attn_output
        x = self.norm1(x)
        cross_attn_output, _ = self.cross_attention(
            query=x, key=encoder_output, value=encoder_output,
            padding_mask=padding_mask, padding_value=padding_value
        )
        x += self.dropout(cross_attn_output) if self.dropout else cross_attn_output 
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x += self.dropout(ff_output) if self.dropout else ff_output
        x = self.norm3(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_layers: int, dim_feedforward: int, max_len: int = 5000,
                 dropout: float | None  = 0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(embed_dim, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, target_tokens: torch.Tensor,
                encoder_output: torch.Tensor,
                look_ahead_mask: torch.Tensor | None = None,
                padding_mask: torch.Tensor | None = None,
                padding_value:float = 0):
        x = self.embedding(target_tokens)  # (batch_size, seq_len, embed_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x) if self.dropout else x
        for layer in self.layers:
            x = layer(x, encoder_output=encoder_output,
                      look_ahead_mask=look_ahead_mask,
                      padding_mask=padding_mask, padding_value=padding_value)
        return x


def create_padding_mask(input_tokens, pad_token=0):
    """input_tokens: shape (batch_size, seq_len)"""
    # mask.shape() = (batch_size, 1, 1, seq_len)
    mask = (input_tokens != pad_token).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(seq_len: int, device: int | str | device = "cpu"):
    """Create a mask where each position i can only attend to positions <= i"""
    # mase.shape() = (1, 1, seq_len, seq_len)
    mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).unsqueeze(0)
    return mask.to(device)


if __name__ == "__main__":
    # Define the parameters
    src_vocab_size = 10000  # Source vocabulary size
    tgt_vocab_size = 10000  # Target vocabulary size
    embed_dim = 512            # Embedding size
    num_heads = 8              # Number of attention heads
    num_encoder_layers = 6   # Number of encoder layers
    num_decoder_layers = 6   # Number of decoder layers
    dim_feedforward = 2048    # Feedforward network size
    max_len = 500            # Maximum sequence length
    dropout = 0.1            # Dropout rate

    # Create dummy input and target sequences
    batch_size = 32
    input_tokens = torch.randint(0, src_vocab_size, (batch_size, 25))
    target_tokens = torch.randint(0, tgt_vocab_size, (batch_size, 30))

    # Create padding masks (assuming no padding here; using all ones)
    # Create look-ahead mask for the target sequence
    src_padding_mask = create_padding_mask(input_tokens)
    tgt_look_ahead_mask = create_look_ahead_mask(target_tokens.size(1))

    #--------------------------------------------------
    # Transfomer piece by piece
    #--------------------------------------------------
    encoder = TransformerEncoder(
        vocab_size=src_vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
        max_len=max_len
    )
    encoder_output = encoder(
        input_tokens=input_tokens, padding_mask=src_padding_mask, padding_value=0
    )
    encoder_output.shape

    # decoder
    decoder = TransformerDecoder(
        vocab_size=tgt_vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
        max_len=max_len, dropout=dropout
    )
    decoder_output = decoder(
        target_tokens=target_tokens, encoder_output=encoder_output,
        look_ahead_mask=tgt_look_ahead_mask, padding_mask=src_padding_mask,
        padding_value=0
    )
    decoder_output.shape

    #fc output layer
    fc = nn.Linear(embed_dim, tgt_vocab_size)
    fc_output = fc(decoder_output)
    fc_output.shape

    #--------------------------------------------------
    # Transfomer model
    #--------------------------------------------------
    # Transformer
    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, embed_dim, num_heads, num_encoder_layers,
        num_decoder_layers, dim_feedforward, max_len, dropout
    )

    output = transformer(
        input_tokens=input_tokens, target_tokens=target_tokens,
        src_padding_mask=src_padding_mask,
        tgt_look_ahead_mask=tgt_look_ahead_mask
    )
    # Expected shape: (batch_size, tgt_seq_len, tgt_vocab_size)
    print("Output shape:", output.shape)
