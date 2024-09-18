import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, 
                 num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        # Encoder
        self.encoder = TransformerEncoder(src_vocab_size, d_model, n_heads, num_encoder_layers, dim_feedforward, max_len, dropout)
        
        # Decoder
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, n_heads, num_decoder_layers, dim_feedforward, max_len, dropout)
        
        # Final linear layer that projects decoder's output to the target vocabulary size
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Softmax is typically applied later during inference or training (like using cross-entropy loss)
    
    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None, 
                tgt_look_ahead_mask=None):
        # Encode the source sequence
        encoder_output = self.encoder(src, src_padding_mask)
        
        # Decode the target sequence with attention to the encoder output
        decoder_output = self.decoder(tgt, encoder_output, tgt_look_ahead_mask, tgt_padding_mask)
        
        # Project the decoder's output to the target vocabulary space
        output = self.fc_out(decoder_output)  # Shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        
        return output

 
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head (same for Q, K, and V)

        # Linear layers for projecting Q, K, V
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # Output linear layer
        self.out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, padding_mask=None):
        batch_size, seq_len_q, d_model = Q.size()
        seq_len_k = K.size(1)  # Sequence length for K (which could differ from Q)

        # Step 1: Project Q, K, V
        Q = self.query(Q)  # (batch_size, seq_len_q, d_model)
        K = self.key(K)    # (batch_size, seq_len_k, d_model)
        V = self.value(V)  # (batch_size, seq_len_k, d_model)

        # Step 2: Reshape into (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1))
        attn_weights /= torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # Step 4: Apply the padding mask if provided
        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(padding_mask == 0, float('-inf'))

        # Step 5: Apply softmax to get attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Step 6: Weighted sum of the values
        attention_output = torch.matmul(attn_weights, V)

        # Step 7: Concatenate the heads and pass through output linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, d_model
        )
        output = self.out(attention_output)
        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        # Self-attention layer (with mask)
        self.self_attention = SelfAttention(d_model, n_heads)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        # Step 1: Self-attention layer with residual connection and layer normalization
        # attn_output.shape() = (batch_size, seq_len, d_model)
        attn_output = self.self_attention(x, x, x, padding_mask)  
        x = self.norm1(x + self.dropout(attn_output))

        # Step 2: Feed-forward network with residual connection and layer normalization
        ff_output = self.feed_forward(x)  # (batch_size, seq_len, d_model)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model) for positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + self.pe[:, :x.size(1), :].clone().detach()
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, 
                 dim_feedforward, max_len=5000, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Transformer encoder blocks (stack of layers)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, n_heads, dim_feedforward, dropout) 
                for _ in range(num_layers)
            ]
        )

        # Dropout after embedding + positional encoding
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tokens, padding_mask=None):
        # Step 1: Embed the input tokens
        x = self.embedding(input_tokens)  # (batch_size, seq_len, d_model)

        # Step 2: Add positional encoding
        x = self.positional_encoding(x)

        # Step 3: Apply dropout
        x = self.dropout(x)

        # Step 4: Pass through the encoder layers
        for layer in self.layers:
            x = layer(x, padding_mask)

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        # Masked self-attention for the decoder
        self.self_attention = SelfAttention(d_model, n_heads)
        # Cross-attention (encoder-decoder attention)
        self.cross_attention = SelfAttention(d_model, n_heads)
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        # Step 1: Masked self-attention with residual connection and layer normalization
        attn_output = self.self_attention(x, x, x, look_ahead_mask)  
        x = self.norm1(x + self.dropout(attn_output))

        # Step 2: Cross-attention with residual connection and layer normalization
        cross_attn_output = self.cross_attention(
            x, encoder_output, encoder_output, padding_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Step 3: Feed-forward network with residual connection and layer normalization
        ff_output = self.feed_forward(x)  
        x = self.norm3(x + self.dropout(ff_output))

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, 
                 dim_feedforward, max_len=5000, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        # Embedding layer for input tokens (target sequence)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding for target sequences
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, n_heads, dim_feedforward, dropout) 
                for _ in range(num_layers)
            ]
        )

        # Dropout after embedding + positional encoding
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_tokens, encoder_output, look_ahead_mask=None, 
                padding_mask=None):
        # Step 1: Embed the target tokens
        x = self.embedding(target_tokens)  # (batch_size, seq_len, d_model)

        # Step 2: Add positional encoding
        x = self.positional_encoding(x)

        # Step 3: Apply dropout
        x = self.dropout(x)

        # Step 4: Pass through each decoder block
        for layer in self.layers:
            x = layer(x, encoder_output, look_ahead_mask, padding_mask)

        return x


def create_padding_mask(input_tokens, pad_token=0):
    """input_tokens: shape (batch_size, seq_len)"""
    # mask.shape() = (batch_size, 1, 1, seq_len)
    mask = (input_tokens != pad_token).unsqueeze(1).unsqueeze(2)  
    return mask


def create_look_ahead_mask(seq_len):
    """Create a mask where each position i can only attend to positions <= i"""
    # mase.shape() = (1, 1, seq_len, seq_len)
    mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).unsqueeze(0) 
    return mask


# Hyperparameters
src_vocab_size = 10000  # Source vocabulary size
tgt_vocab_size = 10000  # Target vocabulary size
d_model = 512  # Embedding size
n_heads = 8  # Number of attention heads
num_encoder_layers = 6  # Number of encoder layers
num_decoder_layers = 6  # Number of decoder layers
dim_feedforward = 2048  # Feedforward network size
max_len = 500  # Maximum sequence length
dropout = 0.1  # Dropout rate

# Create the Transformer model
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, 
                    num_encoder_layers, num_decoder_layers, 
                    dim_feedforward, max_len, dropout)


# Example input (source and target sequences)
src = torch.randint(0, src_vocab_size, (32, 50))  # Source: (batch_size, src_seq_len)
tgt = torch.randint(0, tgt_vocab_size, (32, 50))  # Target: (batch_size, tgt_seq_len)

out = model.encoder(src)

out.shape
model.decoder(tgt, out).shape

src.shape

# Padding masks (for both source and target)
src_padding_mask = create_padding_mask(src)
tgt_padding_mask = create_padding_mask(tgt)

# Look-ahead mask for the target sequence
tgt_look_ahead_mask = create_look_ahead_mask(tgt.size(1))

# Forward pass through the Transformer
output = model(src, tgt, src_padding_mask, tgt_padding_mask, tgt_look_ahead_mask)

print(output.shape)  # Output shape: (batch_size, tgt_seq_len, tgt_vocab_size)
