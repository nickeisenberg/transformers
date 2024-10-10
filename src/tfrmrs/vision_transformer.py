import torch
import torch.nn.functional as F
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear embedding of image patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Shape of x: (batch_size, channels, height, width)
        x = self.proj(x)  # Shape: (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # Flatten height and width into patches
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for CLS token

    def forward(self, x):
        return x + self.pos_embed


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
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
        
    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # Compute the attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        attn_scores /= torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # Scale by √d_k

        # Apply the padding mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout to the attention weights
        attn_weights = self.dropout(attn_weights)
        
        # Compute the weighted sum of the values
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projection to compute Q, K, and V
        Q = self.q_proj(query)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(key)    # (batch_size, seq_len, embed_dim)
        V = self.v_proj(value)  # (batch_size, seq_len, embed_dim)

        # Reshape for multi-head attention (split into multiple heads)
        # New shape: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Perform scaled dot-product attention for each head, with optional padding mask
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate the attention outputs from all heads
        # New shape: (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Apply the final linear layer to combine the outputs from all heads
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Apply multi-head self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # Apply MLP
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)
        
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder layers
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                embed_dim, num_heads, mlp_ratio, dropout
            ) for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Get patch embeddings
        x = self.patch_embed(x)  # Shape: (batch_size, num_patches, embed_dim)
        
        # Add the class token at the beginning
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch_size, num_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = self.pos_embed(x)
        
        # Pass through transformer encoder layers
        for encoder in self.encoders:
            x = encoder(x)
        
        # Final classification head
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Extract the CLS token (first token)
        x = self.head(cls_token_final)  # Output classification logits
        return x


if __name__ == "__main__":
    # Create a dummy input tensor (batch_size, channels, height, width)
    x = torch.randn(8, 144, 224, 224)
    
    # Instantiate the Vision Transformer
    vit = VisionTransformer(in_chans=144)
    
    # Get the classification logits
    logits = vit(x)
    print(logits.shape)  # Should output: (batch_size, num_classes)
