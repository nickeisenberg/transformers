from sys import path
path.append(__file__.split("tests")[0])

import torch
from torch.cuda import is_available as cuda_avail
from torch.mps import is_available as mps_avail

from src.tfrmrs.transformer import (
    SelfAttention,
    create_look_ahead_mask
)

create_look_ahead_mask(10)

if mps_avail():
    DEVICE = "mps"
elif cuda_avail():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

def test_SelfAttention():
    batch_size = 4
    seq_len = 128
    embed_dim = 64
    num_heads = 2
    
    queries = torch.randn((batch_size, seq_len, embed_dim)).to(DEVICE)
    keys = torch.randn((batch_size, seq_len, embed_dim)).to(DEVICE)
    values = torch.randn((batch_size, seq_len, embed_dim)).to(DEVICE)
    
    self_attention = SelfAttention(embed_dim=embed_dim, num_heads=num_heads).to(DEVICE)
    
    output, _ = self_attention(
        queries, keys, values
    )
    
    assert output.shape == torch.Size([batch_size, seq_len, embed_dim])
