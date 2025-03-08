from sys import path
path.append(__file__.split("tests")[0])

import torch
from torch.cuda import is_available as cuda_avail
from torch.mps import is_available as mps_avail

from src.tfrmrs.transformer import (
    SelfAttention,
    TransformerEncoder,
    TransformerDecoder,
    Transformer,
    create_look_ahead_mask,
    create_padding_mask
)

if mps_avail():
    DEVICE = "mps"
elif cuda_avail():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

batch_size = 4
src_vocab_size = 10000
tgt_vocab_size = 10000
seq_len = 128
embed_dim = 512 
num_heads = 8 
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
max_len = 500
dropout = 0.1

input_tokens = torch.randint(0, src_vocab_size, (batch_size, seq_len)).to(DEVICE)
target_tokens = torch.randint(0, tgt_vocab_size, (batch_size, seq_len)).to(DEVICE)
src_padding_mask = create_padding_mask(input_tokens)
tgt_look_ahead_mask = create_look_ahead_mask(
    target_tokens.size(1), device=DEVICE
)

queries = torch.randn((batch_size, seq_len, embed_dim)).to(DEVICE)
keys = torch.randn((batch_size, seq_len, embed_dim)).to(DEVICE)
values = torch.randn((batch_size, seq_len, embed_dim)).to(DEVICE)
encoder_output = torch.randn((batch_size, seq_len, embed_dim)).to(DEVICE)

encoder = TransformerEncoder(
    vocab_size=src_vocab_size, embed_dim=embed_dim, num_heads=num_heads,
    num_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
    max_len=max_len
)
_ = encoder.to(DEVICE)

decoder = TransformerDecoder(
    vocab_size=tgt_vocab_size, embed_dim=embed_dim, num_heads=num_heads,
    num_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
    max_len=max_len, dropout=dropout
)
_ = decoder.to(DEVICE)

transformer = Transformer(
    src_vocab_size, tgt_vocab_size, embed_dim, num_heads, num_encoder_layers,
    num_decoder_layers, dim_feedforward, max_len, dropout
)
_ = transformer.to(DEVICE)

def test_SelfAttention():
    self_attention = SelfAttention(embed_dim=embed_dim, num_heads=num_heads).to(DEVICE)
    output, _ = self_attention(
        queries, keys, values
    )
    assert output.shape == torch.Size([batch_size, seq_len, embed_dim])

def test_TransformerEncoder():
    encoder_output = encoder(
        input_tokens=input_tokens, padding_mask=src_padding_mask, padding_value=0
    )
    assert encoder_output.shape == torch.Size([batch_size, seq_len, embed_dim])

def test_TransformerDecoder():
    decoder_output = decoder(
        target_tokens=target_tokens, encoder_output=encoder_output,
        look_ahead_mask=tgt_look_ahead_mask, padding_mask=src_padding_mask,
        padding_value=0
    )
    assert decoder_output.shape == torch.Size([batch_size, seq_len, embed_dim])

def test_Transformer():
    output = transformer(
        input_tokens=input_tokens, target_tokens=target_tokens,
        src_padding_mask=src_padding_mask,
        tgt_look_ahead_mask=tgt_look_ahead_mask
    )
    assert output.shape == torch.Size([batch_size, seq_len, tgt_vocab_size])
