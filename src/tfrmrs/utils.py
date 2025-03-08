import torch
from torch import device


def fake_batch_of_src_tokens(batch_size: int, seq_len: int, vocab_size: int, 
                             padding_value: int = 0):
    batch = []
    for _ in range(batch_size):
        idx = torch.randint(0, seq_len, (1,)).item()
        tokens = torch.zeros(seq_len) + padding_value
        tokens[:idx] = torch.randint(0, vocab_size, (int(idx),))
        batch.append(tokens)
    return torch.vstack(batch)


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
