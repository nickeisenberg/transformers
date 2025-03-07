import torch
import random


def generate_even_data(sos_token, eos_token):
    x = random.randint(0, 100)
    if x % 2 == 0:
        x = x
    else:
        x = x + 1
    
    src = [x]
    trg = [sos_token]  # Start with <SOS> token
    for _ in range(4):
        src.append(src[-1] + 2)
        trg.append(src[-2] + 2)
    trg.append(src[-1] + 2)
    trg.append(eos_token)  # End with <EOS> token

    return src, trg


def create_batch(batch_size=32, sos_token=-5, eos_token=-1):
    src_batch = []
    tgt_batch = []
    for _ in range(batch_size):
        src_seq, tgt_seq = generate_even_data(sos_token, eos_token)
        src_batch.append(src_seq)
        tgt_batch.append(tgt_seq)
    
    src_batch = torch.tensor(src_batch, dtype=torch.long).to(device)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.long).to(device)
    return src_batch, tgt_batch
