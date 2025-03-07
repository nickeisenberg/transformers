import torch
import torch.nn as nn
import torch.optim as optim

from examples.next_even_number.src.train import (
    train_one_epoch
)

from src.tfrmrs.transformer import (
    Transformer,
)


# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_vocab_size = 200  # Let's assume the even numbers are between 2 and 50
tgt_vocab_size = 200
d_model = 64
n_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 256
max_len_src = 5
max_len_tgt = 6  # Account for <SOS> and <EOS>
dropout = 0.1
sos_token = 1  # Define <SOS> token ID
eos_token = 3  # Define <EOS> token ID

# Instantiate the transformer
model = Transformer(
    src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, 
    embed_dim=d_model, n_heads=n_heads, num_encoder_layers=num_encoder_layers, 
    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, 
    max_len=100, dropout=dropout
).to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    train_one_epoch(model, criterion, optimizer, epoch)

# inference_example
src_seq = [ 
    [6, 8, 10, 12, 14],
]
desired_seq = [[x + 2 for x in xx] for xx in src_seq]

model.eval()
src_tensor = torch.tensor(src_seq, dtype=torch.long).to(device)
generated_seq = model.inference(
    src_tensor, max_len=max_len_tgt, sos_token=1, eos_token=3
)
generated_seq = generated_seq.cpu().numpy()

print(f"Input sequence: {src_seq}")
print(f"Generated sequence: {generated_seq}")
print(f"Desired sequence: {desired_seq}")
