import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformer import (
    Transformer, 
    create_padding_mask, 
    create_look_ahead_mask
)


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
    d_model=d_model, n_heads=n_heads, num_encoder_layers=num_encoder_layers, 
    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, 
    max_len=max_len_tgt, dropout=dropout
).to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    _ = model.train()
    
    # Generate a batch of data
    src, tgt = create_batch(batch_size=32, sos_token=sos_token, eos_token=eos_token)
    
    # Create tgt_input and tgt_output
    tgt_input = tgt[:, :-1]  # Remove <EOS> token
    tgt_output = tgt[:, 1:]  # Shift by one, so we predict <EOS>

    # Create masks
    src_padding_mask = create_padding_mask(src)
    tgt_padding_mask = create_padding_mask(tgt_input)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1), device)
    
    # Forward pass
    output = model(src, tgt_input, src_padding_mask, tgt_padding_mask, tgt_look_ahead_mask)
    
    # Reshape output and target for cross-entropy loss
    output = output.reshape(-1, tgt_vocab_size)
    tgt_output = tgt_output.reshape(-1)

    # Compute loss
    loss = criterion(output, tgt_output)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# inference_example
src_seq = [6, 8, 10, 1, 14]  # Example input sequence
desired_seq = [8, 10, 12, 14, 16]  # Example input sequence

model.eval()
src_tensor = torch.tensor([src_seq], dtype=torch.long).to(device)
generated_seq = model.inference(
    src_tensor, max_len=max_len_tgt, sos_token=1, eos_token=3
)
generated_seq = generated_seq.cpu().numpy()

print(f"Input sequence: {src_seq}")
print(f"Desired sequence: {desired_seq}")
print(f"Generated sequence: {generated_seq}")
