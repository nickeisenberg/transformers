import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformer import (
    Transformer, 
    create_padding_mask, 
    create_look_ahead_mask
)

# Define the Transformer model
# (Assuming the previously defined Transformer model is used here)

# Set device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

# Hyperparameters
src_vocab_size = 50  # Let's assume the even numbers are between 2 and 50
tgt_vocab_size = 50
d_model = 64
n_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 256
max_len_src = 5
max_len_tgt = 6
dropout = 0.1

# Instantiate the transformer
model = Transformer(
    src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, 
    d_model=d_model, n_heads=n_heads, num_encoder_layers=num_encoder_layers, 
    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, 
    max_len=6, dropout=dropout
).to(device)


# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Generate a dataset of even numbers for training


def generate_even_data():
    x = random.randint(0, 100)
    if x % 2 == 0:
        x = x
    else:
        x = x + 1
    
    src = [x]
    trg = [x]
    for _ in range(4):
        src.append(src[-1] + 2)
        trg.append(trg[-1] + 2)

    trg.append(trg[-1] + 2)

    return src, trg


def create_batch(batch_size=32):
    src_batch = []
    tgt_batch = []
    for _ in range(batch_size):
        src_seq, tgt_seq = generate_even_data()
        src_batch.append(src_seq)
        tgt_batch.append(tgt_seq)
    
    src_batch = torch.tensor(src_batch, dtype=torch.long).to(device)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.long).to(device)
    return src_batch, tgt_batch

inps, outs = create_batch()

# Training loop
num_epochs = 200

src, tgt = create_batch(batch_size=32)

src.shape
tgt.shape

# Shift target sequences for training

# Create masks
src_padding_mask = create_padding_mask(src)
tgt_padding_mask = create_padding_mask(tgt)
tgt_look_ahead_mask = create_look_ahead_mask(tgt.size(1), device)

# Forward pass
output = model(src, tgt, src_padding_mask, tgt_padding_mask, tgt_look_ahead_mask)

for epoch in range(num_epochs):
    model.train()
    
    # Generate a batch of data
    src, tgt = create_batch(batch_size=32)
    
    # Shift target sequences for training

    # Create masks
    src_padding_mask = create_padding_mask(src)
    tgt_padding_mask = create_padding_mask(tgt)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt.size(1), device)
    
    # Forward pass
    output = model(src, tgt, src_padding_mask, tgt_padding_mask, tgt_look_ahead_mask)
    
    # Reshape output and target for cross-entropy loss
    output = output.reshape(-1, tgt_vocab_size)
    tgt_output = tgt.reshape(-1)

    # Compute loss
    loss = criterion(output, tgt_output)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Inference to test the model
model.eval()

def inference_example(src_seq):
    src_tensor = torch.tensor([src_seq], dtype=torch.long).to(device)
    generated_seq = model.inference(
        src_tensor, max_len=max_len_tgt, sos_token=src_seq[0], eos_token=0
    )
    return generated_seq.cpu().numpy()

# Test with a simple example
src_seq = [2, 4, 6]  # Example input sequence

generated_seq = inference_example(src_seq)

print(f"Input sequence: {src_seq}")
print(f"Generated sequence: {generated_seq}")

