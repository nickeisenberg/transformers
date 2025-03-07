from src.tfrmrs.transformer import (
    create_padding_mask,
    create_look_ahead_mask
)

from examples.next_even_number.src.data import (
    create_batch,
)

def train_one_epoch(model, criterion, optimizer, epoch):
    _ = model.train()

    # Generate a batch of data
    src, tgt = create_batch(batch_size=32, sos_token=sos_token, eos_token=eos_token)

    # Create tgt_input and tgt_output
    tgt_input = tgt[:, :-1]  # Remove <EOS> token
    tgt_output = tgt[:, 1:]  # Shift by one, so we predict <EOS>

    # Create masks
    src_padding_mask = create_padding_mask(src)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1), device)

    # Forward pass
    output = model(
        input_tokens=src, target_tokens=tgt_input,
        src_padding_mask=src_padding_mask,
        tgt_look_ahead_mask=tgt_look_ahead_mask
    )

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
