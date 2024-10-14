import torch
from transformers import MarianTokenizer
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.tfrmrs.transformer import (
    Transformer, 
    create_look_ahead_mask,
)


def get_tokenizers():
    src_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    tgt_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

    src_tokenizer.add_special_tokens({"bos_token": "<s>"})
    tgt_tokenizer.add_special_tokens({"bos_token": "<s>"})

    return src_tokenizer, tgt_tokenizer


class TranslationDataset(Dataset):
    def __init__(self, path_to_df, src_tokenizer, tgt_tokenizer, max_length=64):
        self.df = pd.read_csv(path_to_df)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_text = self.df.iloc[idx, 0]  # English text (source)
        tgt_text = self.df.iloc[idx, 1]  # Spanish text (target)

        # Manually add the <s> token at the beginning of the source and target texts
        tgt_text = f"<s> {tgt_text}"  # Prepend <s> and append </s> to the target text

        # Tokenize the source text (English)
        src_encoded = self.src_tokenizer(
            src_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Tokenize the target text (Spanish)
        tgt_encoded = self.tgt_tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": src_encoded["input_ids"].squeeze(),  # English tokens
            "attention_mask": src_encoded["attention_mask"].squeeze(),  # Attention mask
            "labels": tgt_encoded["input_ids"].squeeze()  # Spanish tokens (target)
        }


src_tokenizer, tgt_tokenizer = get_tokenizers()

src_vocab_size = src_tokenizer.vocab_size + 1
tgt_vocab_size = tgt_tokenizer.vocab_size + 1

path_to_df = os.path.expanduser("~/datasets/en_to_esp/data.csv")
dataset = TranslationDataset(path_to_df=path_to_df, 
                             src_tokenizer=src_tokenizer, 
                             tgt_tokenizer=tgt_tokenizer, 
                             max_length=64)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

transformer_model = Transformer(
    src_vocab_size=src_vocab_size, 
    tgt_vocab_size=tgt_vocab_size, 
    d_model=512, 
    n_heads=8, 
    num_encoder_layers=6, 
    num_decoder_layers=6, 
    dim_feedforward=2048, 
    max_len=64, 
    dropout=0.1
)

optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.0001)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=src_tokenizer.pad_token_id)

num_epochs = 10
for epoch in range(num_epochs):
    _ = transformer_model.train()
    total_loss = 0

    for batch in dataloader:
        # Get inputs and target labels
        src = batch['input_ids']  # Shape: (batch_size, src_seq_len)
        src_padding_mask = batch['attention_mask'].unsqueeze(1).unsqueeze(2)
        tgt = batch['labels']     # Shape: (batch_size, tgt_seq_len)
        tgt_input = tgt[:, :-1]   # Exclude the last token for input
        tgt_output = tgt[:, 1:]   # Shift target by one for prediction
        
        # Create the padding and look-ahead masks
        tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1))

        # Forward pass through the transformer
        outputs = transformer_model(
            src, tgt_input, 
            src_padding_mask=src_padding_mask, 
            tgt_look_ahead_mask=tgt_look_ahead_mask
        )

        
        # Reshape outputs and labels for computing loss
        outputs = outputs.reshape(-1, tgt_vocab_size)  # Flatten the output
        tgt_output = tgt_output.view(-1)  # Flatten target output
        
        # Compute loss
        loss = loss_fn(outputs, tgt_output)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")
