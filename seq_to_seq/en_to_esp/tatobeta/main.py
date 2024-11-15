import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import MarianTokenizer
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.tfrmrs.transformer import (
    Transformer, 
    create_padding_mask,
    create_look_ahead_mask,
)


class TranslationDataset(Dataset):
    def __init__(self, path_to_src, path_to_tgt, tokenizer, max_length=250):
        # Load source and target texts
        self.src_text = [line.strip() for line in open(path_to_src)]
        self.tgt_text = [line.strip() for line in open(path_to_tgt)]
        
        if len(self.src_text) != len(self.tgt_text):
            raise ValueError("Source and target files must have the same number of lines.")

        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.src_text)

    def __getitem__(self, idx):
        src_text, tgt_text = self.src_text[idx], self.tgt_text[idx]

        # Add <s> token at the start of the target text for decoder input
        tgt_text = f"<s> {tgt_text}"

        # Tokenize the source text
        src_encoded = self.tokenizer(
            src_text, 
            max_length=self.max_length, 
            padding='do_not_pad', 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Tokenize the target text
        tgt_encoded = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='do_not_pad', 
            truncation=True,
            return_tensors="pt"
        )

        # Ensure tensors are squeezed correctly
        input_ids = src_encoded["input_ids"].squeeze(0)
        attention_mask = src_encoded["attention_mask"].squeeze(0)
        decoder_input_ids = tgt_encoded["input_ids"].squeeze(0)[:-1]
        labels = tgt_encoded["input_ids"].squeeze(0)[1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        }

def get_tokenizer():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    tokenizer.add_special_tokens({"bos_token": "<s>"})
    return tokenizer

def get_dataset(tokenizer):
    train_src = os.path.expanduser("~/datasets/en_to_esp/tatobeta/train_en.txt")
    train_tgt = os.path.expanduser("~/datasets/en_to_esp/tatobeta/train_es.txt")
    val_src = os.path.expanduser("~/datasets/en_to_esp/tatobeta/val_en.txt")
    val_tgt = os.path.expanduser("~/datasets/en_to_esp/tatobeta/val_es.txt")
    tokenizer = get_tokenizer()
    train_dataset = TranslationDataset(train_src, train_tgt, tokenizer)
    val_dataset = TranslationDataset(val_src, val_tgt, tokenizer)
    return train_dataset, val_dataset 

def get_dataloader(dataset):
    def pad(l):
        M = max([x.shape[0] for x in l])
        return torch.stack([
            torch.nn.functional.pad(x, (0, M - x.shape[0]), value=65000) for x in l
        ])

    def collate_fn(batch):
        inputs = pad([
            x["input_ids"] for x in batch
        ])
        decoder_inputs = pad([
            x["decoder_input_ids"] for x in batch
        ])
        targets = pad([
            x["labels"] for x in batch
        ])
        return inputs, decoder_inputs, targets

    return DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )

def train_loop(transformer, dataloader, criterion, optimizer, device="cpu"):
    running_loss = 0
    avg_loss = 0
    idx = 0
    pbar = tqdm(dataloader)
    for input_tokens, decoder_input_ids, labels in pbar:
        idx += 1
        input_tokens = input_tokens.to(device) 
        decoder_input_ids = decoder_input_ids.to(device)
        labels = labels.to(device)
    
        src_padding_mask = create_padding_mask(input_tokens, pad_token=65000).to(device)  
        tgt_look_ahead_mask = create_look_ahead_mask(decoder_input_ids.size(1)).to(device)
        
        optimizer.zero_grad()
        
        output = transformer(
            input_tokens=input_tokens, target_tokens=decoder_input_ids,
            src_padding_mask=src_padding_mask, src_padding_token=65000,
            tgt_look_ahead_mask=tgt_look_ahead_mask
        )
        
        loss = criterion(output, labels)
    
        running_loss += loss.item()
        
        loss.backward()
        
        optimizer.step()
    
        if idx % 25:
            avg_loss = running_loss / (idx)
            pbar.set_postfix(loss=avg_loss)

    return avg_loss

def val_loop(transformer, dataloader, criterion, device="cpu"):
    running_loss = 0
    avg_loss = 0
    idx = 0
    pbar = tqdm(dataloader)
    for input_tokens, decoder_input_ids, labels in pbar:
        idx += 1
        input_tokens = input_tokens.to(device) 
        decoder_input_ids = decoder_input_ids.to(device)
        labels = labels.to(device)
    
        src_padding_mask = create_padding_mask(input_tokens, pad_token=65000).to(device)  
        tgt_look_ahead_mask = create_look_ahead_mask(decoder_input_ids.size(1)).to(device)
        
        output = transformer(
            input_tokens=input_tokens, target_tokens=decoder_input_ids,
            src_padding_mask=src_padding_mask, src_padding_token=65000,
            tgt_look_ahead_mask=tgt_look_ahead_mask
        )
        
        loss = criterion(output, labels)
    
        running_loss += loss.item()
    
        if idx % 25:
            avg_loss = running_loss / (idx)
            pbar.set_postfix(loss=avg_loss)

    return avg_loss

tokenizer = get_tokenizer()
train_dataset, val_dataset = get_dataset(tokenizer)
train_dataloader = get_dataloader(train_dataset)
val_dataloader = get_dataloader(val_dataset)

src_vocab_size = tokenizer.vocab_size + 1
tgt_vocab_size = tokenizer.vocab_size + 1
pad_token = tokenizer.encode("<pad>")[0]
embed_dim = 256
n_heads = 4
num_encoder_layers = 4
num_decoder_layers = 4
dim_feedforward = 2048
max_len = 500
dropout = 0.1
epochs = 2

transformer = Transformer(
    src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
    embed_dim=embed_dim, n_heads=n_heads, num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, 
    max_len=max_len, dropout=dropout
).to(0)

def criterion(output, labels):
    return nn.CrossEntropyLoss(ignore_index=pad_token)(
        output.view(-1, output.shape[-1]), labels.view(-1)
    )

optimizer = Adam(transformer.parameters(), lr=1e-4)

for epoch in range(epochs):
    transformer.train()
    train_loss = train_loop(transformer, train_dataloader, criterion, optimizer, "cuda")
    print(f"EPOCH {epoch + 1} Train Loss", train_loss)
    transformer.eval()
    val_loss = val_loop(transformer, val_dataloader, criterion, "cuda")
    print(f"EPOCH {epoch + 1} Val Loss", val_loss)

tokenizer.decode(
    transformer.inference(
        tokenizer.encode("where is the library", return_tensors="pt").to(0), 
        max_len=20, sos_token=65001, eos_token=0
    )[0].to("cpu")
)
