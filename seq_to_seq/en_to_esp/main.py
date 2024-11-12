import torch
import torch.nn as nn
from torch.optim import Adam
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
    def __init__(self, path_to_df, tokenizer, max_length=64):
        self.df = pd.read_csv(path_to_df)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_text = self.df.iloc[idx, 0]  # English text (source)
        tgt_text = self.df.iloc[idx, 1]  # Spanish text (target)

        # Manually add the <s> token at the beginning of the source and target texts
        tgt_text = f"<s> {tgt_text}"  # Prepend <s> and append </s> to the target text

        # Tokenize the source text (English)
        src_encoded = self.tokenizer(
            src_text, 
            max_length=self.max_length, 
            padding='do_not_pad', 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Tokenize the target text (Spanish)
        tgt_encoded = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='do_not_pad', 
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": src_encoded["input_ids"].squeeze(),
            "attention_mask": src_encoded["attention_mask"].squeeze(),
            "decoder_input_ids": tgt_encoded["input_ids"].squeeze()[:-1],
            "labels": tgt_encoded["input_ids"].squeeze()[1:]
        }

def get_tokenizer():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    tokenizer.add_special_tokens({"bos_token": "<s>"})
    return tokenizer

def get_dataset(tokenizer):
    path_to_df = os.path.expanduser("~/datasets/en_to_esp/data.csv")
    dataset = TranslationDataset(path_to_df=path_to_df, 
                                 tokenizer=tokenizer, 
                                 max_length=64)
    return dataset

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
        dataset, batch_size=10, shuffle=True, collate_fn=collate_fn
    )

def train_loop(transformer, dataloader, criterion, optimizer):
    avg_loss = 0
    for i, (input_tokens, decoder_input_ids, labels) in enumerate(dataloader):
    
        src_padding_mask = create_padding_mask(input_tokens, pad_token=65000)  
        tgt_look_ahead_mask = create_look_ahead_mask(decoder_input_ids.size(1))
        
        optimizer.zero_grad()
        
        output = transformer(
            input_tokens=input_tokens, target_tokens=decoder_input_ids,
            src_padding_mask=src_padding_mask, src_padding_token=65000,
            tgt_look_ahead_mask=tgt_look_ahead_mask
        )
        
        loss = criterion(output, labels)
    
        avg_loss += loss.item()
        
        loss.backward()
        
        optimizer.step()
    
        if i % 25:
            print(avg_loss / (i + 1))

tokenizer = get_tokenizer()
dataset = get_dataset(tokenizer)
dataloader = get_dataloader(dataset)

src_vocab_size = tokenizer.vocab_size + 1
tgt_vocab_size = tokenizer.vocab_size + 1
pad_token = tokenizer.encode("<pad>")[0]
embed_dim = 512
n_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
max_len = 500
dropout = 0.1

transformer = Transformer(
    src_vocab_size, tgt_vocab_size, embed_dim, n_heads, num_encoder_layers,
    num_decoder_layers, dim_feedforward, max_len, dropout
)

def criterion(output, labels):
    return nn.CrossEntropyLoss(ignore_index=pad_token)(
        output.view(-1, output.shape[-1]), labels.view(-1)
    )

optimizer = Adam(transformer.parameters(), lr=1e-4)

train_loop(transformer, dataloader, criterion, optimizer)
