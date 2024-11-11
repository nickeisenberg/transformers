from tqdm import tqdm
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
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    tokenizer.add_special_tokens({"bos_token": "<s>"})
    return tokenizer


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
            "input_ids": src_encoded["input_ids"].squeeze(),  # English tokens
            "attention_mask": src_encoded["attention_mask"].squeeze(),  # Attention mask
            "labels": tgt_encoded["input_ids"].squeeze()  # Spanish tokens (target)
        }


tokenizer = get_tokenizers()
src_vocab_size = tokenizer.vocab_size + 1
tgt_vocab_size = tokenizer.vocab_size + 1

path_to_df = os.path.expanduser("~/datasets/en_to_esp/data.csv")
dataset = TranslationDataset(path_to_df=path_to_df, 
                             tokenizer=tokenizer, 
                             max_length=64)

def collate_fn(batch):
    def pad(l):
        M = max([x.shape[0] for x in l])
        return torch.stack([
            torch.nn.functional.pad(x, (0, M - x.shape[0]), value=-1) for x in l
        ])

    inputs = pad([
        x["input_ids"] for x in batch
    ])
    targets = pad([
        x["labels"] for x in batch
    ])

    return inputs, targets


dataloader = DataLoader(
    dataset, batch_size=10, shuffle=True, collate_fn=collate_fn
)

next(iter(dataloader))
