import torch
from transformers import MarianTokenizer
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.tfrmrs.transformer import Transformer


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
path_to_df = os.path.expanduser("~/datasets/en_to_esp/data.csv")
dataset = TranslationDataset(path_to_df, src_tokenizer, tgt_tokenizer)
loader = DataLoader(dataset, 32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
src_vocab_size = src_tokenizer.vocab_size
tgt_vocab_size = tgt_tokenizer.vocab_size
d_model = 64
n_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 256
max_len_src = 64
max_len_tgt = 64 # Account for <SOS> and <EOS>
dropout = 0.1
sos_token = 65001  # Define <SOS> token ID
eos_token = 65000  # Define <EOS> token ID

model = Transformer(
    src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, 
    d_model=d_model, n_heads=n_heads, num_encoder_layers=num_encoder_layers, 
    num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, 
    max_len=max_len_tgt, dropout=dropout
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
