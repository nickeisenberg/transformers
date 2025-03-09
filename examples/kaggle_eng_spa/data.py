from typing import Any, cast
import pandas as pd
from torch.utils.data import Dataset 
from transformers import MarianTokenizer, PreTrainedTokenizerBase



def get_tokenizer():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    tokenizer.add_special_tokens({"bos_token": "<s>"})
    return tokenizer


class TranslationDataset(Dataset):
    def __init__(self, path_to_eng_spa_df: str, 
                 tokenizer: MarianTokenizer,
                 src_tokenizer_kwargs: dict[str, Any],
                 tgt_tokenizer_kwargs: dict[str, Any]):
        """ 
        max_length=64, 
        padding='do_not_pad', 
        truncation=True, 
        """
        self.df = pd.read_csv(path_to_eng_spa_df)
        self.tokenizer = tokenizer
        self.src_tokenizer_kwargs = src_tokenizer_kwargs 
        self.tgt_tokenizer_kwargs = tgt_tokenizer_kwargs 
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_text = self.df.iloc[idx, 0]
        tgt_text = self.df.iloc[idx, 1]

        # Manually add the <s> token at the beginning of the source and target texts
        tgt_text = f"<s> {tgt_text}"  # Prepend <s> and append </s> to the target text

        src_encoded = self.tokenizer(
            src_text, 
            return_tensors="pt",
            **self.src_tokenizer_kwargs
        )
        
        tgt_encoded = self.tokenizer(
            tgt_text,
            return_tensors="pt",
            **self.tgt_tokenizer_kwargs
        )
        
        return {
            "input_ids": src_encoded["input_ids"].squeeze(),
            "attention_mask": src_encoded["attention_mask"].squeeze(),
            "decoder_input_ids": tgt_encoded["input_ids"].squeeze()[:-1],
            "labels": tgt_encoded["input_ids"].squeeze()[1:]
        }


if __name__ == "__main__":

    tokenizer = cast(
        PreTrainedTokenizerBase, 
        MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    )
    batch = ["hello there. how are you doing", "I am nick"]
    tokenized = tokenizer(
        text=batch,
        return_tensors="pt",
        max_length=64, 
        padding=True, 
        truncation=True,
    )
    
    tokenized["input_ids"]
    tokenized["attention_mask"]
