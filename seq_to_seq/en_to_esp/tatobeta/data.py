from transformers import MarianTokenizer
import os
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, path_to_src, path_to_tgt, tokenizer, max_length=250):
        self.src_text = []
        with open(path_to_src) as f:
            for line in f.readlines():
                self.src_text.append(line.strip())

        self.tgt_text = []
        with open(path_to_tgt) as f:
            for line in f.readlines():
                self.tgt_text.append(line.strip())

        if not len(self.src_text) == len(self.tgt_text):
            raise Exception("len of src not the same as tgt")

        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.src_text)

    def __getitem__(self, idx):
        src_text, tgt_text = self.src_text[idx], self.tgt_text[idx]

        # Manually add the <s> token at the beginning of the source and target texts
        # Tokenizer already adds the EOS token at the end
        tgt_text = f"<s> {tgt_text}"

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


src = os.path.expanduser("~/datasets/en_to_esp/tatobeta/train_en.txt")
tgt = os.path.expanduser("~/datasets/en_to_esp/tatobeta/train_es.txt")
tokenizer = get_tokenizer()
train_dataset = TranslationDataset(src, tgt, tokenizer)

tokenizer.decode(train_dataset[0]["input_ids"])
tokenizer.decode(train_dataset[0]["labels"])
