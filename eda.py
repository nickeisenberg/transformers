import torch
import torch.nn as nn
import torch.nn.functional as F

nn.Softmax()

vocab_size = 100
sentence_length = 32
batch_size = 1
emb_dim = 256

# a tokenized input sentence
input = torch.randint(0, vocab_size, (batch_size, sentence_length,))

# the embedding layer for the tokenized sentences
emb = nn.Embedding(vocab_size, emb_dim)

# embend the sentence
input_emb: torch.Tensor = emb(input)

# calculate the weights
raw_weights = torch.bmm(input_emb, input_emb.transpose(1, 2)) / emb_dim ** .5

weights = F.softmax(raw_weights, dim=2)

class SelfAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, mask: bool = False):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.mask = mask

        if not emb_dim % num_heads == 0:
            raise Exception("ERROR: num_heads must divide emb_dim")

        self.to_queries = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_keys = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_values = nn.Linear(emb_dim, emb_dim, bias=False)

        self.unify_heads = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor):

        b, t, k = x.size()

        queries = self.to_queries(x)
        keys = self.to_queries(x)
        values = self.to_queries(x)

        s = k // self.num_heads

        queries: torch.Tensor = queries.view(b, t, self.num_heads, s)
        keys: torch.Tensor = keys.view(b, t, self.num_heads, s)
        values: torch.Tensor = values.view(b, t, self.num_heads, s)

        queries = queries.transpose(1, 2).contiguous().view(b * self.num_heads, t, s)
        keys = keys.transpose(1, 2).contiguous().view(b * self.num_heads, t, s)
        values = values.transpose(1, 2).contiguous().view(b * self.num_heads, t, s)

        weights = F.softmax(
                torch.bmm(queries, keys.transpose(1, 2)) / s ** .5,
                dim=2
        )

        out = torch.bmm(weights,values).view(b, self.num_heads, t, s)

        return self.unify_heads(out)
