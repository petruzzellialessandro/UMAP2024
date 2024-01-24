import torch.nn as nn
import torch
from .positional import PositionalEmbedding
from .random import RandomEmbedding

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        super().__init__()
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.content = RandomEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x =  self.position(sequence) +  self.content(sequence) 
        return self.dropout(x) + x