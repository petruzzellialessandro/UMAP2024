import torch.nn as nn
import torch
from .positional import PositionalEmbedding
from .content import ContentEmbedding

class BERTEmbedding(nn.Module):

    def __init__(self, embed_size, max_len, tokenizer, embeddings_path, knowledge_path, dropout=0.1):
        super().__init__()
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.content = ContentEmbedding(d_model=embed_size, tokenizer=tokenizer, embeddings_path=embeddings_path, knowledge_path=knowledge_path)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        content_ = self.content(sequence).float()
        if torch.cuda.is_available():
            content_ = content_.to('cuda')
        x =  self.position(sequence) +  content_ 
        return self.dropout(x) + x