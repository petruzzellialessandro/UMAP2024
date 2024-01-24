from torch import nn as nn

from .embeddings.bert import BERTEmbedding
from .embeddings.token import TokenEmbedding
from .transformer import TransformerBlock


class BERT(nn.Module):
    def __init__(self, args_dict):
        super().__init__()
        
        max_len = args_dict['bert_max_len']
        tokenizer = args_dict['tokenizer']
        num_items = args_dict['num_items']
        n_layers = args_dict['bert_num_blocks']
        heads = args_dict['bert_num_heads']
        vocab_size = num_items + 2
        hidden = args_dict['bert_hidden_units']
        self.hidden = hidden
        dropout = args_dict['bert_dropout']
        embeddings_path = args_dict['embeddings_path']
        knowledge_path = args_dict['knowledge_path']

        if embeddings_path != None:
        # embedding for BERT, sum of positional, content
            self.embedding = BERTEmbedding(embed_size=self.hidden, max_len=max_len, tokenizer=tokenizer, dropout=dropout,
                                            embeddings_path = embeddings_path, knowledge_path = knowledge_path).requires_grad_(False)
        else: 
        # embedding for BERT, sum of positional, token
            self.embedding = TokenEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])
    
    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass