from .base import BaseModel
from .bert_modules.bert import BERT
from .bert_modules.utils.gelu import GELU
import torch.nn as nn
import numpy as np

class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        tokenizer = args['tokenizer']
        self.not_items_ids = [v for (k, v) in tokenizer.vocab.items() if "I" not in k]
        self.bert = BERT(args)
        self.gelu = GELU()
        self.out_1 = nn.Linear(self.bert.hidden, args['num_items'])
        self.out_2 = nn.Linear(args['num_items'], args['num_items'])
        self.soft = nn.LogSoftmax(dim=-1)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x):
        x = self.bert(x)
        output = self.out_2(self.gelu(self.out_1(x)))
        output[:, :, self.not_items_ids] = np.NINF
        return self.soft(output)