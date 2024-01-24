import torch.nn as nn
import math
import torch
import itertools
import pandas as pd
import numpy as np


class ContentEmbedding(nn.Module):
    def __init__(self, d_model, tokenizer, embeddings_path, knowledge_path):
        super().__init__()
        self.d_model = d_model

        embeddings = pd.read_csv(embeddings_path, sep="\t", names=[str(_) for _ in range(d_model)])
        mapping = pd.read_csv(knowledge_path, sep="\t", names=["ID", "line"])
        known_ids = mapping['ID'].values
        self.tokenizer = tokenizer

        items_idxs = mapping[mapping['ID'].str.startswith('I')]['line']
        mean_item_embedding = embeddings.loc[items_idxs, :].to_numpy().sum(axis=0)
        temp_arrays = np.random.random((tokenizer.vocab_size, d_model)) * 2 -1
        for (k, v) in self.tokenizer.vocab.items():
            if k[1:-1] not in known_ids:
                continue
            temp_idx = mapping[mapping['ID'] == k[1:-1]]['line'].values[0]
            temp_arrays[v] = embeddings.loc[temp_idx, :].to_numpy()
        temp_arrays[self.tokenizer.pad_token_id] = np.zeros(d_model)
        temp_arrays[self.tokenizer.mask_token_id] = np.random.random(d_model) * 2 -1
        self.embeddings = torch.from_numpy(temp_arrays)
        if torch.cuda.is_available():
            self.embeddings = self.embeddings.to('cuda')

    def forward(self, x):
        return self.embeddings[x]