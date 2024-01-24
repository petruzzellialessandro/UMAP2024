from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn
import torch
import numpy as np


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, tokenizer, export_root):
        self.tokenizer = tokenizer
        self.not_items_ids = [v for (k, v) in self.tokenizer.vocab.items() if "I" not in k]
        self.items_ids = [v for (k, v) in self.tokenizer.vocab.items() if "I" in k]
        self.loss_fct = nn.NLLLoss(reduction="sum", ignore_index=self.tokenizer.pad_token_id)#
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        seqs, labels = batch
        row_pos, col_pos = torch.where(labels!=0)
        logits = self.model(seqs) 
        nll_loss = self.loss_fct(
            logits[row_pos, col_pos, :], labels[row_pos, col_pos]
        )
        if nll_loss == np.inf:
            print(seqs, labels)
            exit(0)
        return nll_loss/len(row_pos)

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C #candidates
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics

    def get_stast_infos(self, batch):
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C #candidates
        _, indices = torch.sort(-scores)
        _, col_pos = torch.where(labels!=0)
        _, rank_pos = torch.where(indices == col_pos.unsqueeze(1))
        return rank_pos.tolist()