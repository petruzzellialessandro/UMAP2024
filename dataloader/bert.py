from .base import AbstractDataloader
from dataset.train import SCRSDataset
from dataset.test import SCRSTestDataset

import torch.utils.data as data_utils
from transformers import PreTrainedTokenizerFast
from abc import *
import os 
from pathlib import Path


class BertDataloader(AbstractDataloader):
    def __init__(self, dataset, MAX_SEQ_LENGTH, batch_size, remove_prop=[], mask_prob=0.6):
        self.dataset = dataset
        self.parent_folder = Path(__file__).parent.parent
        self.train_batch_size = batch_size
        tokenizer_config = {
            "tokenizer_file": os.path.join(self.parent_folder, "tokenizer", self.dataset,"scrs_tokenizer.json"),
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
            "model_max_length": MAX_SEQ_LENGTH,
            "padding_side": "left",
            "truncation_side": "left",
        }
        tokenizer = PreTrainedTokenizerFast(**tokenizer_config)
        self.train_conf = {
            "tokenizer": tokenizer,
            "max_len_seq": MAX_SEQ_LENGTH,
            "split_sentiment": True,
            "ignore_sentiment": True,
            "sliding": False,
            "test_truncate": False,
            "pos_only": False,
            "mlm_probability": mask_prob,
            "use_standard_collator": False,
            "train_truncate": True,
            "ignore_role": True,
            "user_only": False,
            "dataset_name": self.dataset,
            "remove_prop": remove_prop
        }
        self.test_conf = {
            "tokenizer": tokenizer,
            "max_len_seq": MAX_SEQ_LENGTH,
            "split_sentiment": True,
            "ignore_sentiment": True,
            "sliding": False,
            "test_truncate": True,
            "pos_only": False,
            "mlm_probability": 0.8,
            "use_standard_collator": False,
            "train_truncate": False,
            "ignore_role": True,
            "user_only": False,
            "evaluation_mode": True,
            "dataset_name": self.dataset
        }
        self.eval_conf = {
            "tokenizer": tokenizer,
            "max_len_seq": MAX_SEQ_LENGTH,
            "split_sentiment": True,
            "ignore_sentiment": True,
            "sliding": False,
            "test_truncate": True,
            "pos_only": False,
            "mlm_probability": 0.8,
            "use_standard_collator": False,
            "train_truncate": False,
            "ignore_role": True,
            "user_only": False,
            "evaluation_mode": True,
            "dataset_name": self.dataset
        }
    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader =  self._get_eval_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = SCRSDataset(os.path.join(self.parent_folder, "data", self.dataset, "train.csv"), **self.train_conf)
        return dataset

    def _get_test_loader(self):
        dataset = self._get_test_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.train_batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_loader(self):
        dataset = self._get_eval_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.train_batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self):
        dataset = SCRSDataset(os.path.join(self.parent_folder, "data", self.dataset, "valid.csv"), **self.eval_conf)
        return dataset

    def _get_test_dataset(self):
        dataset = SCRSTestDataset(os.path.join(self.parent_folder, "data", self.dataset, "test.csv"), **self.test_conf)
        return dataset