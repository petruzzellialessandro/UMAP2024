import numpy as np
import pandas as pd
import os
import torch
import pickle as pkl
from ast import literal_eval
from torch.utils.data import Dataset
from .train import SCRSDataset
from pathlib import Path
import random
from transformers import set_seed

def set_random():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    set_seed(42)

class SCRSTestDataset(Dataset):
    def __init__(
        self,
        path_dir: str,
        tokenizer,
        max_len_seq=60,
        sliding=False,
        mlm_probability=0.2,
        evaluation_mode=True,
        *args,**kwargs
    ):
        self.parent_folder = Path(__file__).parent.parent
        self.max_len_seq = max_len_seq
        self.tokenizer = tokenizer
        self.items_ = [v for (k, v) in self.tokenizer.vocab.items() if "I" in k]
        database_config = {key: value for key, value in locals().items() if key not in ["self", "kwargs", "args"]}
        movie_tokens = pkl.load(open(os.path.join(self.parent_folder, "data", kwargs['dataset_name'], "movie_ids.pkl"), "rb"))
        movie_tokens = ["[I"+str(x)+"]" for x in movie_tokens]
        self.candidates = [tokenizer.encode(tk)[0] for tk in movie_tokens if tokenizer.encode(tk)[0]!=0]
        self.candidates_num = len(self.candidates)
        database_config.update(kwargs)
        test_set = SCRSDataset(
            **database_config,
        )
        self.batch = {}
        # if evaluation_mode is false we use the bidirectional information during
        # the evaluation phase. This should not be done but let's try
        if evaluation_mode == False:
            self.__bidirect_sequence_(test_set.dataset)
            return

        input_ids = []
        attention_mask_list = []
        labels = []
        # The right way to test is use the information collected up to the MASK
        for _, row in test_set.dataset.iterrows():
            sequence = row["truncated"]  # Trucated sequence (last token is [MASK])
            target = row["target"]  # The element to predict
            sequence_enc = self.tokenizer(
                sequence,
                truncation=True,
                padding="max_length",
                max_length=self.max_len_seq,
            )
            target_enc = [-100] * self.max_len_seq
            # Set the last element of target sequence equal to the target element
            target_enc[-1] = self.tokenizer(target)["input_ids"][0]
            input_ids.append(torch.tensor(sequence_enc["input_ids"]))
            attention_mask_list.append(torch.tensor(sequence_enc["attention_mask"]))
            labels.append(torch.tensor(target_enc))
        self.batch["input_ids"] = torch.stack(input_ids)
        self.batch["attention_mask"] = torch.stack(attention_mask_list)
        self.batch["labels"] = torch.stack(labels)
        #set_random()
        #self.candidates = random.sample(self.items_, self.candidates_num)
        #self.candidates_num = len(self.candidates)

    def __bidirect_sequence_(self, test_df):
        input_ids = []
        attention_mask_list = []
        labels = []
        for _, row in test_df.iterrows():
            sequence = row["train_sequence"]  # Complete sequence with [MASK]
            target = row["target"]  # The element to predict
            sequence_enc = self.tokenizer(
                sequence,
                truncation=True,
                padding="max_length",
                max_length=self.max_len_seq,
            )
            # let's take the index of the MASK in the endoded sequence
            # we will use this index to put the target element in the right pos
            # in the target sequence
            index_mask = sequence_enc["input_ids"].index(self.tokenizer.mask_token_id)
            target_enc = [-100] * self.max_len_seq
            # Set the target element in the target sequence
            target_enc[index_mask] = self.tokenizer(target)["input_ids"][0]
            input_ids.append(torch.tensor(sequence_enc["input_ids"]))
            attention_mask_list.append(torch.tensor(sequence_enc["attention_mask"]))
            labels.append(torch.tensor(target_enc))
        self.batch["input_ids"] = torch.stack(input_ids)
        self.batch["attention_mask"] = torch.stack(attention_mask_list)
        self.batch["labels"] = torch.stack(labels)

    def __len__(self) -> int:
        return len(self.batch["labels"])

    def __getitem__(self, index: int):
        candidates = torch.LongTensor(self.candidates)
        sequence_labels = self.batch["labels"][index]
        if sequence_labels[sequence_labels!=-100] not in candidates:
                remove_idx = random.randint(0, self.candidates_num-1)
                candidates[torch.tensor(remove_idx)] = sequence_labels[sequence_labels!=-100].item()
        if len(candidates)!=self.candidates_num:
                print(candidates)
                print(sequence_labels[sequence_labels!=-100])
                print(index)
                exit(0)

        lables_ = np.zeros(candidates.shape)
        lables_[candidates == sequence_labels[sequence_labels!=-100]] = 1.0
        return self.batch["input_ids"][index], torch.LongTensor(candidates), torch.LongTensor(lables_)
