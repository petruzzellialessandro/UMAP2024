from dataloader.bert import BertDataloader
from transformers import PreTrainedTokenizerFast
from model.bert import BERTModel
from trainers.bert import BERTTrainer
import torch
from pathlib import Path 

from transformers import set_seed
import random
import numpy as np
import sys
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
set_seed(42)

import argparse
import os 
import re

parser = argparse.ArgumentParser()
parser.add_argument('--embedding-conf', type=str, required=True)
parser.add_argument('--dim', type=int, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--mask_prob', type=str, required=False, default='0.6')
parser.add_argument('--remove_prop', type=str, required=False, default='')
args = parser.parse_args()
dim = args.dim
embedding_conf = args.embedding_conf.strip()
dataset = args.dataset.strip()
remove_prop = args.remove_prop.strip()
mask_prob = float(args.mask_prob.strip())

assert dataset in ['redial', 'inspired']
confs = list(map(lambda x: re.sub('_k=[0-9]*', "", x), os.listdir(f"./data/{dataset}/Embeddings/")))
confs.append("random")
assert embedding_conf in confs
sub_folder = None
sub_folder = embedding_conf
assert dim in [16, 32, 64, 128, 256]
if len(remove_prop)==0:
    remove_prop = []
else:
    remove_prop = remove_prop.split("_")
batch_size = 256
MAX_SEQ_LENGTH = 60 if dataset=="redial" else 50
data_loader = BertDataloader(dataset, MAX_SEQ_LENGTH, batch_size, remove_prop=remove_prop, mask_prob=mask_prob)
train_loader, val_loader, test_loader = data_loader.get_pytorch_dataloaders()
trainer_args = {
    'device': 'cuda' if torch.cuda.is_available() else "cpu",
    'num_gpu': 1,
    'device_idx': '0',
    'optimizer': 'sgd',
    'lr': 0.005 if dataset=="redial" else 0.0005,
    'enable_lr_schedule': True,
    'gamma': 0.1,
    'num_epochs': 10 if dataset=="redial" else 100,
    'metric_ks': [1, 5, 10, 20, 50, 100],
    'best_metric': 'Recall@1',
    'weight_decay': 5 if dataset=="redial" else 2,
    'decay_step': 128,
    'log_period_as_iter': batch_size,
    'train_batch_size': batch_size,
    'momentum': 0.6 if dataset=="redial" else 0.9,
    'embedding_conf': embedding_conf,
    'dim': dim,
    'dataset': dataset,
}
parent_folder = Path(__file__).parent
tokenizer_config = {
    "tokenizer_file": os.path.join(parent_folder, "tokenizer", dataset, "scrs_tokenizer.json"),
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "mask_token": "[MASK]",
    "model_max_length": MAX_SEQ_LENGTH,
    "padding_side": "left",
    "truncation_side": "left",
}
tokenizer = PreTrainedTokenizerFast(**tokenizer_config)
args_dict = {
    'bert_max_len': MAX_SEQ_LENGTH,
    'tokenizer': tokenizer,
    'num_items': tokenizer.vocab_size,
    'bert_num_blocks': 2,
    'bert_num_heads': 2 if dataset=="redial" else 2,
    'bert_hidden_units': dim,
    'bert_dropout': 0.5 if dataset=="redial" else 0.3,
    'knowledge_path': os.path.join(parent_folder, "data", dataset, "Embeddings", f"{sub_folder}_k={dim}", "entities_to_id.tsv") if embedding_conf!="random" else None, 
    'embeddings_path': os.path.join(parent_folder, "data", dataset, "Embeddings", f"{sub_folder}_k={dim}", "embeddings.tsv") if embedding_conf!="random" else None, 
}
export_root = os.path.join(parent_folder, dataset, f'mask={mask_prob}', f"{embedding_conf}")
print(export_root)
os.makedirs(export_root, exist_ok=True)
model = BERTModel(args_dict)
trainer = BERTTrainer(trainer_args, model, train_loader, val_loader, test_loader, tokenizer, export_root)
trainer.train()
trainer.test()