# \[UMAP 2024\] KASCRS

This repository provides the code used to reproduce the experiments
presented in the paper
`Improving Transformer-based Sequential Conversational Recommendations through Knowledge Graph Embeddings`,
which was submitted to `UMAP 2024`.

# Project Structure

``` {bash}
.
│   main.py
│   requirements.txt
│
├───data
│   ├───inspired
│   │   │   movie_ids.pkl
│   │   │   test.csv
│   │   │   train.csv
│   │   │   valid.csv
│   │   │
│   │   └───Embeddings
│   │
│   └───redial
│       │   movie_ids.pkl
│       │   test.csv
│       │   train.csv
│       │   valid.csv
│       │
│       └───Embeddings
│
├───dataloader
│       base.py
│       bert.py
│       init.py
│
├───dataset
│       test.py
│       train.py
│
├───KgEmbedding
│   │   KgEmbedding.py
│   │   requirements.txt
│   │
│   ├───inspired
│   │       n_inspired_test.tsv
│   │       n_inspired_train.tsv
│   │
│   └───redial
│           n_redial_test.tsv
│           n_redial_train.tsv
│
├───model
│   │   base.py
│   │   bert.py
│   │
│   └───bert_modules
│       │   bert.py
│       │   init.py
│       │   transformer.py
│       │
│       ├───attention
│       │       init.py
│       │       multi_head.py
│       │       single.py
│       │
│       ├───embeddings
│       │       bert.py
│       │       content.py
│       │       init.py
│       │       positional.py
│       │       random.py
│       │       token.py
│       │
│       └───utils
│               feed_forward.py
│               gelu.py
│               init.py
│               layer_norm.py
│               sublayer.py
│
├───tokenizer
│   ├───inspired
│   │       scrs_tokenizer.json
│   │
│   └───redial
│           scrs_tokenizer.json
│
└───trainers
        base.py
        bert.py
        loggers.py
        utils.py
```

# Preliminaries

To run the KASCRS, you first need to set up the Python environment and then execute the script. This will learn the knowledge graph embedding.

To execute the scripts you need an Ubuntu machine with CUDA 12.4 and Python 3.9.13.

## Environment for Knowledge Graph Embedding

1.  Create a new Python virtual environment for Knowledge Graph
    Embedding using the command:

``` {bash}
python -m venv /path/to/env/kg_embedding
```

2.  Activate the virtual environment:

``` {bash}
source /path/to/env/kg_embedding/bin/activate
```

3.  Install the required libraries using the command:

``` {bash}
pip install -r ./KgEmbedding/requirements.txt
```

## Run the Knowledge Graph Embedding Learner

1.  Activate the virtual environment:

``` {bash}
source /path/to/env/kg_embedding/bin/activate
```

2.  Run the script

``` {bash}
cd ./KgEmbedding
python KgEmbedding.py --dataset={dataset} --emb_model={emb_model} --emb_dim={emb_dim} --gcn_layer={gcn_layer}
```

The script accepts the following arguments:

- `dataset`: Specify the name of the dataset to be used for training. It
  can be `redial` or `inspired`
- `emb_model`: Choose the embedding model to pre-train the graph
  representations It can be `TransE`, `TransH`, `RGCN`, `CompGCN`
- `emb_dim`: Choose the embedding dimension. It can be `16`, `32`, `64`,
  `128`
- `gcn_layer`: Choose the embedding layer of GCN models (RGCN, CompGCN).
  It can be `1` or `2`. Not mandatory.

The embedding results can be found in the folder
`results/{dataset}/{emb_model}_k={emb_dim}}` or
`results/{dataset}/{emb_model}_layers={gcn_layer}_k={emb_dim}}` for GCN
models. To incorporate the embedding into KASCRS training, copy the
folder named after the embedding model into the
`data/{dataset}/Embedding` directory

# Model Training

To train and test the model, create a second virtual environment using
the following commands:

1.  Create the virtual environment:

``` {bash}
python -m venv /path/to/env/KASCRS_env
```

2.  Activate the virtual environment:

``` {bash}
source /path/to/env/KASCRS_env/bin/activate
```

3.  Install the required libraries using:

``` {bash}
pip install -r requirements.txt
```

To train and test the model:

``` {bash}
python main.py --embedding-conf={embedding-conf} --dim={dim} --dataset={dataset} --mask_prob={mask_prob}
```

The script accepts the following arguments:

- `embedding-conf`: This parameter specifies the configuration file for
  the graph embedding layer. The configuration name should be the same
  name of the folder where the embedding are stored excluding
  `_k={dim}`.
- `dim`: This parameter sets the dimension of the pretrained graph
  embeddings.
- `dataset`: This parameter specifies the dataset that the model will be
  trained and tested on. It can be `redial` or `inspired`
- `mask-prob`: This parameter sets the probability of masking items in
  the sequences during training. Masking involves randomly replacing
  words in the input sequence with special tokens called masks.

# ReDial Results

| Model Name       | Embedding Size | Mask Prob | Recall@1 | Recall@10 | Recall@50 | Weight Decay | Dropout | LR    |
|------------------|----------------|-----------|----------|-----------|-----------|--------------|---------|-------|
| CompGCN_layers=2 | 32             | 0.6       | 0.1209   | 0.4092    | 0.7594    | 5            | 0.5     | 0.005 |
| CompGCN_layers=1 | 32             | 0.6       | 0.1206   | 0.4003    | 0.7457    | 5            | 0.5     | 0.005 |
| RGCN_layers=1    | 32             | 0.6       | 0.1159   | 0.4079    | 0.7269    | 5            | 0.5     | 0.005 |
| RGCN_layers=2    | 32             | 0.6       | 0.1038   | 0.4027    | 0.7137    | 5            | 0.5     | 0.005 |
| TransE           | 32             | 0.6       | 0.1059   | 0.4028    | 0.7168    | 5            | 0.5     | 0.005 |
| TransH           | 32             | 0.6       | 0.1131   | 0.4053    | 0.7468    | 5            | 0.5     | 0.005 |
| CompGCN_layers=2 | 32             | 0.2       | 0.1057   | 0.4018    | 0.7451    | 5            | 0.5     | 0.005 |
| CompGCN_layers=2 | 32             | 0.4       | 0.1145   | 0.4102    | 0.7519    | 5            | 0.5     | 0.005 |
| CompGCN_layers=2 | 32             | 0.8       | 0.1129   | 0.3960    | 0.7479    | 5            | 0.5     | 0.005 |
| CompGCN_layers=2 | 16             | 0.6       | 0.1181   | 0.4054    | 0.7266    | 5            | 0.5     | 0.005 |
| CompGCN_layers=2 | 64             | 0.6       | 0.1206   | 0.4037    | 0.7435    | 5            | 0.5     | 0.005 |
| CompGCN_layers=2 | 128            | 0.6       | 0.1134   | 0.3957    | 0.7681    | 5            | 0.5     | 0.005 |

# INSPIRED Results

| Model Name       | Embedding Size | Mask Prob | Recall@1 | Recall@10 | Recall@50 | Weight Decay | Dropout | LR     |
|------------------|----------------|-----------|----------|-----------|-----------|--------------|---------|--------|
| CompGCN_layers=2 | 64             | 0.2       | 0.1609   | 0.4480    | 0.6477    | 2            | 0.3     | 0.0005 |
| CompGCN_layers=1 | 64             | 0.2       | 0.1585   | 0.4203    | 0.6480    | 2            | 0.3     | 0.0005 |
| RGCN_layers=1    | 64             | 0.2       | 0.1586   | 0.3863    | 0.6461    | 2            | 0.3     | 0.0005 |
| RGCN_layers=2    | 64             | 0.2       | 0.1576   | 0.4203    | 0.6519    | 2            | 0.3     | 0.0005 |
| TransE           | 64             | 0.2       | 0.1586   | 0.4203    | 0.6441    | 2            | 0.3     | 0.0005 |
| TransH           | 64             | 0.2       | 0.1586   | 0.4203    | 0.6441    | 2            | 0.3     | 0.0005 |
| CompGCN_layers=2 | 64             | 0.4       | 0.1238   | 0.4363    | 0.6267    | 2            | 0.3     | 0.0005 |
| CompGCN_layers=2 | 64             | 0.6       | 0.0477   | 0.4057    | 0.6320    | 2            | 0.3     | 0.0005 |
| CompGCN_layers=2 | 64             | 0.8       | 0.0477   | 0.4269    | 0.6360    | 2            | 0.3     | 0.0005 |
| CompGCN_layers=2 | 16             | 0.2       | 0.1581   | 0.3918    | 0.6179    | 2            | 0.3     | 0.0005 |
| CompGCN_layers=2 | 32             | 0.2       | 0.1593   | 0.3937    | 0.6261    | 2            | 0.3     | 0.0005 |
| CompGCN_layers=2 | 128            | 0.2       | 0.1547   | 0.4059    | 0.6578    | 2            | 0.3     | 0.0005 |
