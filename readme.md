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

To run the KASCRS, you first need to set up the Python environment and
then execute the script. This will learn the knowledge graph embedding.

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
pip install -r /KgEmbedding/requirements.txt
```

## Run the Knowledge Graph Embedding Learner

1.  Activate the virtual environment:

``` {bash}
source /path/to/env/kg_embedding/bin/activate
```

2.  Run the script

``` {bash}
python /KgEmbedding/KgEmbedding.py --dataset={dataset} --emb_model={emb_model} --emb_dim={emb_dim} --gcn_layer={gcn_layer}
```

The script accepts the following arguments:

- `dataset`: Specify the name of the dataset to be used for training. It
  can be `redial` or `inspired`
- `emb_model`: Choose the embedding model to pre-train the graph
  representations It can be `TransE`, `TransH`, `RGCN`, `CompGCN`
- `emb_dim`: Choose the embedding dimension. It can be `16`, `32`, `64`,
  `128`
- `gcn_layer`: Choose the embedding layer of GCN models (RGCN, CompGCN).
  It can be `1` or `2`. Not mandatory. \## TO DO Write where the the
  embedding are saved and where they have to be moved.

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
pip main.py --embedding-conf={embedding-conf} --dim={dim} --dataset={dataset} --mask_prob={mask_prob}
```

The script accepts the following arguments:

- `embedding-conf`: This parameter specifies the configuration file for
  the graph embedding layer. The configuration name should be the same
  name of the folder where the embedding are stored excluding
  `_k={dim}`.
- `dim`: This parameter sets the dimension of the pretrained graph
  embeddings.
- `--dataset`: This parameter specifies the dataset that the model will
  be trained and tested on. It can be `redial` or `inspired`
- `mask-prob`: This parameter sets the probability of masking items in
  the sequences during training. Masking involves randomly replacing
  words in the input sequence with special tokens called masks.

# Redial Results

| model_name       | embedding_size | mask_prob | Recall@1       | Recall@10      | Recall@50      | weight_decay | dropout |
|------------------|----------------|-----------|----------------|----------------|----------------|--------------|---------|
| CompGCN_layers=2 | 32             | 0.6       | 0.120925377123 | 0.409154431894 | 0.759361531585 | 5.00         | 0.5     |
| CompGCN_layers=1 | 32             | 0.6       | 0.120589978304 | 0.400284415413 | 0.745676117979 | 5.00         | 0.5     |
| RGCN_layers=1    | 32             | 0.6       | 0.115851154178 | 0.407927275000 | 0.726942160000 | 5.00         | 0.5     |
| RGCN_layers=2    | 32             | 0.6       | 0.103840450000 | 0.402694216000 | 0.713700000000 | 5.00         | 0.5     |
| TransE           | 32             | 0.6       | 0.105899783038 | 0.402844154127 | 0.716761179789 | 5.00         | 0.5     |
| TransH           | 32             | 0.6       | 0.113058997830 | 0.405284415413 | 0.746761179789 | 5.00         | 0.5     |
| CompGCN_layers=2 | 16             | 0.6       | 0.118138806429 | 0.405441809446 | 0.726580179110 | 5.00         | 0.5     |
| CompGCN_layers=2 | 64             | 0.6       | 0.120629647087 | 0.403732825071 | 0.743528083563 | 5.00         | 0.5     |
| CompGCN_layers=2 | 128            | 0.6       | 0.113449622877 | 0.395726697519 | 0.768100079149 | 5.00         | 0.5     |

# Inspired Results

| model_name       | embedding_size | mask_prob | Recall@1 | Recall@10 | Recall@50 | weight_decay | dropout | lr     |
|------------------|----------------|-----------|----------|-----------|-----------|--------------|---------|--------|
| CompGCN_layers=2 | 64             | 0.2       | 0.16094  | 0.44805   | 0.64766   | 2            | 0.3     | 0.0005 |
| CompGCN_layers=1 | 64             | 0.2       | 0.15859  | 0.42031   | 0.64805   | 2            | 0.3     | 0.0005 |
| RGCN_layers=1    | 64             | 0.2       | 0.15859  | 0.38633   | 0.64609   | 2            | 0.3     | 0.0005 |
| RGCN_layers=2    | 64             | 0.2       | 0.15762  | 0.42031   | 0.65195   | 2            | 0.3     | 0.0005 |
| TransE           | 64             | 0.2       | 0.15859  | 0.42031   | 0.64414   | 2            | 0.3     | 0.0005 |
| TransH           | 64             | 0.2       | 0.15859  | 0.42031   | 0.64414   | 2            | 0.3     | 0.0005 |
| CompGCN_layers=2 | 16             | 0.2       | 0.15816  | 0.39179   | 0.61787   | 2            | 0.3     | 0.0005 |
| CompGCN_layers=2 | 32             | 0.2       | 0.15931  | 0.39375   | 0.62611   | 2            | 0.3     | 0.0005 |
| CompGCN_layers=2 | 128            | 0.2       | 0.15469  | 0.40586   | 0.65781   | 2            | 0.3     | 0.0005 |
