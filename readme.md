# \[UMAP 2024\]KASCRS

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
