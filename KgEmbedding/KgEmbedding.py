from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from os import path
import torch
from termcolor import colored

# organized as list so that it is easy to automatically iterate 
# if you want to add other datasets, models, or embedding dimensions
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--emb_model', type=str, required=True)
parser.add_argument('--emb_dim', type=int, required=True)
parser.add_argument('--gcn_layer', type=int, required=False, default=0)
args = parser.parse_args()

dataset = args.dataset
emb_model = args.emb_model
emb_dim = args.emb_dim
gcn_layer = args.gcn_layer
emb_epochs = 50

printline = emb_model+' - k='+str(emb_dim)
print(colored('Starting ' + printline,'blue'))

if emb_model in ['CompGCN', 'RGCN']:
    folder = f'results/{dataset}/'+emb_model+'_layers='+str(gcn_layer)+'_k='+str(emb_dim)
else:
    folder = f'results/{dataset}/'+emb_model+'_k='+str(emb_dim)

train_path = dataset + '/' + f'n_{dataset}_train.tsv'
test_path = dataset + '/' +f'n_{dataset}_test.tsv'

checkpoint_name_file = 'check_checkpoint_'+emb_model+'_k='+str(emb_dim)
print(folder)
try:      

    print(colored('Starting learning:' + folder,'blue'))
    print("Starting learning:", printline)
    

    emb_training = TriplesFactory.from_path(
        train_path,
        create_inverse_triples=False if emb_model == 'RGCN' else True,
    )

    emb_testing = TriplesFactory.from_path(
        test_path,
        entity_to_id=emb_training.entity_to_id,
        relation_to_id=emb_training.relation_to_id,
        create_inverse_triples=False if emb_model == 'RGCN' else True,
    )
    if emb_model in ['CompGCN']:
        model_kwargs=dict(embedding_dim=emb_dim, encoder_kwargs=dict(num_layers=gcn_layer))
    elif emb_model in ['RGCN']:
        model_kwargs=dict(embedding_dim=emb_dim, num_layers=gcn_layer)
    else:
        model_kwargs=dict(embedding_dim=emb_dim)

    result = pipeline(
        training=emb_training,
        testing=emb_testing,
        model=emb_model,
        model_kwargs=model_kwargs,
        evaluation_fallback = True,
        training_kwargs=dict(
            num_epochs=emb_epochs,
            checkpoint_name=checkpoint_name_file,
            checkpoint_directory=folder,
            checkpoint_frequency=1
        ),
         device='cuda'
    )

    if not os.path.exists(folder):
        os.mkdir(folder)

    torch.save(result, folder+'/pipeline_result.dat')

    map_ent = pd.DataFrame(data=list(emb_training.entity_to_id.items()))
    map_ent.to_csv(folder+'/entities_to_id.tsv', sep='\t', header=False, index=False)
    map_ent = pd.DataFrame(data=list(emb_training.relation_to_id.items()))
    map_ent.to_csv(folder+'/relations_to_id.tsv', sep='\t', header=False, index=False)


    # save mappings
    result.save_to_directory(folder, save_training=True, save_metadata=True)

    # extract embeddings with gpu
    #entity_embedding_tensor = result.model.entity_representations[0](indices = None)
    # save entity embeddings to a .tsv file (gpu)
    #df = pd.DataFrame(data=entity_embedding_tensor.data.numpy())

    # extract embeddings with cpu
    entity_embedding_tensor = result.model.entity_representations[0](indices=None).cpu().detach().numpy()
    # save entity embeddings to a .tsv file (cpu)
    df = pd.DataFrame(data=entity_embedding_tensor.astype(float))

    outfile = folder + '/embeddings.tsv'
    df.to_csv(outfile, sep='\t', header=False, index=False)

    print(colored('Completed ' + printline,'green'))

except Exception as e:

    print(colored('An error occoured in ' + printline, 'red'))
    print(colored(e, 'red'))
    exit()