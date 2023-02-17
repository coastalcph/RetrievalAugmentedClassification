import os
import json
from tqdm import tqdm
import argparse 
import h5py
import random
random.seed(42)

from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import torch

from data import AUTH_KEY, DATA_PATH


def load_data(args):
    
    corpus = load_dataset(DATA_PATH, args.dataset_name, split="train", use_auth_token=AUTH_KEY)
    
    sample_ids = random.sample(range(len(corpus)), k=min(10000, len(corpus)))
    queries = {}
    queries['train'] = corpus.select(sample_ids)
    queries['val'] = load_dataset(DATA_PATH, args.dataset_name, split="validation", use_auth_token=AUTH_KEY)
    queries['test'] = load_dataset(DATA_PATH, args.dataset_name, split="test", use_auth_token=AUTH_KEY)

    return corpus, queries

def find_neighbors(embedder, corpus, corpus_embeddings, queries):

    query2neighbors = {}

    top_k = min(args.k + 1, len(corpus))
    for query in queries:
        query_embedding = embedder.encode(query['text'], convert_to_tensor=True)
 
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        
        query2neighbors[query['doc_id']] = []
        for score, idx in zip(top_results[0].tolist(), top_results[1].tolist()):
            # discard retrieved doc if it's identical to the query
            if query['doc_id'] == corpus[idx]['doc_id']:
                continue
            query2neighbors[query['doc_id']].append({'sim_score': score, 'doc_id': corpus[idx]['doc_id']})
        query2neighbors[query['doc_id']] = query2neighbors[query['doc_id']][:args.k]

    return query2neighbors

def write_embeddings(args, corpus, corpus_embeddings):
    h5py_file = h5py.File(os.path.join(args.output_dir, 'corpus.hdf5'), 'w')
    for doc_id, embedding in zip(corpus['doc_id'], corpus_embeddings):
        h5py_file.create_dataset(doc_id, embedding.shape, data=embedding.cpu())

def write_neighbors(args, split, query2neighbors):
    output_path = os.path.join(args.output_dir, '{}.json'.format(split))
    json.dump(query2neighbors, open(output_path, 'w'))

def main(args):
    embedder = SentenceTransformer(args.model_name)

    corpus, queries = load_data(args)
    print('Embedding corpus...')
    corpus_embeddings = embedder.encode(corpus['text'][:100], convert_to_tensor=True)
    write_embeddings(args, corpus, corpus_embeddings)

    for split in ['train', 'val', 'test']:
        print('Processing {} split'.format(split))
        query2neighbors = find_neighbors(embedder, corpus, corpus_embeddings, queries[split])
        write_neighbors(args, split, query2neighbors)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--dataset_name", type=str, help="Name of dataset as stored on HF")
    parser.add_argument("--output_dir", type=str, help="Where to store the cached embedding vectors and NN ids")

    parser.add_argument("--model_name", type=str, help="Directory where trained model is saved")

    parser.add_argument("--k", type=int, default=10, help="Number of NNs to save")

    args = parser.parse_args()

    main(args)
