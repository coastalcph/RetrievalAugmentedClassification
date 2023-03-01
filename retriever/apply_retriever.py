import os
import json
import argparse
import h5py
import random
random.seed(42)

from sentence_transformers import SentenceTransformer, util, models
from datasets import load_dataset
import torch

from data import AUTH_KEY, DATA_DIR, DATA_PATH


def load_data(args):
    
    corpus = load_dataset(DATA_PATH, args.dataset_name, split="train", use_auth_token=AUTH_KEY)

    sample_ids = random.sample(range(len(corpus)), k=min(10000, len(corpus)))
    queries = {}
    queries['train'] = corpus.select(sample_ids)
    queries['validation'] = load_dataset(DATA_PATH, args.dataset_name, split="validation", use_auth_token=AUTH_KEY)
    queries['test'] = load_dataset(DATA_PATH, args.dataset_name, split="test", use_auth_token=AUTH_KEY)

    return corpus, queries

def find_neighbors(embedder, corpus, corpus_embeddings, queries, query_embeddings, split):

    query2neighbors = {}
    top_k = min(args.k + 1, len(corpus))
    for query, query_embedding in zip(queries, query_embeddings):
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

def write_embeddings(args, corpus, corpus_embeddings, name):
    if not os.path.exists(os.path.join(DATA_DIR, args.output_dir)):
        os.mkdir(os.path.join(DATA_DIR, args.output_dir))
    h5py_file = h5py.File(os.path.join(DATA_DIR, args.output_dir, '{}.hdf5'.format(name)), 'w')
    for doc_id, embedding in zip(corpus['doc_id'], corpus_embeddings):
        h5py_file.create_dataset(doc_id, embedding.shape, data=embedding.cpu())

def write_neighbors(args, split, datastore, query2neighbors):
    if not os.path.exists(os.path.join(DATA_DIR, args.output_dir)):
        os.mkdir(os.path.join(DATA_DIR, args.output_dir))
    if datastore == 'corpus':
        output_path = os.path.join(DATA_DIR, args.output_dir, '{}.json'.format(split))
    else:
        output_path = os.path.join(DATA_DIR, args.output_dir, '{}_nns-{}.json'.format(datastore, split))
    json.dump(query2neighbors, open(output_path, 'w'))

def main(args):
    if 'sentence' in args.model_name or 'setfit' in args.model_name:
        embedder = SentenceTransformer(args.model_name)
    else:
        from torch import nn
        class Pooler(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x

        word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length)
        if word_embedding_model.auto_model.config.model_type == 'bert':
            word_embedding_model.auto_model.pooler = Pooler()
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
        embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    corpus, queries = load_data(args)
    print('Embedding corpus...')
    corpus_embeddings = embedder.encode(corpus['text'], convert_to_tensor=True)
    write_embeddings(args, corpus, corpus_embeddings, 'corpus')
    
    query_embeddings = {}
    for split in ['train', 'validation', 'test']:
        print('Processing {} split'.format(split))
        query_embeddings[split] = embedder.encode(queries[split]['text'], convert_to_tensor=True)
        write_embeddings(args, queries[split], query_embeddings[split], split)
        query2corpus_neighbors = find_neighbors(embedder, corpus, corpus_embeddings, queries[split], query_embeddings[split], split)
        write_neighbors(args, split, 'corpus', query2corpus_neighbors)
        query2train_neighbors = find_neighbors(embedder, queries['train'], query_embeddings['train'], queries[split], query_embeddings[split], split)
        write_neighbors(args, split, 'train', query2train_neighbors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--dataset_name", type=str, default='bioasq-l2', help="Name of dataset as stored on HF")
    parser.add_argument("--output_dir", type=str, help="Where to store the cached embedding vectors and NN ids")

    parser.add_argument("--model_name", type=str, help="Directory where trained model is saved")

    parser.add_argument("--max_seq_length", type=int, default=512, help="Max length")

    parser.add_argument("--k", type=int, default=32, help="Number of NNs to save")

    args = parser.parse_args()

    main(args)
