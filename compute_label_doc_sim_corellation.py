import json
import os
import argparse
from datasets import load_dataset
import numpy as np
import random
from data import AUTH_KEY, DATA_DIR
from scipy.stats import kendalltau, spearmanr, pearsonr


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--dataset_name', default='bioasq-l2')
    parser.add_argument('--no_neighbors', default=4)
    parser.add_argument('--embeddings_path', default='bioasq-biomedbert-base-embeddings')
    config = parser.parse_args()

    train_dataset = load_dataset('kiddothe2b/multilabel_bench', config.dataset_name, split="train",
                                 use_auth_token=AUTH_KEY)

    with open(os.path.join(DATA_DIR, config.embeddings_path, 'train.json')) as file:
        neighbors = json.load(file)

    doc_labels = {}
    for document in train_dataset:
        doc_labels[document['doc_id']] = document['labels']

    random.seed(42)
    sample_ids = random.sample(range(len(train_dataset)), k=10000)
    train_dataset = train_dataset.select(sample_ids)

    overlap_ratio = []
    kendalltau_ratio = []
    pearsonr_ratio = []
    spearmanr_ratio = []
    for document in train_dataset:
        doc_sims = []
        label_sims = []
        for neighbor in neighbors[document['doc_id']][:config.no_neighbors]:
            doc_sims.append(neighbor['sim_score'])
            label_sim = len(set(doc_labels[document['doc_id']]).intersection(doc_labels[neighbor['doc_id']])) / \
                        min(len(doc_labels[document['doc_id']]), len(doc_labels[neighbor['doc_id']]))
            label_sims.append(label_sim)
        overlap_ratio.append(np.mean(label_sims))
        if not (np.array(doc_sims == doc_sims[0]).all() or np.array(label_sims == label_sims[0]).all()):
            kendalltau_ratio.append(kendalltau(doc_sims, label_sims)[0])
            pearsonr_ratio.append(pearsonr(doc_sims, label_sims)[0])
            spearmanr_ratio.append(spearmanr(doc_sims, label_sims)[0])

    print(f'{config.embeddings_path}'
          f'Overlap: {np.mean(overlap_ratio):.2f}\tKendall T: {np.mean([r for r in kendalltau_ratio if r is not np.nan]):.2f}\t'
          f'Pearson R: {np.mean([r for r in pearsonr_ratio if r is not np.nan]):.2f}\tSpearman R: {np.mean([r for r in spearmanr_ratio if r is not np.nan]):.2f}\t')


if __name__ == '__main__':
    main()
