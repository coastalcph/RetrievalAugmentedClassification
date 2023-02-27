import json
import os
import argparse

import numpy as np

from data import DATA_DIR


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--level', default='l2')
    parser.add_argument('--dataset', default='bioasq')
    parser.add_argument('--n_neighbors', default=4)
    parser.add_argument('--subset', default='eval')
    parser.add_argument('--model', default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-ra')
    config = parser.parse_args()

    for n_layers in [1, 2, 4]:
        dataset_line = f'N={n_layers}'
        for aheads in [1, 2, 4]:
            BASE_DIR = f'{DATA_DIR}/{config.dataset}-{config.level}/{config.model}/nn-{config.n_neighbors}-dl-{n_layers}-{aheads}'
            scores = {'eval_micro-f1': [], 'eval_macro-f1': [], 'predict_micro-f1': [], 'predict_macro-f1': []}
            dataset_line += ' & '
            with open(os.path.join(BASE_DIR, 'all_results.json')) as json_file:
                json_data = json.load(json_file)
                dev_micro_f1 = float(json_data['eval_micro-f1'])
                scores['eval_micro-f1'].append(dev_micro_f1)
                dev_macro_f1 = float(json_data['eval_macro-f1'])
                scores['eval_macro-f1'].append(dev_macro_f1)
                test_micro_f1 = float(json_data['predict_micro-f1'])
                scores['predict_micro-f1'].append(test_micro_f1)
                test_macro_f1 = float(json_data['predict_macro-f1'])
                scores['predict_macro-f1'].append(test_macro_f1)

            dataset_line += f" {dev_micro_f1*100:.1f} & {dev_macro_f1*100:.1f}"
        dataset_line += f' \\\\'
        print(dataset_line)


if __name__ == '__main__':
    main()
