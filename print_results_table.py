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
    parser.add_argument('--subset', default='eval')
    config = parser.parse_args()
    bracket = '\\small{'
    closing_bracket = '}'

    for model in [('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', 'PubMedBERT (base)'),
                  ('microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract', 'PubMedBERT (large)'),
                  ('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-ra', 'PubMedBERT (base) + RA'),
                  ('microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract-ra', 'PubMedBERT (large) + RA')
                  ]:
        dataset_line = f'{model[1]:>15}'
        for dataset in ['bioasq', 'mimic', 'ecthr']:
            BASE_DIR = f'{DATA_DIR}/{dataset}-{config.level}/{model[0]}/'
            scores = {'eval_micro-f1': [], 'eval_macro-f1': [], 'predict_micro-f1': [], 'predict_macro-f1': []}
            dataset_line += ' & '
            try:
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
            except:
                pass
            dataset_line += f'{np.mean(scores[f"{config.subset}_micro-f1"]) * 100 if len(scores[f"{config.subset}_micro-f1"]) else 0:.1f} ' \
                            f'$\pm$ {bracket}{np.std(scores[f"{config.subset}_micro-f1"]) * 100 if len(scores[f"{config.subset}_micro-f1"]) else 0:.1f}{closing_bracket} & '
            dataset_line += f'{np.mean(scores[f"{config.subset}_macro-f1"])  * 100if len(scores[f"{config.subset}_macro-f1"]) else 0:.1f} ' \
                            f'$\pm$ {bracket}{np.std(scores[f"{config.subset}_macro-f1"]) * 100 if len(scores[f"{config.subset}_macro-f1"]) else 0:.1f}{closing_bracket}'
        dataset_line += f' \\\\'
        print(dataset_line)


if __name__ == '__main__':
    main()
