import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from data import AUTH_KEY
import numpy as np
import matplotlib.pyplot as plt


dataset_name = 'mimic-l2'
max_length = 2048
model_path = 'microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract'
# model_path = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'

tokenizer = AutoTokenizer.from_pretrained(model_path)
train_dataset = load_dataset('kiddothe2b/multilabel_bench', dataset_name, split="train", use_auth_token=AUTH_KEY)


lengths = []
for sample in tqdm.tqdm(train_dataset):
    lengths.append(len(tokenizer.tokenize(sample['text'])))

print(f'MEAN: {np.mean(lengths):.0f} +/- {np.std(lengths):.0f}')
plt.hist(lengths, range=(0, max_length), bins=50)
plt.title(f'MEAN: {np.mean(lengths):.0f} +/- {np.std(lengths):.0f}')
plt.show()
