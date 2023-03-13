"""
Adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/unsupervised_learning/SimCSE/train_simcse_from_file.py

This file loads sentences from a provided text file. It is expected, that the there is one sentence per line in that text file.

SimCSE will be training using these sentences. Checkpoints are stored every 500 steps to the output folder.

"""
import os
import gzip
import argparse
import tqdm
import logging
import math
from datetime import datetime

from datasets import load_dataset
from data import AUTH_KEY, DATA_PATH

from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample


def load_model(args):
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
 
    if args.max_seq_length <= 512:
        
        word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length,
                model_args={'use_auth_token': args.use_auth_token},
                tokenizer_args={'use_auth_token': args.use_auth_token})
    else:
        word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length, 
                model_args={'attention_window': [128] * 12, 'use_auth_token': args.use_auth_token})

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    return model

def load_data(args):
    train_samples = []
    for line in load_dataset(DATA_PATH, args.dataset_name, split="train", use_auth_token=AUTH_KEY):
            train_samples.append(InputExample(texts=[line['text'], line['text']]))
    logging.info("Train sentences: {}".format(len(train_samples)))
    
    return train_samples

def train_model(args, model, train_samples):
     # We train our model using the MultipleNegativesRankingLoss
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size, drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))
    
    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.num_epochs,
              warmup_steps=warmup_steps,
              optimizer_params={'lr': args.learning_rate},
              checkpoint_path=os.path.join(args.experiments_dir, args.experiment_name),
              show_progress_bar=True,
              use_amp=False  # Set to True, if your GPU supports FP16 cores
              )

def main(args):

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    train_samples = load_data(args)
    model = load_model(args)
    train_model(args, model, train_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--dataset_name", type=str, help="Name of dataset as stored on HF")
    parser.add_argument("--experiments_dir", type=str, default="retriever/models/", help="Directory where trained models will be saved")
    parser.add_argument("--experiment_name", type=str, default=None, help="Sub directory where trained models will be saved")

    parser.add_argument("--model_name", type=str, help="Model name")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
 
    parser.add_argument("--use_auth_token", type=str, default=None, help="HF auth token")
    
    args = parser.parse_args()

    main(args)
