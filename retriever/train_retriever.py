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

from torch.utils.data import DataLoader
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample


def load_model(args):
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    return model

def load_data(args):
    ################# Read the train corpus  #################
    train_samples = []
    with gzip.open(args.data_path, 'rt', encoding='utf8') if args.data_path.endswith('.gz') else open(args.data_path, encoding='utf8') as fIn:
        for line in tqdm.tqdm(fIn, desc='Read file'):
            line = line.strip()
            if '\t' in line:
                line1, line2 = line.split('\t')
            else:
                line1, line2 = line, line
            if len(line1) >= 10 and len(line2)>10:
                train_samples.append(InputExample(texts=[line1, line2]))
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
    parser.add_argument("--data_path", type=str, help="Path to data. File could contain two tab-separated columns (for supervised training) or one column (for usnuspervised training)")
    parser.add_argument("--experiments_dir", type=str, default="experiments/", help="Directory where trained models will be saved")
    parser.add_argument("--experiment_name", type=str, default=None, help="Sub directory where trained models will be saved")

    parser.add_argument("--model_name", type=str, default='allenai/longformer-base-4096', help="Model name")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")
    
    args = parser.parse_args()

    main(args)
