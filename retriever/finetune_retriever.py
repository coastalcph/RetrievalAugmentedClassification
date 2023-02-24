# adapted from https://github.com/huggingface/setfit
from collections import Counter
import numpy as np
import os
import argparse

from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer, sample_dataset

from data import AUTH_KEY, DATA_PATH

def load_data(args):
    # Load a dataset from the Hugging Face Hub
    dataset = load_dataset(DATA_PATH, args.dataset_name, split="train", use_auth_token=AUTH_KEY)
    num_labels = max(sum(dataset['labels'], [])) + 1

    counts = Counter(sum(dataset['labels'],[]))
    drop_feats = []
    for feat in range(num_labels):
        if counts[feat] < 1:
            drop_feats.append(feat)

    def one_hot(example): 
        return {str(i):1 if i in example['labels'] else 0 for i in range(num_labels)}
    def encode_labels(record):
        return {"labels": [record[feature] for feature in features]}

    dataset = dataset.map(one_hot)
    features = dataset.column_names
    for feat in drop_feats:
        features.remove(str(feat))
    features.remove('text')
    features.remove('labels')
    features.remove('doc_id')

    num_samples = 16
    samples = np.concatenate([np.random.choice(np.where(dataset[f])[0], num_samples) for f in features])

    train_dataset = dataset.select(samples) #sample_dataset(dataset, label_column="labels_one_hot", num_samples=8)
    train_dataset = train_dataset.map(encode_labels)

    eval_dataset = load_dataset(DATA_PATH, args.dataset_name, split="validation", use_auth_token=AUTH_KEY)
    eval_dataset = eval_dataset.map(one_hot)
    eval_dataset = eval_dataset.map(encode_labels)
 
    return train_dataset, eval_dataset

def load_model(args):
    # Load a SetFit model from Hub
    model = SetFitModel.from_pretrained(
        args.model_name,
        use_differentiable_head=True,
        multi_target_strategy='multi-output'
        #head_params={"out_features": num_classes},
    )
    return model
    
def main(args):

    train_dataset, eval_dataset = load_data(args)
    model = load_model(args)
    
    # Create trainer
    trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=args.batch_size,
    num_iterations=20, # The number of text pairs to generate for contrastive learning
    num_epochs=args.num_epochs, # The number of epochs to use for contrastive learning
    column_mapping={"text": "text", "labels": "label"} # Map dataset columns to text/label expected by trainer
    )

    # Train and evaluate
    trainer.freeze() # Freeze the head
    trainer.train(
            learning_rate=args.learning_rate) # Train only the body
    trainer.model.save_pretrained(os.path.join(args.experiments_dir, args.experiment_name))
    # Download from Hub and run inference
    #model = SetFitModel.from_pretrained("lewtun/my-awesome-setfit-model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')

    parser.add_argument("--dataset_name", type=str, default='bioasq-l2', help="Name of dataset as stored on HF")
    parser.add_argument("--experiments_dir", type=str, default="retriever/models/", help="Directory where trained models will be saved")
    parser.add_argument("--experiment_name", type=str, default=None, help="Sub directory where trained models will be saved")

    parser.add_argument("--model_name", type=str, help="Directory where trained model is saved")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    #parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length")

    args = parser.parse_args()

    main(args)
