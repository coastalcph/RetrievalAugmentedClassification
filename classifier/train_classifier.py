#!/usr/bin/env python
# coding=utf-8
""" Finetuning Longformer-based Multi-Label Classifiers."""

import logging
import os
import random
import re
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report
from scipy.special import expit
import glob
import shutil
import json
import h5py

import transformers
from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from data import AUTH_KEY, DATA_DIR
from data_collator import DataCollatorForMultiLabelClassification
from ra_classifier import RABERTForSequenceClassification, RALongformerForSequenceClassification, RARoBERTaForSequenceClassification

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.20.0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

from tokenizers.normalizers import NFKD
from tokenizers.pre_tokenizers import WhitespaceSplit

normalizer = NFKD()
pre_tokenizer = WhitespaceSplit()


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="eurlex-l1",
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    embeddings_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path where the neighbor documents embeddings are saved."
        },
    )
    predictions_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path where the predictions are saved."
        },
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    do_train_eval: Optional[bool] = field(default=False, metadata={"help": "Evaluate on training set."})
    bootstrap_dataset: Optional[bool] = field(default=False, metadata={"help": "Use predictions on training set."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    encode_document: Optional[bool] = field(
        default=True, metadata={"help": "Whether to encode input document or not"}
    )
    freeze_encoder: Optional[bool] = field(
            default=False, metadata={"help": "If true, encoder is not finetuned"}
    )
    dec_layers: Optional[int] = field(
        default=1, metadata={"help": "Number of decoder layers for RA"}
    )
    dec_attention_heads: Optional[int] = field(
        default=1, metadata={"help": "Number of attention heads for RA"}
    )
    retrieval_augmentation: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use retrieval augmentation or not"}
    )
    finetune_retrieval: Optional[bool] = field(
            default=False, metadata={"help": "Whether to finetune the retrieval query encoder"}
    )
    no_neighbors: Optional[int] = field(
        default=16, metadata={"help": "Number of top K retrieved documents to be used"}
    )
    augment_with_documents: Optional[bool] = field(
            default=False, metadata={"help": "Whether to include the document embeddings of retrieved neighbors"}
    )    
    augment_with_labels: Optional[bool] = field(
            default=False, metadata={"help": "Whether to include the labels of retrieved neighbors"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: str = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def update_dataset_neighbors(dataset, datastore, doc2labels, embeddings_path, no_neighbors=16):
    # Augment with neighbors
    with open(embeddings_path) as filename:
        neighbors = json.load(filename)

    def add_neighbors(example):
        doc_neighbors = [neighbor['doc_id'] for neighbor in neighbors[example['doc_id']][:no_neighbors]]
        example["neighbor_embeddings"] = [datastore[doc_id][:] for doc_id in doc_neighbors]
        example["neighbor_labels"] = [doc2labels[doc_id] for doc_id in doc_neighbors]
        return example
    
    return dataset.map(add_neighbors, load_from_cache_file=False)


def update_dataset_bootstrap(dataset, predictions_path, document_ids):
    # Augment with neighbors
    with open(predictions_path) as file:
        predictions = json.load(file)

    def relabel_docs(example):
        if example['doc_id'] not in document_ids:
            example["labels"] = predictions[example['doc_id']]
        return example

    return dataset.map(relabel_docs)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    label_list = None

    if model_args.retrieval_augmentation and not model_args.finetune_retrieval:
        datastore = h5py.File(os.path.join(DATA_DIR, data_args.embeddings_path, 'corpus.hdf5'))

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading eurlex dataset from the hub.
    if training_args.do_train or training_args.do_predict or data_args.do_train_eval:
        train_dataset = load_dataset('kiddothe2b/multilabel_bench', data_args.dataset_name, split="train",
                                     cache_dir=model_args.cache_dir, use_auth_token=AUTH_KEY)

        # Labels
        label_list = list(
            range(train_dataset.features['labels'].feature.num_classes))
        labels_codes = train_dataset.features['labels'].feature.names
        num_labels = len(label_list)

    if training_args.do_eval:
        eval_dataset = load_dataset('kiddothe2b/multilabel_bench', data_args.dataset_name, split="validation",
                                    cache_dir=model_args.cache_dir, use_auth_token=AUTH_KEY)

        if label_list is None:
            # Labels
            label_list = list(
                range(eval_dataset.features['labels'].feature.num_classes))
            labels_codes = eval_dataset.features['labels'].feature.names
            num_labels = len(label_list)

    if training_args.do_predict:
        predict_dataset = load_dataset('kiddothe2b/multilabel_bench', data_args.dataset_name, split="test",
                                       cache_dir=model_args.cache_dir, use_auth_token=AUTH_KEY)

        if label_list is None:
            # Labels
            label_list = list(
                range(predict_dataset.features['labels'].feature.num_classes))
            labels_codes = predict_dataset.features['labels'].feature.names
            num_labels = len(label_list)

    # Label descriptors mode
    label_desc2id = {label_desc: idx for idx, label_desc in enumerate(labels_codes)}

    print(f'LabelDesc2Id: {label_desc2id}')

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id={l: i for i, l in enumerate(labels_codes)},
        id2label={i: l for i, l in enumerate(labels_codes)},
        use_auth_token=model_args.use_auth_token,
        finetuning_task=data_args.dataset_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    # add retrieval_augmentation param in config
    if model_args.retrieval_augmentation:
        assert model_args.augment_with_documents or model_args.augment_with_labels, "Specify at least one of 'documents' or 'labels' for RA"
    if model_args.finetune_retrieval: 
        assert not model_args.augment_with_labels, "Finetuned retrieval with label augmentation is currently not implemented."

    if model_args.finetune_retrieval:
        config.full_embeddings_path = os.path.join(DATA_DIR, data_args.embeddings_path, 'corpus.hdf5')
    config.retrieval_augmentation = model_args.retrieval_augmentation
    config.finetune_retrieval = model_args.finetune_retrieval
    config.no_neighbors = model_args.no_neighbors
    config.augment_with_documents = model_args.augment_with_documents
    config.augment_with_labels = model_args.augment_with_labels
    config.encode_document = model_args.encode_document
    config.dec_layers = model_args.dec_layers
    config.dec_attention_heads = model_args.dec_attention_heads

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        use_auth_token=model_args.use_auth_token,
        revision=model_args.model_revision,
    )

    if config.model_type == 'longformer':
        config.attention_window = [128] * config.num_hidden_layers
        model = RALongformerForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            use_auth_token=model_args.use_auth_token,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
    elif config.model_type == 'roberta':
        model = RARoBERTaForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            use_auth_token=model_args.use_auth_token,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
    elif config.model_type == 'bert':
        model = RABERTForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            use_auth_token=model_args.use_auth_token,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
        )
    else:
        raise NotImplementedError(f'Models of type {config.model_type} are not supported!!!')
    
    if model_args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        if 'mimic' in data_args.dataset_name:
            for idx, text in enumerate(examples['text']):
                if 'Service:' in text:
                    text = 'Service:' + re.split('Service:', text, maxsplit=1)[1]
                elif 'Sex:' in text:
                    text = re.split('\n', re.split('Sex:', text, maxsplit=1)[1], maxsplit=1)[1]
                text = normalizer.normalize_str(text)
                text = ' '.join([token[0] for token in pre_tokenizer.pre_tokenize_str(text)])
                text = re.sub('[^a-z ]{2,}', ' ', text, flags=re.IGNORECASE)
                text = re.sub(' +', ' ', text, flags=re.IGNORECASE)
                examples['text'][idx] = text

        # Tokenize the texts
        batch = tokenizer(
            examples["text"],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

        if 'longformer' in model_args.model_name_or_path:
            batch["global_attention_mask"] = np.zeros((len(examples["text"]), data_args.max_seq_length), dtype=int)
            batch["global_attention_mask"][:, 0] = 1
            batch["global_attention_mask"] = list(batch["global_attention_mask"])

        if model_args.retrieval_augmentation and not model_args.finetune_retrieval:
            if model_args.augment_with_documents:
                batch["decoder_input_ids"] = examples["neighbor_embeddings"]
            if model_args.augment_with_labels:
                batch_neighbor_labels = []
                for neighbor_labels in examples["neighbor_labels"]:
                    batch_neighbor_labels.append([[1.0 if label in labels else 0.0 for label in label_list] for labels in neighbor_labels])
                batch["decoder_manyhot_ids"] = batch_neighbor_labels
            
            batch["decoder_attention_mask"] = np.ones((len(examples["neighbor_embeddings"]), model_args.no_neighbors), dtype=int)

        if not model_args.encode_document:
            batch["input_embeds"] = examples['doc_embedding']

        batch["label_ids"] = [[1.0 if label in labels else 0.0 for label in label_list] for labels in examples["labels"]]
        batch['labels'] = batch['label_ids']

        return batch

    doc2labels = {item['doc_id']: item['labels'] for item in train_dataset}
    if data_args.max_train_samples is not None:
        random.seed(42)
        sample_ids = random.sample(range(len(train_dataset)), k=min(data_args.max_train_samples, len(train_dataset)))
        train_dataset = train_dataset.select(sample_ids)

    if training_args.do_train or data_args.do_train_eval:
        if model_args.retrieval_augmentation and not model_args.finetune_retrieval:
            train_dataset = update_dataset_neighbors(train_dataset, datastore, doc2labels,
                                                     embeddings_path=os.path.join(DATA_DIR, data_args.embeddings_path, 'train.json'),
                                                     no_neighbors=model_args.no_neighbors)
        if data_args.bootstrap_dataset:
            random.seed(42)
            sample_ids = random.sample(range(len(train_dataset)), k=min(data_args.max_train_samples, len(train_dataset)))
            train_subset = train_dataset.select(sample_ids)
            doc_ids = train_subset['doc_id']
            train_dataset = update_dataset_bootstrap(train_dataset,
                                                     predictions_path=os.path.join(DATA_DIR, data_args.predictions_path, 'train_predictions.json'),
                                                     document_ids=doc_ids
                                                     )
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['labels', 'text'],
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        if model_args.retrieval_augmentation and not model_args.finetune_retrieval:
            eval_dataset = update_dataset_neighbors(eval_dataset, datastore, doc2labels,
                                                    embeddings_path=os.path.join(DATA_DIR, data_args.embeddings_path, 'validation.json'),
                                                    no_neighbors=model_args.no_neighbors)
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['labels', 'text'],
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        if model_args.retrieval_augmentation and not model_args.finetune_retrieval:
            predict_dataset = update_dataset_neighbors(predict_dataset, datastore, doc2labels,
                                                       embeddings_path=os.path.join(DATA_DIR, data_args.embeddings_path, 'test.json'),
                                                       no_neighbors=model_args.no_neighbors)
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['labels', 'text'],
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = (expit(logits) > 0.5).astype('int32')
        macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
        return {'macro-f1': macro_f1, 'micro-f1': micro_f1}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultiLabelClassification(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if data_args.do_train_eval:
        logger.info("*** Predict training ***")
        predictions, labels, _ = trainer.predict(train_dataset, metric_key_prefix="predict")
        hard_predictions = (expit(predictions) > 0.5).astype('int32')

        output_predict_file = os.path.join(DATA_DIR, training_args.output_dir, "train_predictions.json")
        predictions = {}
        if not os.path.exists(os.path.join(DATA_DIR, training_args.output_dir)):
            os.mkdir(os.path.join(DATA_DIR, training_args.output_dir))
        if trainer.is_world_process_zero():
            for sample, hard_p in zip(train_dataset, hard_predictions):
                predictions[sample['doc_id']] = [idx for idx in label_list if hard_p[idx] == 1]
            with open(output_predict_file, "w") as writer:
                json.dump(predictions, writer)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        hard_predictions = (expit(predictions) > 0.5).astype('int32')

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        report_predict_file = os.path.join(training_args.output_dir, "test_classification_report.txt")
        output_predict_file = os.path.join(training_args.output_dir, "test_predictions.csv")
        if trainer.is_world_process_zero():
            cls_report = classification_report(y_true=labels, y_pred=hard_predictions,
                                               target_names=list(config.label2id.keys()),
                                               zero_division=0)
            cls_report_dict = classification_report(y_true=labels, y_pred=hard_predictions,
                                               target_names=list(config.label2id.keys()),
                                               zero_division=0, output_dict=True)

            # compute binned metrics
            thresholds = [10, 100, 500, 1000, 10000]
            train_labels = sum(train_dataset['labels'] ,[])
            bin_data = {threshold: [] for threshold in thresholds}
            for label_id, label_name in zip(label_list, labels_codes):
                count = train_labels.count(label_id)
                label_entry = cls_report_dict[label_name]
                for threshold in thresholds:
                    if count <= threshold:
                        bin_data[threshold].append(label_entry)
                        break

            def append_to_line(ave_score):
                formatted_score = '{}'.format(ave_score)
                return ' '*(10 - len(formatted_score)) + formatted_score
                
            for threshold in thresholds:
                line = '{} macro '.format(threshold)
                line = ' '*(13 - len(line)) + line
                for metric in ['precision', 'recall', 'f1-score']:
                    scores = [item[metric] for item in bin_data[threshold]]
                    ave_score = np.round(np.average(scores), 2)
                    line += append_to_line(ave_score)
                support = sum([item['support'] for item in bin_data[threshold]])
                line += append_to_line(support)
                cls_report += '\n' + line

            with open(report_predict_file, "w") as writer:
                writer.write(cls_report)
            with open(output_predict_file, "w") as writer:
                try:
                    for index, pred_list in enumerate(predictions[0]):
                        pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
                        writer.write(f"{index}\t{pred_line}\n")
                except:
                    try:
                        for index, pred_list in enumerate(predictions):
                            pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
                            writer.write(f"{index}\t{pred_line}\n")
                    except:
                        pass

            logger.info(cls_report)

    # Clean up checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


if __name__ == "__main__":
    main()
