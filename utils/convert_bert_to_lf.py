import argparse
import random

import torch
import copy
import warnings
from data import DATA_DIR
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
warnings.filterwarnings("ignore")


def convert_bert_to_lf():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--bert_checkpoint', default='bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
                        help='Path to the pre-trained RoBERTa model directory')
    parser.add_argument('--output_model_path', default='bio-longformer-base',
                        help='Path to the pre-trained RoBERTa model directory')
    parser_config = parser.parse_args()
    BERT_CHECKPOINT = parser_config.bert_checkpoint

    # load pre-trained bert model and tokenizer
    bert_model = AutoModelForMaskedLM.from_pretrained(BERT_CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(BERT_CHECKPOINT, model_max_length=4096)

    # load dummy config and change specifications
    bert_config = bert_model.config
    lf_config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
    # Text length parameters
    lf_config.max_position_embeddings = 4096 + bert_config.pad_token_id + 2
    lf_config.model_max_length = 4096
    lf_config.num_hidden_layers = bert_config.num_hidden_layers
    # Transformer parameters
    lf_config.hidden_size = bert_config.hidden_size
    lf_config.intermediate_size = bert_config.intermediate_size
    lf_config.num_attention_heads = bert_config.num_attention_heads
    lf_config.hidden_act = bert_config.hidden_act
    lf_config.attention_window = [512] * lf_config.num_hidden_layers
    # Vocabulary parameters
    lf_config.vocab_size = bert_config.vocab_size
    lf_config.pad_token_id = bert_config.pad_token_id
    lf_config.bos_token_id = bert_config.bos_token_id
    lf_config.eos_token_id = bert_config.eos_token_id
    lf_config.cls_token_id = tokenizer.cls_token_id
    lf_config.sep_token_id = tokenizer.sep_token_id
    lf_config.type_vocab_size = bert_config.type_vocab_size

    # load dummy hi-transformer model
    lf_model = AutoModelForMaskedLM.from_config(lf_config)

    # copy embeddings
    lf_model.longformer.embeddings.position_embeddings.weight.data[0] = torch.zeros((bert_config.hidden_size,))
    k = 1
    step = bert_config.max_position_embeddings - 1
    while k < lf_config.max_position_embeddings - 1:
        if k + step >= lf_config.max_position_embeddings:
            remaining_tokens = len(lf_model.longformer.embeddings.position_embeddings.weight.data[k:])
            lf_model.longformer.embeddings.position_embeddings.weight.data[k:] = bert_model.bert.embeddings.position_embeddings.weight[1:(remaining_tokens + 1)]
        else:
            lf_model.longformer.embeddings.position_embeddings.weight.data[k:(k + step)] = bert_model.bert.embeddings.position_embeddings.weight[1:]
        k += step
    lf_model.longformer.embeddings.word_embeddings.load_state_dict(bert_model.bert.embeddings.word_embeddings.state_dict())
    lf_model.longformer.embeddings.token_type_embeddings.load_state_dict(bert_model.bert.embeddings.token_type_embeddings.state_dict())
    lf_model.longformer.embeddings.LayerNorm.load_state_dict(bert_model.bert.embeddings.LayerNorm.state_dict())

    # copy transformer layers
    for i in range(len(bert_model.bert.encoder.layer)):
        # generic
        lf_model.longformer.encoder.layer[i].intermediate.dense = copy.deepcopy(
            bert_model.bert.encoder.layer[i].intermediate.dense)
        lf_model.longformer.encoder.layer[i].output.dense = copy.deepcopy(
            bert_model.bert.encoder.layer[i].output.dense)
        lf_model.longformer.encoder.layer[i].output.LayerNorm = copy.deepcopy(
            bert_model.bert.encoder.layer[i].output.LayerNorm)
        # attention output
        lf_model.longformer.encoder.layer[i].attention.output.dense = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.output.dense)
        lf_model.longformer.encoder.layer[i].attention.output.LayerNorm = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.output.LayerNorm)
        # local q,k,v
        lf_model.longformer.encoder.layer[i].attention.self.query = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.value)
        # global q,k,v
        lf_model.longformer.encoder.layer[i].attention.self.query_global = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key_global = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value_global = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.value)

    # copy lm_head
    lf_model.lm_head.dense.load_state_dict(bert_model.cls.predictions.transform.dense.state_dict())
    lf_model.lm_head.layer_norm.load_state_dict(bert_model.cls.predictions.transform.LayerNorm.state_dict())
    lf_model.lm_head.decoder.load_state_dict(bert_model.cls.predictions.decoder.state_dict())
    lf_model.lm_head.bias = copy.deepcopy(bert_model.cls.predictions.bias)

    # save model
    lf_model.save_pretrained(f'{DATA_DIR}/{parser_config.output_model_path}')

    # save tokenizer
    tokenizer.save_pretrained(f'{DATA_DIR}/{parser_config.output_model_path}')

    print(f'BERT-based Longformer model is ready to run!')


if __name__ == '__main__':
    convert_bert_to_lf()
