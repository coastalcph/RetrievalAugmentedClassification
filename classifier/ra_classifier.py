# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model."""
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Tuple, Union
import torch
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel, \
    LongformerModel, LongformerClassificationHead, LongformerSequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertPreTrainedModel, SequenceClassifierOutput, BertModel
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from transformers.models.led.modeling_led import LEDDecoder, LEDConfig


class RALongformerForSequenceClassification(LongformerPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"ra_decoder",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"ra_decoder",
    ]

    def __init__(self, config):
        super().__init__(config)

        # document encoder
        self.longformer = LongformerModel(config, add_pooling_layer=False)

        # retrieval-augmented decoder
        self.ra_config = LEDConfig()
        self.ra_config.d_model = config.hidden_size
        self.ra_config.decoder_ffn_dim = config.hidden_size
        self.ra_config.decoder_attention_heads = 1
        self.ra_config.decoder_layers = 1
        self.ra_decoder = LEDDecoder(self.ra_config)

        # classifier
        self.num_labels = config.num_labels
        self.classifier = LongformerClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            global_attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerSequenceClassifierOutput]:
        r"""
                labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                    Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                    config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                    `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
                """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        encoder_outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_cls_output = torch.unsqueeze(encoder_outputs[0][:, 0, :], dim=1)
        sequence_cls_mask = torch.ones_like(sequence_cls_output).to(sequence_cls_output.device)

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.ra_decoder(
            input_ids=None,
            attention_mask=sequence_cls_mask,
            encoder_hidden_states=decoder_input_ids,
            encoder_attention_mask=decoder_attention_mask,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=sequence_cls_output,
            use_cache=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + decoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            global_attentions=decoder_outputs.global_attentions,
        )


class RABERTForSequenceClassification(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"ra_decoder", r"cls", r"classifier",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"ra_decoder", r"cls",  r"classifier",
    ]

    def __init__(self, config):
        super().__init__(config)

        # document encoder
        self.bert = BertModel(config, add_pooling_layer=False)

        self.retrieval_augmentation = config.retrieval_augmentation
        if config.retrieval_augmentation:
            # retrieval-augmented decoder
            self.ra_config = LEDConfig()
            self.ra_config.d_model = config.hidden_size
            self.ra_config.decoder_ffn_dim = config.hidden_size
            self.ra_config.decoder_attention_heads = 1
            self.ra_config.decoder_layers = 1
            self.ra_decoder = LEDDecoder(self.ra_config)

        # classifier
        self.num_labels = config.num_labels
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            decoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
                labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                    Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                    config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                    `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
                """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.retrieval_augmentation:
            sequence_cls_output = torch.unsqueeze(encoder_outputs[0][:, 0, :], dim=1)
            sequence_cls_mask = torch.ones_like(sequence_cls_output).to(sequence_cls_output.device)

            # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
            decoder_outputs = self.ra_decoder(
                input_ids=None,
                attention_mask=sequence_cls_mask,
                encoder_hidden_states=decoder_input_ids,
                encoder_attention_mask=decoder_attention_mask,
                head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                inputs_embeds=sequence_cls_output,
                use_cache=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = decoder_outputs[0]
        else:
            sequence_output = encoder_outputs[0]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + decoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=decoder_outputs.hidden_states if self.retrieval_augmentation else encoder_outputs.hidden_states,
            attentions=decoder_outputs.cross_attentions if self.retrieval_augmentation else encoder_outputs.attentions,
        )


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import numpy as np
    # model = RALongformerForSequenceClassification.from_pretrained('kiddothe2b/legal-longformer-base')
    # tokenizer = AutoTokenizer.from_pretrained('kiddothe2b/legal-longformer-base')
    # inputs = tokenizer(['dog ' * 1024 for _ in range(3)], truncation=True, max_length=1024,
    #                    padding='max_length', return_tensors='pt')
    # decode_inputs = torch.tensor(np.zeros((3, 8, 768)), dtype=torch.float32)
    # decoder_attention_mask = torch.tensor(np.ones((3, 8)), dtype=torch.int32)
    # model(inputs['input_ids'],
    #       attention_mask=inputs['attention_mask'],
    #       decoder_input_ids=decode_inputs,
    #       decoder_attention_mask=decoder_attention_mask,
    #       labels=torch.zeros(len(inputs['input_ids']), 20))
    # print()

    model = RABERTForSequenceClassification.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
    tokenizer = AutoTokenizer.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
    inputs = tokenizer(['dog ' * 256 for _ in range(3)], truncation=True, max_length=256,
                       padding='max_length', return_tensors='pt')
    decode_inputs = torch.tensor(np.zeros((3, 8, 768)), dtype=torch.float32)
    decoder_attention_mask = torch.tensor(np.ones((3, 8)), dtype=torch.int32)
    model(inputs['input_ids'],
          attention_mask=inputs['attention_mask'],
          decoder_input_ids=decode_inputs,
          decoder_attention_mask=decoder_attention_mask,
          labels=torch.zeros(len(inputs['input_ids']), 20))
    print()
