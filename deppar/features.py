# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer

from .utils import BERT_TOKEN_MAPPING, from_numpy


class WordSequence(nn.Module):

    def __init__(self, parser):
        super(WordSequence, self).__init__()

        self.bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.bert_model = AutoModel.from_pretrained("xlm-roberta-base")
        self.bert_project = nn.Linear(self.bert_model.pooler.dense.in_features, parser._proj_dims, bias=False)

    def forward(self, batch):
        raw = batch["raw"]
        seq_max_len = max([len(x) for x in raw])

        all_input_ids = np.zeros((len(raw), 2048), dtype=int)
        all_input_type_ids = np.zeros((len(raw), 2048), dtype=int)
        all_input_mask = np.zeros((len(raw), 2048), dtype=int)
        all_word_end_mask = np.zeros((len(raw), 2048), dtype=int)

        subword_max_len = 0

        for snum, sentence in enumerate(raw):
            tokens = []
            word_end_mask = []

            tokens.append("[CLS]")
            word_end_mask.append(1)

            cleaned_words = []
            for word in sentence[1:]:
                word = BERT_TOKEN_MAPPING.get(word, word)
                if word == "n't" and cleaned_words:
                    cleaned_words[-1] = cleaned_words[-1] + "n"
                    word = "'t"
                cleaned_words.append(word)

            for i, word in enumerate(cleaned_words):
                word_tokens = self.bert_tokenizer.tokenize(word)
                if len(word_tokens) == 0:
                    word_tokens = ['[MASK]']
                for _ in range(len(word_tokens)):
                    word_end_mask.append(0)
                word_end_mask[-1] = 1
                tokens.extend(word_tokens)

            tokens.append("[SEP]")

            for i in range(seq_max_len - len(sentence)):
                word_end_mask.append(1)

            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            subword_max_len = max(subword_max_len, len(word_end_mask) + 1)

            all_input_ids[snum, :len(input_ids)] = input_ids
            all_input_mask[snum, :len(input_mask)] = input_mask
            all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask

        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids[:, :subword_max_len])).to(batch['mask'].device)
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask[:, :subword_max_len])).to(batch['mask'].device)
        all_word_end_mask = from_numpy(np.ascontiguousarray(all_word_end_mask[:, :subword_max_len])).to(batch['mask'].device)
        _, _, all_encoder_layers = self.bert_model(all_input_ids, attention_mask=all_input_mask, output_hidden_states=True, return_dict=False)

        features = all_encoder_layers[-2]

        features_packed = features.masked_select(all_word_end_mask.to(torch.bool).unsqueeze(-1)).reshape(len(raw), seq_max_len, features.shape[-1])

        outputs = self.bert_project(features_packed)

        return outputs

    @staticmethod
    def load_data(parser, graph, pred=False):
        raw = [n.word for n in graph.nodes[:]]

        return {'raw': raw}
