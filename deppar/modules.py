# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .attention import BilinearMatrixAttention
from .dropout import SharedDropout
from .algorithm import find_best_heap


class ParserModule(ABC):

    @property
    @classmethod
    @abstractmethod
    def name(cls):
        pass

    @staticmethod
    @abstractmethod
    def load_data(parser, graph):
        pass

    @staticmethod
    @abstractmethod
    def batch_label(batch):
        pass

    @abstractmethod
    def evaluate(self, graphs, parser, results, pred, gold, mask, batch, train=False):
        pass

    @abstractmethod
    def metrics(self, results):
        pass


class PointerSelector(nn.Module, ParserModule):

    def __init__(self, parser, hidden_size, dropout=0.):
        super(PointerSelector, self).__init__()
        print("build pointer selector ...", self.__class__.name)
        self.head_mlp = nn.Sequential(
            nn.Linear(parser._proj_dims, hidden_size),
            nn.ReLU(),
            SharedDropout(dropout),
        )

        self.dep_mlp = nn.Sequential(
            nn.Linear(parser._proj_dims, hidden_size),
            nn.ReLU(),
            SharedDropout(dropout),
        )

        self.attention = BilinearMatrixAttention(hidden_size, hidden_size, True)
        self.loss = nn.NLLLoss(ignore_index=-1, reduction='sum')

    def calculate_loss(self, seq_features, batch, finetune=None):
        batch_label = self.batch_label(batch)
        mask = batch["mask"]
        mask_h = self.batch_cand_mask(batch)

        heads = self.head_mlp(seq_features)
        deps = self.dep_mlp(seq_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask_att = mask.unsqueeze(2) * mask_h.unsqueeze(1)
        scores = self.attention(deps, heads).masked_fill((1-mask_att).bool(), float("-inf")).view(batch_size * seq_len, -1)
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq  = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len) + 1

        total_loss = self.loss(scores, (batch_label - 1).view(batch_size * seq_len))
        total_loss = total_loss / float(batch_size)

        return total_loss, tag_seq

    def forward(self, parser, seq_features, batch):
        batch_label = self.batch_label(batch)
        mask = batch["mask"]

        mask_h = self.batch_cand_mask(batch)
        heads = self.head_mlp(seq_features)
        deps = self.dep_mlp(seq_features)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask_att = mask.unsqueeze(2) * mask_h.unsqueeze(1)
        scores = self.attention(deps, heads).masked_fill((1-mask_att).bool(), float("-inf")).view(batch_size * seq_len, -1)
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq  = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len) + 1

        scores = scores.view(batch_size, seq_len, -1).cpu().data.numpy().astype('float64')
        for i in range(batch_size):
            l = len(batch["raw"][i])
            s = scores[i, :l, :l]
            heads = find_best_heap(s)
            tag_seq[i, :l] = torch.Tensor(np.array(heads) + 1)

        tag_seq = mask.long() * tag_seq
        batch["pred_head"] = tag_seq

        return tag_seq

    def scores(self, seq_features, batch):
        batch_label = self.batch_label(batch)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)
        mask = batch["mask"]
        mask_h = self.batch_cand_mask(batch)

        heads = self.head_mlp(seq_features)
        deps = self.dep_mlp(seq_features)

        mask_att = mask.unsqueeze(2) * mask_h.unsqueeze(1)
        scores = self.attention(deps, heads).masked_fill((1-mask_att).bool(), float("-inf")).view(batch_size * seq_len, -1)
        scores = F.log_softmax(scores, dim=1).view(batch_size, seq_len, -1)

        return scores

    def evaluate(self, results, parser, graphs, pred, gold, mask, batch, train=False):
        overlaped = (pred == gold)
        correct = float((overlaped * mask).sum())
        total = float(mask.sum())
        results["{}-c".format(self.__class__.name)] += correct
        results["{}-t".format(self.__class__.name)] += total

    def metrics(self, results):
        correct = results["{}-c".format(self.__class__.name)]
        total = results["{}-t".format(self.__class__.name)]
        results["metrics/{}-acc".format(self.__class__.name)] = correct / (total + 1e-10) * 100.
        del results["{}-c".format(self.__class__.name)]
        del results["{}-t".format(self.__class__.name)]

    @staticmethod
    @abstractmethod
    def batch_cand_mask(batch):
        pass


class HSelParser(PointerSelector):

    name = "HSel"

    @staticmethod
    def load_data(parser, graph):
        return {"head": graph.heads + 1}

    @staticmethod
    def batch_label(batch):
        return batch["head"]

    @staticmethod
    def batch_cand_mask(batch):
        return batch["mask_h"]


class RelLabeler(nn.Module, ParserModule):

    name = "Rel"

    def __init__(self, parser, hidden_size, dropout=0.):
        super(RelLabeler, self).__init__()
        print("build rel labeler...", self.__class__.name)
        self.head_mlp = nn.Sequential(
            nn.Linear(parser._proj_dims, hidden_size),
            nn.ReLU(),
            SharedDropout(dropout),
        )

        self.dep_mlp = nn.Sequential(
            nn.Linear(parser._proj_dims, hidden_size),
            nn.ReLU(),
            SharedDropout(dropout),
        )

        self.attention = nn.Bilinear(hidden_size, hidden_size, len(parser._rels), True)
        self.bias_x = nn.Linear(hidden_size, len(parser._rels), False)
        self.bias_y = nn.Linear(hidden_size, len(parser._rels), False)
        self.loss = nn.NLLLoss(ignore_index=-1, reduction='sum')
        self.parser = parser

    def calculate_loss(self, seq_features, batch, finetune=None):
        _, batch_label = self.batch_label(batch)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask = batch["mask"]
        head = torch.abs(batch["head"] - 1)

        ran = torch.arange(batch_size, device=head.device).unsqueeze(1) * seq_len
        idx = (head + ran).view(batch_size * seq_len)

        heads = self.head_mlp(seq_features).view(batch_size * seq_len, -1)[idx]
        deps = self.dep_mlp(seq_features).view(batch_size * seq_len, -1)

        scores = self.attention(deps, heads) + self.bias_x(heads) + self.bias_y(deps)
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq  = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)

        total_loss = self.loss(scores, batch_label.view(batch_size * seq_len))
        total_loss = total_loss / float(batch_size)

        return total_loss, (head, tag_seq)

    def forward(self, parser, seq_features, batch):
        _, batch_label = self.batch_label(batch)
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        mask = batch["mask"]
        if "pred_head" in batch:
            head = torch.abs(batch["pred_head"] - 1)
        else:
            head = torch.abs(batch["head"] - 1)

        ran = torch.arange(batch_size, device=head.device).unsqueeze(1) * seq_len
        idx = (head + ran).view(batch_size * seq_len)

        heads = self.head_mlp(seq_features).view(batch_size * seq_len, -1)[idx]
        deps = self.dep_mlp(seq_features).view(batch_size * seq_len, -1)

        scores = self.attention(deps, heads) + self.bias_x(heads) + self.bias_y(deps)
        scores = F.log_softmax(scores, dim=1)

        _, tag_seq  = torch.max(scores, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        tag_seq = mask.long() * tag_seq

        batch["pred_rel"] = tag_seq

        return head, tag_seq

    def scores(self, seq_features, batch):
        batch_label = self.batch_label(batch)[0]
        batch_size = batch_label.size(0)
        seq_len = batch_label.size(1)

        heads = self.head_mlp(seq_features)
        deps = self.dep_mlp(seq_features)

        ret = torch.matmul(deps.unsqueeze(1), self.attention.weight) @ torch.transpose(heads.unsqueeze(1), -1, -2)
        ret = ret + self.bias_x(heads).transpose(-1, -2).unsqueeze(-2)
        ret = ret + self.bias_y(deps).transpose(-1, -2).unsqueeze(-1)
        ret = ret + self.attention.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return F.log_softmax(ret, dim=1).transpose(-1, -3)

    def evaluate(self, results, parser, graphs, pred, gold, mask, batch, train=False):
        overlaped = (pred[0] == (gold[0] - 1)) * (pred[1] == gold[1])
        correct = float((overlaped * mask).sum())
        total = float(mask.sum())
        results["{}-c".format(self.__class__.name)] += correct
        results["{}-t".format(self.__class__.name)] += total

    def metrics(self, results):
        correct = results["{}-c".format(self.__class__.name)]
        total = results["{}-t".format(self.__class__.name)]
        results["metrics/{}-acc".format(self.__class__.name)] = correct / (total + 1e-10) * 100.
        del results["{}-c".format(self.__class__.name)]
        del results["{}-t".format(self.__class__.name)]

    @staticmethod
    def load_data(parser, graph):
        labels = [-1] + [parser._rels.get(r, -1) if h >= 0 else -1 for h, r in zip(graph.heads[1:], graph.rels[1:])]
        return {"rel": labels}

    @staticmethod
    def batch_label(batch):
        return batch["head"], batch["rel"]
