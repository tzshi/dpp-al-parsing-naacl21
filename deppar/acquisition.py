# -*- coding: utf-8 -*-

import random
from collections import defaultdict, Counter
from abc import ABC

import torch
import scipy.linalg
import numpy as np
import sklearn
from torch.utils.data import DataLoader
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import pairwise

from .algorithm import find_best_heap
from .dpp import dpp_greedy_map, dpp_greedy_map_tokens
from .data import DataCollate


def fill_budget_full(lengths, lst, budget):
    ret = []
    cum = 0
    cur = 0
    while cur < len(lst):
        length = lengths[lst[cur]]
        if length + cum <= budget:
            ret.append(lst[cur])
            cur += 1
            cum += length
        else:
            break
    return ret


def find_best_parse(ret, hsel_scores, rel_scores):
    if "p_heads" in ret and "p_rels" in ret:
        return

    p_heads = find_best_heap(hsel_scores)
    p_rels = [-1]
    for i in range(1, len(hsel_scores)):
        r = np.argmax(rel_scores[p_heads[i], i])
        p_rels.append(r)

    ret["p_heads"] = p_heads
    ret["p_rels"] = p_rels


def find_marginal(ret, hsel_scores, rel_scores):
    if "marginal" in ret:
        return

    A = np.exp(hsel_scores).T
    for i in range(len(hsel_scores)):
        A[i, i] = 0.
    L = np.zeros_like(hsel_scores)
    for h in range(len(hsel_scores)):
        for m in range(len(hsel_scores)):
            if h == m:
                L[h, m] = np.sum(A[:, m])
            else:
                L[h, m] = -A[h, m]

    L_inv = scipy.linalg.inv(L[1:, 1:])
    marginal = np.zeros_like(hsel_scores)

    for h in range(len(hsel_scores)):
        for m in range(1, len(hsel_scores)):
            if h == 0:
                marginal[h, m] = A[0, m] * L_inv[m - 1, m - 1]
            else:
                marginal[h, m] = A[h, m] * L_inv[m - 1, m - 1] - A[h, m] * L_inv[m - 1, h - 1]

    ret["marginal"] = marginal


class QualityFunction(ABC):

    def __init__(self):
        print("Quality function selected", self.__class__.__name__)
        pass

    @staticmethod
    def get_function(func):
        if func == "amp":
            return AMPQualityFunction()
        if func == "bald":
            return BALDQualityFunction()
        else:
            return RandomQualityFunction()


class RandomQualityFunction(QualityFunction):

    @staticmethod
    def sentence_step(ret, hsel_scores, rel_scores, seq_features):
        ret["r_q"] = random.random()

    @staticmethod
    def sentence_aggregate(rets):
        ret = np.array([x["r_q"] for x in rets])
        return ret

    @staticmethod
    def token_step(ret, i, hsel_scores, rel_scores, seq_features):
        ret["r_q"] = random.random()

    @staticmethod
    def token_aggregate(rets):
        ret = np.array([x["r_q"] for x in rets])
        return ret


class AMPQualityFunction(QualityFunction):

    @staticmethod
    def sentence_step(ret, hsel_scores, rel_scores, seq_features):
        find_best_parse(ret, hsel_scores, rel_scores)
        find_marginal(ret, hsel_scores, rel_scores)

        p_heads = ret["p_heads"]
        marginal = ret["marginal"]
        probs = []
        for j in range(1, len(hsel_scores)):
            probs.append(abs(1. - marginal[p_heads[j], j]) + 1e-10)
        prob = sum(probs) / len(probs)
        ret["amp_q"] = abs(prob)

    @staticmethod
    def sentence_aggregate(rets):
        ret = np.array([x["amp_q"] for x in rets])
        return ret

    @staticmethod
    def token_step(ret, i, hsel_scores, rel_scores, seq_features):
        find_best_parse(ret, hsel_scores, rel_scores)
        find_marginal(ret, hsel_scores, rel_scores)

        p_heads = ret["p_heads"]
        marginal = ret["marginal"]

        ret["amp_q"] = abs(1. - marginal[p_heads[i], i]) + 1e-10

    @staticmethod
    def token_aggregate(rets):
        ret = np.array([x["amp_q"] for x in rets])
        return ret


class BALDQualityFunction(QualityFunction):

    @staticmethod
    def sentence_step(ret, hsel_scores, rel_scores, seq_features):
        p_heads = find_best_heap(hsel_scores)

        if "bald_heads" in ret:
            ret["bald_heads"].append(p_heads)
        else:
            ret["bald_heads"] = [p_heads]

    @staticmethod
    def sentence_aggregate(rets):
        quality = []
        for x in rets:
            p_heads = x["bald_heads"]
            length = len(p_heads[0])
            vals = []
            for i in range(1, length):
                counter = Counter([y[i] for y in p_heads])
                vals.append(1. - counter.most_common(1)[0][1] / len(p_heads))
            quality.append(np.mean(vals))

        return np.array(quality)

    @staticmethod
    def token_step(ret, sent_ret, i, hsel_scores, rel_scores, seq_features):
        p_heads = sent_ret["bald_heads"]
        counter = Counter([y[i] for y in p_heads])
        ret["bald_q"] = 1. - counter.most_common(1)[0][1] / len(p_heads)

    @staticmethod
    def token_aggregate(rets):
        ret = np.array([x["bald_q"] for x in rets])

        return ret


class DiversityFunction(ABC):

    def __init__(self):
        print("Diversity function selected", self.__class__.__name__)
        pass

    @staticmethod
    def get_function(func):
        if func == "subtree":
            return SubTreeDiversityFunction()
        if func == "avg":
            return AvgDiversityFunction()
        else:
            return NoopDiversityFunction()


class NoopDiversityFunction(DiversityFunction):

    @staticmethod
    def sentence_step(ret, hsel_scores, rel_scores, seq_features):
        return

    @staticmethod
    def sentence_aggregate(rets):
        return

    @staticmethod
    def token_step(ret, i, hsel_scores, rel_scores, seq_features):
        return

    @staticmethod
    def token_aggregate(rets):
        return


class SubTreeDiversityFunction(DiversityFunction):

    @staticmethod
    def sentence_step(ret, hsel_scores, rel_scores, seq_features):
        find_best_parse(ret, hsel_scores, rel_scores)
        p_heads = ret["p_heads"]
        p_rels = ret["p_rels"]

        dic = {}
        for ii in range(1, len(p_heads)):
            tt = (p_rels[p_heads[ii]], p_rels[ii], "L" if p_heads[ii] < ii else "R")
            if tt in dic:
                dic[tt] += 1.
            else:
                dic[tt] = 1.
        ret["st_div"] = dic

    @staticmethod
    def sentence_aggregate(rets):
        # hand-crafted features
        features = [x["st_div"] for x in rets]
        featurizer = DictVectorizer()
        diversity = featurizer.fit_transform(features).todense()
        diversity = sklearn.feature_extraction.text.TfidfTransformer().fit_transform(diversity).todense()

        # L2 normalize
        diversity = sklearn.preprocessing.normalize(diversity)

        return diversity


class AvgDiversityFunction(DiversityFunction):

    @staticmethod
    def sentence_step(ret, hsel_scores, rel_scores, seq_features):
        ret["avg_div"] = np.average(seq_features[1:], axis=0)

    @staticmethod
    def sentence_aggregate(rets):
        # use pooled features
        diversity = np.stack([x["avg_div"] for x in rets], axis=0)

        # L2 normalize
        diversity = sklearn.preprocessing.normalize(diversity)

        return diversity

    @staticmethod
    def token_step(ret, i, hsel_scores, rel_scores, seq_features):
        ret["avg_div"] = seq_features[i]

    @staticmethod
    def token_aggregate(rets):
        diversity = np.stack([x["avg_div"] for x in rets], axis=0)
        diversity = np.array(diversity)
        diversity = sklearn.preprocessing.normalize(diversity)
        return diversity


class QualityAdjustmentFunction(ABC):

    def __init__(self):
        print("Quality adjustment function selected", self.__class__.__name__)
        pass

    @staticmethod
    def get_function(func):
        if func == "id":
            return InformationDensityFunction()
        else:
            return NoopAdjustmentFunction()


class NoopAdjustmentFunction(QualityAdjustmentFunction):

    @staticmethod
    def adjust(quality, diversity):
        return quality, diversity


class InformationDensityFunction(QualityAdjustmentFunction):

    @staticmethod
    def adjust(quality, diversity):
        sims = pairwise.cosine_similarity(diversity)
        sims = np.mean(sims, axis=1)
        return quality * sims, diversity


class ComboMethod(ABC):

    def __init__(self):
        print("Diversity method selected", self.__class__.__name__)
        pass

    @staticmethod
    def get_function(func):
        if func == "dpp":
            return DPPComboMethod()
        else:
            return TopKComboMethod()


class DPPComboMethod(ComboMethod):

    @staticmethod
    def sentence_selection(quality, diversity, lengths, sample_size):
        quality = abs(quality) ** 0.5
        samples = dpp_greedy_map_tokens(quality, diversity, sample_size, lengths)
        return samples

    @staticmethod
    def token_selection(quality, diversity, sample_size):
        factor = 1.0
        quality = abs(quality) ** 0.5
        samples = dpp_greedy_map(quality, diversity, sample_size)
        return samples


class TopKComboMethod(ComboMethod):

    @staticmethod
    def sentence_selection(quality, diversity, lengths, sample_size):
        # quality -- the higher the better
        cands = np.argsort(-quality)
        samples = fill_budget_full(lengths, cands, sample_size)
        return samples

    @staticmethod
    def token_selection(quality, diversity, sample_size):
        cands = np.argsort(-quality)
        samples = cands[:sample_size]
        return samples


class PartialAcquisitionProcedure():

    def __init__(self, quality="random", diversity="none", adjust_method="noop", combo_method="topk"):
        self.quality = QualityFunction.get_function(quality)
        self.repeat = 5 if quality == "bald" else 1
        self.diversity = DiversityFunction.get_function(diversity)
        self.token_diversity = DiversityFunction.get_function("avg")
        self.adjust_method = QualityAdjustmentFunction.get_function(adjust_method)
        self.combo_method = ComboMethod.get_function(combo_method)
        self.last_picked = []

    def step(self, train_graphs, parser, sample_size):
        ratio = 5.

        train_graphs.labeled = False
        parser._model.eval()
        loader = DataLoader(train_graphs, batch_size=parser._eval_batch_size,
                            shuffle=False, num_workers=1,
                            collate_fn=DataCollate(parser))

        hsel_module = parser._hsel_parser
        rel_module = parser._rel_labeler

        cands = defaultdict(dict)

        with torch.no_grad():
            for batch in loader:
                if parser._gpu:
                    for k in batch:
                        if k != "graphidx" and k != "raw":
                            batch[k] = batch[k].cuda()

                graphidx = batch["graphidx"]
                batch_graphs = [train_graphs.graphs[idx] for idx in batch["graphidx"]]

                seq_features = parser._seqrep(batch)
                hsel_scores = hsel_module.scores(seq_features, batch).cpu().data.numpy().astype("float64")
                rel_scores = rel_module.scores(seq_features, batch).cpu().data.numpy().astype("float64")
                seq_features = seq_features.cpu().data.numpy().astype("float64")

                for idx, g, f_s, h_s, r_s in zip(graphidx, batch_graphs, seq_features, hsel_scores, rel_scores):
                    length = len(g.nodes)

                    h_s = h_s[:length, :length]
                    r_s = r_s[:length, :length, :]
                    f_s = f_s[:length]

                    if self.repeat == 1:
                        self.quality.sentence_step(cands[idx], h_s, r_s, f_s)
                    self.diversity.sentence_step(cands[idx], h_s, r_s, f_s)

                if self.repeat > 1:
                    parser._model.train()
                    for repeat in range(self.repeat):
                        seq_features = parser._seqrep(batch)
                        hsel_scores = hsel_module.scores(seq_features, batch).cpu().data.numpy().astype("float64")
                        rel_scores = rel_module.scores(seq_features, batch).cpu().data.numpy().astype("float64")
                        seq_features = seq_features.cpu().data.numpy().astype("float64")

                        for idx, g, f_s, h_s, r_s in zip(graphidx, batch_graphs, seq_features, hsel_scores, rel_scores):
                            length = len(g.nodes)

                            h_s = h_s[:length, :length]
                            r_s = r_s[:length, :length, :]
                            f_s = f_s[:length]

                            self.quality.sentence_step(cands[idx], h_s, r_s, f_s)

                    parser._model.eval()

        cands = sorted(list(cands.items()))
        idxs = [x[0] for x in cands]
        rets = [x[1] for x in cands]
        lengths = [len(train_graphs.graphs[x].nodes) - 1 for x in idxs]

        quality = self.quality.sentence_aggregate(rets)
        diversity = self.diversity.sentence_aggregate(rets)
        quality, diversity = self.adjust_method.adjust(quality, diversity)

        samples = self.combo_method.sentence_selection(quality, diversity, lengths, int(sample_size * ratio))

        samples = [idxs[x] for x in samples]

        # second stage: token selection
        train_graphs.target_lst = samples
        train_graphs.targeted = True
        loader = DataLoader(train_graphs, batch_size=parser._eval_batch_size,
                            shuffle=False, num_workers=1,
                            collate_fn=DataCollate(parser))

        cands = defaultdict(dict)
        sent_cands = defaultdict(dict)
        with torch.no_grad():
            for batch in loader:
                if parser._gpu:
                    for k in batch:
                        if k != "graphidx" and k != "raw":
                            batch[k] = batch[k].cuda()

                graphidx = batch["graphidx"]
                batch_graphs = [train_graphs.graphs[idx] for idx in batch["graphidx"]]

                seq_features = parser._seqrep(batch)
                hsel_scores = hsel_module.scores(seq_features, batch).cpu().data.numpy().astype("float64")
                rel_scores = rel_module.scores(seq_features, batch).cpu().data.numpy().astype("float64")
                seq_features = seq_features.cpu().data.numpy().astype("float64")

                for idx, g, f_s, h_s, r_s in zip(graphidx, batch_graphs, seq_features, hsel_scores, rel_scores):
                    length = len(g.nodes)

                    h_s = h_s[:length, :length]
                    r_s = r_s[:length, :length, :]
                    f_s = f_s[:length]

                    for j in range(1, length):
                        if self.repeat == 1:
                            self.quality.token_step(cands[(idx, j)], j, h_s, r_s, f_s)
                        self.token_diversity.token_step(cands[(idx, j)], j, h_s, r_s, f_s)

                if self.repeat > 1:
                    parser._model.train()
                    for repeat in range(self.repeat):
                        seq_features = parser._seqrep(batch)
                        hsel_scores = hsel_module.scores(seq_features, batch).cpu().data.numpy().astype("float64")
                        rel_scores = rel_module.scores(seq_features, batch).cpu().data.numpy().astype("float64")
                        seq_features = seq_features.cpu().data.numpy().astype("float64")

                        for idx, g, f_s, h_s, r_s in zip(graphidx, batch_graphs, seq_features, hsel_scores, rel_scores):
                            length = len(g.nodes)

                            h_s = h_s[:length, :length]
                            r_s = r_s[:length, :length, :]
                            f_s = f_s[:length]

                            self.quality.sentence_step(sent_cands[idx], h_s, r_s, f_s)

                    for idx, g, f_s, h_s, r_s in zip(graphidx, batch_graphs, seq_features, hsel_scores, rel_scores):
                        length = len(g.nodes)
                        for j in range(1, length):
                            self.quality.token_step(cands[(idx, j)], sent_cands[idx], j, h_s, r_s, f_s)

                    parser._model.eval()

        train_graphs.targeted = False

        cands = sorted(list(cands.items()))
        idxs = [x[0] for x in cands]
        rets = [x[1] for x in cands]

        quality = self.quality.token_aggregate(rets)
        diversity = self.token_diversity.token_aggregate(rets)
        quality, diversity = self.adjust_method.adjust(quality, diversity)

        samples = self.combo_method.token_selection(quality, diversity, sample_size)
        samples = [idxs[x] for x in samples]

        partial_lst = defaultdict(list)
        for sent_id, tok_id in samples:
            partial_lst[sent_id].append(tok_id)

        partial_lst = partial_lst.items()
        self.last_picked = [x[0] for x in partial_lst]
        train_graphs.partial_label([x[0] for x in partial_lst], [x[1] for x in partial_lst])
        train_graphs.labeled = True

        return
