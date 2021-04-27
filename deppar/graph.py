# -*- coding: utf-8 -*-

import numpy as np


class Word:

    def __init__(self, word):
        self.word = word


class Sentence:

    def __init__(self, words, heads, rels):
        self.nodes = np.array([Word("*ROOT*")] + list(words))
        self.heads = np.array([-1] + list(heads))
        self.rels = np.array(["_"] + list(rels))
        self.pred_heads = np.array([-1] * len(self.heads))
        self.pred_rels = np.array(["_"] * len(self.rels))
