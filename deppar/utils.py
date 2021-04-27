# -*- coding: utf-8 -*-

from collections import Counter

import torch


if torch.cuda.is_available():
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).cuda()
else:
    from torch import from_numpy


BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
}


def buildVocab(sents):
    relsCount = Counter()

    for sent in sents:
        relsCount.update(sent.rels[1:])

    print("Rel set containing {} tags".format(len(relsCount)), relsCount)

    ret = {
        "rels": list(relsCount.keys()),
    }

    return ret
