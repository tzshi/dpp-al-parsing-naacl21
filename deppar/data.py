# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .io import read_conll


class DataProcessor(Dataset):

    def __init__(self, filename, parser, modules, duplicate=1, active=False):
        print("Duplicated", duplicate, "times")
        data = []
        for i in range(duplicate):
            data.extend(read_conll(filename))

        self.graphs = data
        print("Read", filename, len(data), "sentences", sum([len(x.nodes) - 1 for x in data]), "words")

        self.data = [{"graphidx": i} for i, d in enumerate(data)]
        self.modules = modules
        self.parser = parser
        self.update()

        self.full = False
        self.labeled = True
        self.targeted = False
        self.target_lst = list()

        if active:
            self.unlabeled_lst = list(range(len(self.data)))
            self.labeled_lst = list()
        else:
            self.labeled_lst = list(range(len(self.data)))
            self.unlabeled_lst = list()

    def update(self):
        for m in self.parser._model:
            for d, d_ in zip(self.data, self.graphs):
                d.update(m.load_data(self.parser, d_))

    def label(self, lst):
        self.unlabeled_lst = list(sorted(set(self.unlabeled_lst) - set(lst)))
        self.labeled_lst = list(sorted(set(self.labeled_lst) | set(lst)))

    def partial_label(self, lst, idx):
        self.unlabeled_lst = list(sorted(set(self.unlabeled_lst) - set(lst)))
        self.labeled_lst = list(sorted(set(self.labeled_lst) | set(lst)))
        for sent_id, deps in zip(lst, idx):
            heads = self.graphs[sent_id].heads
            pheads = [-1 for i in range(len(heads))]
            for i in deps:
                pheads[i] = heads[i]
            self.graphs[sent_id].heads = np.array(pheads)

    def unlabeled(self):
        return self.unlabeled_lst

    def __len__(self):
        if self.targeted:
            return len(self.target_lst)
        elif self.full:
            return len(self.data)
        elif self.labeled:
            return len(self.labeled_lst)
        else:
            return len(self.unlabeled_lst)

    def __getitem__(self, idx):
        if self.targeted:
            return self.data[self.target_lst[idx]]
        elif self.full:
            return self.data[idx]
        elif self.labeled:
            return self.data[self.labeled_lst[idx]]
        else:
            return self.data[self.unlabeled_lst[idx]]


class DataCollate:

    def __init__(self, parser):
        self.parser = parser

    def __call__(self, data):
        ret = {}
        batch_size = len(data)
        keywords = set(data[0].keys()) - {"graphidx", "raw"}
        graphidx = [d["graphidx"] for d in data]
        raw = [d["raw"] for d in data]

        word_seq_lengths = torch.LongTensor(list(map(len, raw)))
        max_seq_len = word_seq_lengths.max().item()
        mask = torch.zeros((batch_size, max_seq_len)).byte()
        mask_h = torch.zeros((batch_size, max_seq_len)).byte()
        for idx, seqlen in enumerate(word_seq_lengths):
            seqlen = seqlen.item()
            mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
            mask_h[idx, :seqlen] = torch.Tensor([1]*seqlen)

        mask[:, 0] = 0

        for keyword in keywords:
            label_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
            labels = [d[keyword] for d in data]
            for idx, (label, seqlen) in enumerate(zip(labels, word_seq_lengths)):
                label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
            ret[keyword] = label_seq_tensor

        ret.update({
            "graphidx": graphidx,
            "raw": raw,
            "mask": mask,
            "mask_h": mask_h,
        })

        return ret


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
