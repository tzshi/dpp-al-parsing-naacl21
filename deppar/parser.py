# -*- coding: utf-8 -*-

import json
import fire
import sys
import time
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .modules import HSelParser, RelLabeler
from .features import WordSequence
from .io import read_conll
from .utils import buildVocab
from .data import DataProcessor, DataCollate, InfiniteDataLoader
from .adamw import AdamW
from .acquisition import PartialAcquisitionProcedure


class DepParser:

    def __init__(self, **kwargs):
        pass

    def create_parser(self, **kwargs):
        self._verbose = kwargs.get("verbose", True)
        if self._verbose:
            print("Parameters (others default):")
            for k in sorted(kwargs):
                print(k, kwargs[k])
            sys.stdout.flush()

        self._args = kwargs

        self._gpu = kwargs.get("gpu", True)

        self._learning_rate = kwargs.get("learning_rate", 0.001)
        self._beta1 = kwargs.get("beta1", 0.9)
        self._beta2 = kwargs.get("beta2", 0.999)
        self._epsilon = kwargs.get("epsilon", 1e-8)
        self._weight_decay = kwargs.get("weight_decay", 0.)
        self._warmup = kwargs.get("warmup", 1)

        self._clip = kwargs.get("clip", 5.)

        self._batch_size = kwargs.get("batch_size", 16)
        self._min_batch_size = kwargs.get("min_batch_size", 16)
        self._eval_batch_size = kwargs.get("eval_batch_size", 32)

        self._proj_dims = kwargs.get("proj_dims", 256)

        self._hsel_dims = kwargs.get("hsel_dims", 200)
        self._hsel_dropout = kwargs.get("hsel_dropout", 0.0)

        self._rel_dims = kwargs.get("rel_dims", 50)
        self._rel_dropout = kwargs.get("rel_dropout", 0.0)

        self._bert = kwargs.get("bert", False)

        self.init_model()
        return self

    def _load_vocab(self, vocab):
        self._fullvocab = vocab
        self._irels = ["unk"] + vocab["rels"]
        self._rels = {w: i for i, w in enumerate(self._irels)}

    def load_vocab(self, filename):
        with open(filename, "rb") as f:
            vocab = json.load(f)
        self._load_vocab(vocab)
        return self

    def save_vocab(self, filename):
        with open(filename, "wb") as f:
            f.write(json.dumps(self._fullvocab).encode('utf-8'))
        return self

    def build_vocab(self, filename):
        sents = read_conll(filename)
        self._fullvocab = buildVocab(sents)
        self._load_vocab(self._fullvocab)

        return self

    def update_vocab(self, graphs):
        print("Updating vocab from", len(graphs), "sentences")
        self._fullvocab = buildVocab(graphs)
        self._load_vocab(self._fullvocab)

        return self

    def save_model(self, filename):
        print("Saving model to", filename)
        self.save_vocab(filename + ".vocab")
        with open(filename + ".params", "wb") as f:
            f.write(json.dumps(self._args).encode('utf-8'))
        with open(filename + ".model", "wb") as f:
            torch.save(self._model.state_dict(), f)

    def load_model(self, filename, **kwargs):
        print("Loading model from", filename)
        self.load_vocab(filename + ".vocab")
        with open(filename + ".params", "rb") as f:
            args = json.load(f)
            args.update(kwargs)
            self.create_parser(**args)

        with open(filename + ".model", "rb") as f:
            if kwargs.get('gpu', False):
                self._model.load_state_dict(torch.load(f))
            else:
                self._model.load_state_dict(torch.load(f, map_location="cpu"))
        return self

    def init_model(self):
        self._seqrep = WordSequence(self)

        self._hsel_parser = HSelParser(self, self._hsel_dims, self._hsel_dropout)
        self._rel_labeler = RelLabeler(self, self._rel_dims, self._hsel_dropout)
        self._modules = [self._hsel_parser, self._rel_labeler]

        modules = [self._seqrep] + self._modules

        self._model = nn.ModuleList(modules)

        if self._gpu:
            self._model.cuda()

        return self

    def _trainingLoop(self,
                      train_loader,
                      eval_steps=100,
                      decay_evals=5,
                      decay_times=0,
                      decay_ratio=0.5,
                      dev_graphs=None,
                      test_graphs=None,
                      **kwargs
                      ):
        best_params = None
        optimizer = AdamW(
            self._model.parameters(), lr=self._learning_rate,
            betas=(self._beta1, self._beta2),
            eps=self._epsilon, weight_decay=self._weight_decay, warmup=self._warmup
        )
        scheduler = ReduceLROnPlateau(
            optimizer, 'max', factor=decay_ratio, patience=decay_evals, verbose=True, cooldown=1
        )

        t0 = time.time()
        results, eloss = defaultdict(float), 0.
        max_dev = 0.
        max_dev_dict = None
        max_test_dict = None

        for batch_i, batch in enumerate(train_loader):
            if self._gpu:
                for k in batch:
                    if k != "graphidx" and k != "raw":
                        batch[k] = batch[k].cuda()

            mask = batch["mask"]

            self._model.train()
            self._model.zero_grad()

            loss = []

            seq_features = self._seqrep(batch)

            for module in self._modules:
                l, pred = module.calculate_loss(seq_features, batch)
                batch_label = module.batch_label(batch)

                if l is not None:
                    loss.append(l)
                    module.evaluate(results, self, None, pred, batch_label, mask, batch, train=True)

            loss = sum(loss)
            eloss += loss.item()
            loss.backward()

            nn.utils.clip_grad_norm_(self._model.parameters(), self._clip)
            optimizer.step()

            if batch_i and batch_i % 100 == 0:
                for module in self._modules:
                    module.metrics(results)
                results["loss/loss"] = eloss
                print(batch_i // 100, "{:.2f}s".format(time.time() - t0), end=" ")
                sys.stdout.flush()
                results, eloss = defaultdict(float), 0.
                t0 = time.time()

            if batch_i and (batch_i % eval_steps == 0):
                results = self.evaluate(dev_graphs)

                performance = results["metrics/HSel-acc"]
                print()
                print("Dev UAS", performance)
                print()

                if performance >= max_dev:
                    max_dev = performance
                    max_dev_dict = results
                    best_params = deepcopy(self._model.state_dict())

                results = defaultdict(float)
                scheduler.step(performance)
                if scheduler.in_cooldown:
                    optimizer.state = defaultdict(dict)
                    if decay_times <= 0:
                        break
                    else:
                        decay_times -= 1

        self._model.load_state_dict(best_params)
        max_test_dict = self.evaluate(test_graphs)
        performance = max_test_dict["metrics/HSel-acc"]
        print()
        print("Test UAS", performance)
        print()

        return max_dev_dict, max_test_dict

    def activeTrain(self,
                    filename,
                    eval_steps=100,
                    decay_evals=5,
                    decay_times=0,
                    decay_ratio=0.5,
                    dev=None,
                    test=None,
                    save_prefix=None,
                    sample_size=1280,
                    init_size=128,
                    duplicate=1,
                    quality="amp",
                    diversity="subtree",
                    adjust_method="noop",
                    combo_method="dpp",
                    **kwargs
                    ):

        trajectory = []
        train_graphs = DataProcessor(filename, self, self._model, duplicate=duplicate, active=True)
        first_batch = [train_graphs.unlabeled_lst[i * (len(train_graphs.unlabeled_lst) // duplicate // init_size)] for i in range(init_size)]
        train_graphs.label(first_batch)

        batch_size = max(self._min_batch_size, min(self._batch_size, len(train_graphs.labeled_lst) // 128))
        train_loader = InfiniteDataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=DataCollate(self))
        if dev is not None:
            dev_graphs = DataProcessor(dev, self, self._model)
        else:
            dev_graphs = None

        if test is not None:
            test_graphs = DataProcessor(test, self, self._model)
        else:
            test_graphs = None

        acquisition_fn = PartialAcquisitionProcedure(quality=quality, diversity=diversity, adjust_method=adjust_method, combo_method=combo_method)
        acquisition_fn.last_picked = first_batch

        epoch = 0
        while True:
            self.update_vocab([train_graphs.graphs[x] for x in train_graphs.labeled_lst])
            self.init_model()

            train_graphs.update()
            if dev_graphs:
                dev_graphs.update()
            if test_graphs:
                test_graphs.update()

            train_loader = InfiniteDataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=DataCollate(self))

            cur_eval_steps = min(eval_steps, len(train_graphs.labeled_lst) * 4 // batch_size)

            ret_dev, ret_test = self._trainingLoop(
                train_loader, eval_steps=cur_eval_steps, decay_evals=decay_evals, decay_times=decay_times,
                decay_ratio=decay_ratio, dev_graphs=dev_graphs, test_graphs=test_graphs
            )

            labeled_tokens = 0
            for x in train_graphs.labeled_lst:
                for h in train_graphs.graphs[x].heads[1:]:
                    if h >= 0:
                        labeled_tokens += 1

            trajectory.append((len(train_graphs.labeled_lst), labeled_tokens, ret_dev, ret_test, acquisition_fn.last_picked))

            if sum([len(train_graphs.graphs[x].nodes) - 1 for x in train_graphs.unlabeled_lst]) < sample_size:
                break

            if epoch % 5 == 0 and save_prefix:
                self.save_model("{}model_{}".format(save_prefix, epoch))

            acquisition_fn.step(train_graphs, self, sample_size)

            batch_size = max(self._min_batch_size, min(self._batch_size, len(train_graphs.labeled_lst) // 128))

            print()
            print("Picked sentences' ids:", acquisition_fn.last_picked)
            print("New training batch size", batch_size)
            print()

            epoch += 1

        if save_prefix:
            with open(filename + ".traj", "wb") as f:
                f.write(json.dumps(trajectory).encode('utf-8'))
            self.save_model("{}model".format(save_prefix))

        return self


    def evaluate(self, data):
        results = defaultdict(float)
        pred_results = []
        gold_results = []
        self._model.eval()
        batch_size = self._batch_size
        start_time = time.time()
        train_num = len(data)

        dev_loader = DataLoader(data, batch_size=self._eval_batch_size, shuffle=False, num_workers=1, collate_fn=DataCollate(self))
        dev_loader = tqdm(dev_loader)

        with torch.no_grad():
            for batch in dev_loader:
                graphs = [data.graphs[idx] for idx in batch["graphidx"]]
                if self._gpu:
                    for k in batch:
                        if k != "graphidx" and k != "raw":
                            batch[k] = batch[k].cuda()

                mask = batch["mask"]

                seq_features = self._seqrep(batch)

                for module in self._modules:
                    batch_label = module.batch_label(batch)
                    pred = module(self, seq_features, batch)
                    module.evaluate(results, self, graphs, pred, batch_label, mask, batch, train=False)

                if "pred_head" in batch and "pred_rel" in batch:
                    for idx, h, r in zip(batch["graphidx"], batch["pred_head"].cpu().data.numpy(), batch["pred_rel"].cpu().data.numpy()):
                        g = data.graphs[idx]
                        for i in range(1, len(g.nodes)):
                            g.pred_heads[i] = h[i] - 1
                            g.pred_rels[i] = self._irels[r[i]]

        decode_time = time.time() - start_time
        results["speed/speed"] = len(data)/decode_time

        for module in self._modules:
            module.metrics(results)

        print(results)
        return results

    def finish(self, **kwargs):
        print()
        sys.stdout.flush()


if __name__ == '__main__':
    fire.Fire(DepParser)
