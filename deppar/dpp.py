# -*- coding: utf-8 -*-

'''
Pytorch implementation of DPP and associated algorithms from:
Alex Kulesza. Learning with Determinantal Point Processes. PhD Thesis. 2012.
'''

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


class DeterminantalPointProcess:

    def __init__(self, quality, diversity=None):
        self._q = quality
        self._phi = diversity
        self._B = None

    def init_examples(self):
        # Section 3.3, dual representation of DPP, $B^T B$ instead of $B B^T$

        q = scipy.sparse.diags(self._q)
        self._B = q.dot(self._phi).T

    def unnorm_prob(self, A):
        # unnormalized probability of a subset of examples, indexed by A

        if len(A) == 0:
            return 1.

        B = self._B
        B_A = B[:, A]
        L_A = B_A.T.dot(B_A)
        L_A = L_A + np.eye(len(L_A)) * 1e-5
        ret = scipy.linalg.det(L_A)
        return ret

    def greedy_map_incremental_batch(self):
        Yhat = set()
        Yhat_idx = []
        p_unnorm = 1.

        n = len(Yhat)

        while True:
            new_ps = []
            if n == 0:
                for i in range(self._B.shape[1]):
                    if i in Yhat:
                        new_ps.append(-1.)
                    else:
                        new_ps.append( self.unnorm_prob(Yhat_idx + [i]) )
            else:
                partial_B = self._B[:, Yhat_idx]
                partial_B21 = self._B.T.dot(partial_B)
                partial_B12 = partial_B.T.dot(self._B)
                partial_B22 = np.sum(np.multiply(self._B, self._B), axis=0).T

                partial_L = partial_B.T.dot(partial_B)
                partial_L_inv = scipy.linalg.inv(partial_L + np.eye(partial_L.shape[0]) * 1e-10)
                partial_B21_L11_inv = partial_B21.dot(partial_L_inv)
                partial_L_det = p_unnorm

                dets = partial_L_det * (partial_B22 - np.sum(partial_B21_L11_inv * partial_B12.T, axis=1))

                for i in Yhat:
                    dets[i] = -1.

                new_ps = dets

            best_idx = np.argmax(new_ps)
            Yhat_idx.append(best_idx)
            Yhat.add(best_idx)
            p_unnorm = new_ps[best_idx]

            n += 1
            yield best_idx


def dpp_greedy_map(quality, diversity, bsize):
    dpp = DeterminantalPointProcess(quality, diversity=diversity)
    dpp.init_examples()

    ret = []
    for s in dpp.greedy_map_incremental_batch():
        ret.append(s)
        if len(ret) >= bsize:
            break

    return ret


def dpp_greedy_map_tokens(quality, diversity, bsize, lengths):
    dpp = DeterminantalPointProcess(quality, diversity=diversity)
    dpp.init_examples()

    ret = []
    cum = 0
    cur = 0
    for s in dpp.greedy_map_incremental_batch():
        length = lengths[s]

        if length + cum <= bsize:
            ret.append(s)
            cur += 1
            cum += length
        else:
            break

    return ret
