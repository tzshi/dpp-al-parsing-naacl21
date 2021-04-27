# -*- coding: utf-8 -*-

import heapq

import numpy as np


def find_best_heap(scores, req=None, banned=None):
    scores = np.copy(scores)
    if req:
        for src, tgt in req:
            for i in range(len(scores)):
                if i != src:
                    scores[tgt, i] = -np.inf
    if banned:
        for src, tgt in banned:
            scores[tgt, src] = -np.inf
    for i in range(len(scores)):
        scores[i, i] = -np.inf
        scores[0, i] = -np.inf
    nodes = [i for i in range(1, len(scores))]
    new_i = len(scores)
    edges = {}
    for i in range(len(scores)):
        ret = []
        for j in range(len(scores)):
            if scores[i, j] != -np.inf:
                heapq.heappush(ret, (-scores[i, j], (j, i)))
        edges[i] = ret

    B = {}
    B_score = {}
    C = {}
    beta = {}
    node_mappings = {}

    while len(nodes) > 0:
        i = nodes.pop()
        b_score, (b, b_t) = heapq.heappop(edges[i])
        B[i] = node_mappings.get(b, b)
        B_score[i] = -b_score
        beta[i] = (b, b_t)

        # detect cycle
        has_cycle = False
        cycle_nodes = {i}
        cur = i
        while cur in B:
            cur = B[cur]
            cycle_nodes.add(cur)
            if cur == i:
                has_cycle = True
                break

        # collapse cycle
        if has_cycle:
            for j in cycle_nodes:
                C[j] = new_i
            full_nodes = set(edges.keys()) | {0}
            rest_nodes = full_nodes - cycle_nodes
            edges[new_i] = []

            for j in cycle_nodes:
                for e_score, (e_s, e_t) in edges[j]:
                    e_score = -e_score
                    if node_mappings.get(e_s, e_s) not in cycle_nodes:
                        heapq.heappush(edges[new_i], (-e_score + B_score[node_mappings.get(e_t, e_t)], (e_s, e_t)))

            for j in cycle_nodes:
                del edges[j]
            for j in cycle_nodes:
                del B[j]
                del B_score[j]

            for j in B:
                if B[j] in cycle_nodes:
                    B[j] = new_i

            for j in node_mappings:
                if node_mappings[j] in cycle_nodes:
                    node_mappings[j] = new_i
            for j in cycle_nodes:
                node_mappings[j] = new_i

            nodes.append(new_i)
            new_i += 1

    skip_set = set()
    for i in range(new_i - 1, len(scores) - 1, -1):
        if i in skip_set:
            continue

        src_node, tgt_node = beta[i]
        beta[tgt_node] = beta[i]

        while tgt_node != i:
            tgt_node = C[tgt_node]
            skip_set.add(tgt_node)

    heads = [-1] * len(scores)
    for i in range(1, len(scores)):
        heads[i] = beta[i][0]

    return heads
