# -*- coding: utf-8 -*-

from .graph import Word, Sentence


ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)


def read_conll(filename, chinese=False):
    def get_instance(graph, graphs):
        words = [Word(row[FORM]) for row in graph]
        heads = [int(row[HEAD]) for row in graph]
        rels = [row[DEPREL] for row in graph]

        sentence = Sentence(words, heads, rels)
        graphs.append(sentence)

    file = open(filename, "rb")

    graphs = []
    graph = []

    sent_count = 0
    for line in file.readlines():
        line = line.decode('utf-8').strip()

        if len(line):
            if line[0] == "#":
                continue
            if "-" in line.split()[0] or "." in line.split()[0]:
                continue
            graph.append(line.split("\t"))
        else:
            sent_count += 1
            get_instance(graph, graphs)
            graph = []

    if len(graph):
        get_instance(graph, graphs)

    print("Read", sent_count, "sents")

    file.close()

    return graphs
