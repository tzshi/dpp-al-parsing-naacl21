#!/usr/bin/env bash

# Sample script to convert the Revised English News Text Treebank:
#
# https://catalog.ldc.upenn.edu/LDC2015T13
#
# to UD, and partition as in the experiments described in
# "Diversity-Aware Batch Active Learning for Dependency Parsing".
# Assumes you have Stanford CoreNLP (run with version 4.0.0) installed,
# and CORENLP_PATH set appropriately.

TREEBANK_DIR=${1}  # Directory where the LDC corpus is extracted.

rm -f train.mrg
rm -f dev.mrg
rm -f test.mrg
for SECTION in {02..21}; do
    cat ${TREEBANK_DIR}/${SECTION}/*.mrg >> train.mrg
done
cat $TREEBANK_DIR/22/*.mrg >> dev.mrg
cat $TREEBANK_DIR/23/*.mrg >> test.mrg
for FOLD in train dev test: do
    java -mx8g -cp "${CORENLP_PATH}/*" edu.stanford.nlp.trees.ud.UniversalDependenciesConverter -treeFile ${FOLD}.mrg > ${FOLD}.conllu
done
