#!/bin/bash

EVAL_STEPS=800
DECAY_EVALS=1
DECAY_TIMES=1
DECAY_RATIO=0.1

BATCH_SIZE=16

LEARNING_RATE=1e-5
BETA1=0.9
BETA2=0.999
EPSILON=1e-8
WEIGHT_DECAY=0.
WARMUP=640

CLIP=5.0

GPU=True

PROJ_DIMS=1024

HSEL_DIMS=500
HSEL_DROPOUT=0.33

REL_DIMS=100
REL_DROPOUT=0.33

QUALITY=amp
# QUALITY=random
# QUALITY=bald

# ADJUST_METHOD=id
ADJUST_METHOD=noop

DIVERSITY=subtree
# DIVERSITY=avg

COMBO_METHOD=dpp
# COMBO_METHOD=topk

SAMPLE_SIZE=500

DUPLICATE=1

TRAIN_FILE=en_ewt-ud-train.conllu
DEV_FILE=en_ewt-ud-dev.conllu
TEST_FILE=en_ewt-ud-test.conllu

LOG_FOLDER=./models/

mkdir -p $LOG_FOLDER

RUN=testrun

SAVE_PREFIX=${LOG_FOLDER}/${RUN}

mkdir -p $SAVE_PREFIX

TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=3

python3 -m deppar.parser \
    - build-vocab $TRAIN_FILE \
    - create-parser --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE --beta1 $BETA1 --beta2 $BETA2 --epsilon $EPSILON --clip $CLIP \
        --proj-dims $PROJ_DIMS \
        --hsel-dims $HSEL_DIMS --hsel-dropout $HSEL_DROPOUT \
        --rel-dims $REL_DIMS --rel-dropout $REL_DROPOUT \
        --weight-decay $WEIGHT_DECAY \
        --warmup $WARMUP \
        --gpu $GPU \
        --local true \
    - activeTrain $TRAIN_FILE --dev $DEV_FILE --test $TEST_FILE --sample-size $SAMPLE_SIZE \
        --quality $QUALITY --diversity $DIVERSITY --combo-method $COMBO_METHOD --adjust-method $ADJUST_METHOD \
        --duplicate $DUPLICATE \
        --eval-steps $EVAL_STEPS --decay-evals $DECAY_EVALS --decay-times $DECAY_TIMES --decay-ratio $DECAY_RATIO \
        --save-prefix $SAVE_PREFIX/ \
    - finish
