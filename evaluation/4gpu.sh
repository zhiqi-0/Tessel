#!/bin/bash

GPU=V100
NGPUS=4

echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

GBS=64 # global batch size


# ================================= GPT =============================
LOGS=logs/gpt
mkdir -p $LOGS

LAYERS=32
HEADS=32
HIDDEN=4096
VOCAB_K=1024
VOCAB=`expr ${VOCAB_K} \* 1000`

set -ex

# GPT - 1f1b
PREMISE=1f1b

PARAM_LIMIT=22 \
torchrun --nproc_per_node=$NGPUS \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

# GPT - 1f1b plus
PREMISE=1f1b+

torchrun --nproc_per_node=$NGPUS \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

# GPT - mshape
PREMISE=mshape

ASYNC_COMM=1 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

# GPT - Chimera
# OOM


# ================================= mT5 =============================
LOGS=logs/mt5
mkdir -p $LOGS

LAYERS=24
HEADS=16
HIDDEN=1024
VOCAB_K=1024
VOCAB=`expr ${VOCAB_K} \* 1000`


# mT5 - 1f1b
PREMISE=1f1b

torchrun --nproc_per_node=$NGPUS \
        examples/mt5/train.py \
            --fp16 --mbs 4 --gbs 64 --premise $PREMISE --recompute \
            --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
            --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
        2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

# mT5 - 1f1b-plus
PREMISE=1f1b+

torchrun --nproc_per_node=$NGPUS \
    examples/mt5/train.py \
        --fp16 --mbs 4 --gbs 64 --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

# mT5 - chimera
PREMISE=chimera

torchrun --nproc_per_node=$NGPUS \
    examples/mt5/train.py \
        --fp16 --mbs 16 --gbs 64 --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

# mT5 - nnshape-eager
PREMISE=nnshape_eager

ASYNC_COMM=1 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS \
    examples/mt5/train.py \
        --fp16 --mbs 4 --gbs 64 --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log
