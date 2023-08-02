#!/bin/bash

LOG_DIR=logs

GPU=V100
NGPUS=8
NNODES=4
TOTAL_GPUS=32
HOSTNAME=node-0

echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

GBS=128 # global batch size


# ================================= GPT =============================
LOGS=${LOG_DIR}/gpt
mkdir -p $LOGS

LAYERS=80
HEADS=64
HIDDEN=8192
VOCAB_K=1536
VOCAB=`expr ${VOCAB_K} \* 1000`

set -ex


# GPT - 1f1b
PREMISE=1f1b

PARAM_LIMIT=21 \
torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute --max-pp 4 \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.rank${NODE_RANK}.log

# GPT - 1f1b plus
PREMISE=1f1b+

torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.rank${NODE_RANK}.log

# GPT - mshape
PREMISE=mshape

ASYNC_COMM=1 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.rank${NODE_RANK}.log

# GPT - Chimera
# OOM


# ================================= mT5 =============================
LOGS=${LOG_DIR}/mt5
mkdir -p $LOGS

LAYERS=40
HEADS=64
HIDDEN=8192
VOCAB_K=1536
VOCAB=`expr ${VOCAB_K} \* 1000`


# mT5 - 1f1b
PREMISE=1f1b

PARAM_LIMIT=28 \
torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/mt5/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute --max-pp 2 \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.rank${NODE_RANK}.log


# mT5 - 1f1b-plus
PREMISE=1f1b+

torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/mt5/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.rank${NODE_RANK}.log

# mT5 - chimera (OOM)

# mT5 - nnshape-eager
PREMISE=nnshape_eager

ASYNC_COMM=1 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/mt5/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.rank${NODE_RANK}.log
