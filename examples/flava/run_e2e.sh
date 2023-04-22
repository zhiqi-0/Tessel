#!/bin/bash

GPU=V100
LOGS=logs/flava
mkdir -p $LOGS

set -ex

export NCCL_DEBUG=0
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4
export ASYNC_COMM=0

LAYERS=12
HIDDEN=2048
HEADS=32

# PREMISE=yshape
# PREMISE=1f1b
PREMISE=tp

NGPUS=2
NNODES=1
NODE_RANK=0

# HOSTNAME=worker-0
HOSTNAME=GCRSANDBOX109


TIME=`date "+%m-%d-%H-%M"`
TOTAL_GPUS=`expr ${NGPUS} \* ${NNODES}`

torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --master_addr=$HOSTNAME --node_rank=$NODE_RANK \
    examples/flava/infer.py \
        --fp16 --mbs 2 --gbs 32 --premise $PREMISE \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS \
        --db-cache flava_${GPU}_db.json --load-tsched flava.yshape.tsched.4stages.json \
    2>&1 | tee -a ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.${TIME}.log
