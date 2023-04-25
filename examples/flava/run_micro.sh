#!/bin/bash

GPU=V100
LOGS=logs/flava
mkdir -p $LOGS

set -ex

export NCCL_DEBUG=0
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4
export ASYNC_COMM=0

LAYERS=24
HIDDEN=4096
HEADS=32

ALL_GBS=(1 2 4 8 16 32 64 128)

# PREMISE=yshape
PREMISE=1f1b
# PREMISE=tp

NGPUS=4
NNODES=1
NODE_RANK=0

HOSTNAME=worker-0
HOSTNAME=GCRSANDBOX109


TIME=`date "+%m-%d-%H-%M"`
TOTAL_GPUS=`expr ${NGPUS} \* ${NNODES}`

for GBS in ${ALL_GBS[@]}; do
    torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
        --master_addr=$HOSTNAME --node_rank=$NODE_RANK \
        examples/flava/infer.py \
            --fp16 --mbs 1 --gbs $GBS --premise $PREMISE \
            --layers $LAYERS --hidden $HIDDEN --heads $HEADS \
            --db-cache flava_${GPU}_db.json --load-tsched flava.yshape.tsched.4stages.json \
        2>&1 | tee -a ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.${TIME}.log
done
