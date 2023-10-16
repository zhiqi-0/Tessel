#!/bin/bash

LOG_DIR=logs

GPU=V100
NGPUS=4
TOTAL_GPUS=4

echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4
export NCCL_DEBUG=0

export ASYNC_COMM=0

GBS=128 # global batch size

LOGS=${LOG_DIR}/flava
mkdir -p $LOGS


LAYERS=24
HIDDEN=4096
HEADS=32


PREMISE=(1f1b tetris)
GBS=(1 2 4 8 16 32 64 128)
# PREMISE=(tp)
# GBS=(1)

for p in ${PREMISE[@]}; do
    for gbs in ${GBS[@]}; do
        torchrun --nproc_per_node=$NGPUS \
            examples/flava/infer.py \
            --fp16 --mbs 1 --gbs $gbs --premise $p \
            --layers $LAYERS --hidden $HIDDEN --heads $HEADS \
            --db-cache flava_${GPU}_db.json --load-tsched flava.yshape.tsched.4stages.json \
            2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$p.${gbs}gbs.${LAYERS}L.${HIDDEN}H.${HEADS}h.log
    done
done
