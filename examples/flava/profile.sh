#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

GPU=V100

LAYERS=12
HIDDEN=768
HEADS=12

LAYERS=12
HIDDEN=1024
HEADS=16

export ASYNC_COMM=1

torchrun --nproc_per_node=4 \
    examples/flava/infer.py \
        --fp16 --mbs 1 --gbs 8 --premise 1f1b \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS \
        --db-cache flava_${GPU}_db.json --load-tsched flava.yshape.tsched.4stages.json
