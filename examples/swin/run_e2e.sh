#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

LOGS=logs/swin
mkdir -p $LOGS

GPU=V100

# model arch
LAYERS=32
HIDDEN=512
HEADS=16

set -ex

# PREMISE=tp
# PREMISE=piper
PREMISE=mshape

if [ $PREMISE == "mshape" ]; then
    echo "enabling async communication"
    export DISABLE_INTER_RVD=1
    export ASYNC_COMM=1
fi

if [ $PREMISE == "piper" ]; then
    echo "setting activation limit"
    export ACT_LIMIT=24
fi

# export MEM_LIMIT=16


# training config
NGPUS=4
NNODES=1
HOSTNAME=node-0
HOSTNAME=GCRSANDBOX109
NODE_RANK=0

TOTAL_GPUS=`expr ${NGPUS} \* ${NNODES}`
TIME=`date "+%m-%d-%H-%M"`

RESOLUTION=1536
WINDOW_SIZE=48

torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --resolution $RESOLUTION --window-size $WINDOW_SIZE \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS \
        --db-cache swin_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee -a ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.res1536.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.${TIME}.log
