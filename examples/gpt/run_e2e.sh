#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

LOGS=logs/gpt
mkdir -p $LOGS

GPU=V100

# model arch
LAYERS=32
HIDDEN=4096
HEADS=32
VOCAB_K=768  # 512 1024


VOCAB=`expr ${VOCAB_K} \* 1000`

set -ex

# PREMISE=tp
PREMISE=1f1b
PREMISE=gpipe
# PREMISE=mshape

if [ $PREMISE == "mshape" ]; then
    echo "enabling async communication"
    export DISABLE_INTER_RVD=1
    export ASYNC_COMM=1
fi

if [ $PREMISE == "gpipe" ] || [ $PREMISE == '1f1b' ]; then
    echo "setting param limit"
    export PARAM_LIMIT=20
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


torchrun --nproc_per_node=$NGPUS \
    --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee -a ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.${TIME}.log
