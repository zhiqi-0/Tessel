#!/bin/bash

echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

LOGS=logs/mt5
mkdir -p $LOGS

GPU=V100

# model arch
LAYERS=40
HIDDEN=8192
HEADS=64
VOCAB_K=1536

NGPUS=8
NNODES=4
HOSTNAME=worker-0
# HOSTNAME=GCRSANDBOX109
# NODE_RANK=0

VOCAB=`expr ${VOCAB_K} \* 1000`

set -ex

# PREMISE=tp
PREMISE=1f1b
# PREMISE=gpipe
# PREMISE=chimera
# PREMISE=nnshape_eager

if [ $PREMISE == "nnshape_eager" ] || [ $PREMISE == 'nnshape' ]; then
    echo "enabling async communication"
    export DISABLE_INTER_RVD=1
    export ASYNC_COMM=1
    # export LOG_SCHEDULE=1
fi

if [ $PREMISE == "gpipe" ] || [ $PREMISE == '1f1b' ] || [ $PREMISE == 'chimera' ]; then
    echo "setting param limit"
    export PARAM_LIMIT=28
fi

TOTAL_GPUS=`expr ${NGPUS} \* ${NNODES}`
TIME=`date "+%m-%d-%H-%M"`


if [ "$NNODES" -gt "1" ]; then
    torchrun --nproc_per_node=$NGPUS \
        --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
        examples/mt5/train.py \
            --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
            --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
            --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
        2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log
else
    torchrun --nproc_per_node=$NGPUS \
        examples/mt5/train.py \
            --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
            --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
            --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
        2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log
fi