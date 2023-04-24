#!/bin/bash

echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4

LOGS=logs/gpt
mkdir -p $LOGS

GPU=V100

# model arch
LAYERS=(24 28 32 36 40 44 48)
HIDDEN=4096
HEADS=32
VOCAB_K=1024  # 512 1024

NGPUS=4
NNODES=1
HOSTNAME=worker-0
# HOSTNAME=GCRSANDBOX109
NODE_RANK=0

VOCAB=`expr ${VOCAB_K} \* 1000`

set -ex

# PREMISE=1f1b
# PREMISE=gpipe
PREMISE=mshape

if [ $PREMISE == "mshape" ]; then
    echo "enabling async communication"
    export DISABLE_INTER_RVD=1
    export ASYNC_COMM=0
fi

if [ $PREMISE == "gpipe" ] || [ $PREMISE == '1f1b' ]; then
    echo "setting param limit"
    export PARAM_LIMIT=22
fi


TOTAL_GPUS=`expr ${NGPUS} \* ${NNODES}`
TIME=`date "+%m-%d-%H-%M"`


for LAYER in ${LAYERS[@]}; do
    echo "Running with ${LAYER} layers"
    torchrun --nproc_per_node=$NGPUS \
        --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
        examples/gpt/train.py \
            --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
            --layers $LAYER --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
            --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
        2>&1 | tee -a ${LOGS}/micro.sync.${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYER}.hidden${HIDDEN}.heads${HEADS}.${TIME}.log
done
