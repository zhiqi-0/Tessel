LOG_DIR=logs
HOSTNAME=node-0

RUN_NGPUS=(4 8 16 32)

GPU=V100

echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=.:$PYTHONPATH
export OMP_NUM_THREADS=4
export NCCL_DEBUG=0

LOGS=${LOG_DIR}/gpt-sync
mkdir -p $LOGS

LOGS=${LOG_DIR}/mt5-sync
mkdir -p $LOGS

GBS=128 # global batch size

set -ex

# ============================= 4 GPU ============================

if [[ ${RUN_NGPUS[*]} == 4 ]]; then

NGPUS=4
TOTAL_GPUS=4

# GPT - mshape
LOGS=${LOG_DIR}/gpt-sync
LAYERS=32
HEADS=32
HIDDEN=4096
VOCAB_K=1024
VOCAB=`expr ${VOCAB_K} \* 1000`

PREMISE=mshape

ASYNC_COMM=0 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log


# mT5 - nnshape-eager
LOGS=${LOG_DIR}/mt5-sync
LAYERS=24
HEADS=16
HIDDEN=1024
VOCAB_K=1024
VOCAB=`expr ${VOCAB_K} \* 1000`

PREMISE=nnshape_eager

ASYNC_COMM=0 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS \
    examples/mt5/train.py \
        --fp16 --mbs 4 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

fi

# ==================================== 8 GPU ================================
if [[ ${RUN_NGPUS[*]} == 8 ]]; then

NGPUS=8
TOTAL_GPUS=8

# GPT - mshape
LOGS=${LOG_DIR}/gpt-sync
LAYERS=40
HEADS=48
HIDDEN=6144
VOCAB_K=1024
VOCAB=`expr ${VOCAB_K} \* 1000`

PREMISE=mshape

ASYNC_COMM=0 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log


# mT5 - nnshape-eager
LOGS=${LOG_DIR}/mt5-sync
LAYERS=24
HEADS=24
HIDDEN=3072
VOCAB_K=1024
VOCAB=`expr ${VOCAB_K} \* 1000`

PREMISE=nnshape_eager

ASYNC_COMM=0 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS \
    examples/mt5/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.log

fi

# ==================================== 16 GPU ================================
if [[ ${RUN_NGPUS[*]} == 16 ]]; then

NGPUS=8
NNODES=2
TOTAL_GPUS=16

# GPT - mshape
LOGS=${LOG_DIR}/gpt-sync
LAYERS=48
HEADS=64
HIDDEN=8192
VOCAB_K=1024
VOCAB=`expr ${VOCAB_K} \* 1000`

PREMISE=mshape

ASYNC_COMM=0 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.rank${NODE_RANK}.log


# mT5 - nnshape-eager
LOGS=${LOG_DIR}/mt5-sync
LAYERS=32
HEADS=48
HIDDEN=6144
VOCAB_K=1536
VOCAB=`expr ${VOCAB_K} \* 1000`

PREMISE=nnshape_eager

ASYNC_COMM=0 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/mt5/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.rank${NODE_RANK}.log

fi

# ==================================== 32 GPU ================================
if [[ ${RUN_NGPUS[*]} == 32 ]]; then

NGPUS=8
NNODES=4
TOTAL_GPUS=32

# GPT - mshape
LOGS=${LOG_DIR}/gpt-sync
LAYERS=80
HEADS=64
HIDDEN=8192
VOCAB_K=1536
VOCAB=`expr ${VOCAB_K} \* 1000`

PREMISE=mshape

ASYNC_COMM=0 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json --load-tsched gpt.mshape.tsched.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.rank${NODE_RANK}.log


# mT5 - nnshape-eager
LOGS=${LOG_DIR}/mt5-sync
LAYERS=40
HEADS=64
HIDDEN=8192
VOCAB_K=1536
VOCAB=`expr ${VOCAB_K} \* 1000`

PREMISE=nnshape_eager

ASYNC_COMM=0 DISABLE_INTER_RVD=1 \
torchrun --nproc_per_node=$NGPUS --nnodes=$NNODES \
    --node_rank=$NODE_RANK --master_addr=$HOSTNAME \
    examples/mt5/train.py \
        --fp16 --mbs 1 --gbs $GBS --premise $PREMISE --recompute \
        --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json --load-tsched mt5.nnshape.eager.tsched.4stages.json \
    2>&1 | tee ${LOGS}/${TOTAL_GPUS}gpus.$PREMISE.vocab${VOCAB_K}k.layer${LAYERS}.hidden${HIDDEN}.heads${HEADS}.rank${NODE_RANK}.log

fi