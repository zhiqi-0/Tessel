
set -ex

GPU=V100

# PREMISE=piper
PREMISE=mshape

NGPUS=4
VOCAB=512000
# VOCAB=768000
# VOCAB=1024000


#=============== micro-bench test =============#


# PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=$NGPUS \
#     examples/gpt/train.py \
#         --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
#         --layers 24 --hidden 2048 --heads 32 --seqlen 2048 --vocab $VOCAB \
#         --db-cache gpt_${GPU}_db.json \
#     >> gpt.$PREMISE.vocab$VOCAB.log
# 
# 
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=$NGPUS \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --layers 32 --hidden 2560 --heads 32 --seqlen 2048 --vocab $VOCAB \
        --db-cache gpt_${GPU}_db.json \
    >> gpt.$PREMISE.vocab$VOCAB.log
# 
# 
# PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=$NGPUS \
#     examples/gpt/train.py \
#         --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
#         --layers 32 --hidden 4096 --heads 32 --seqlen 2048 --vocab $VOCAB \
#         --db-cache gpt_${GPU}_db.json \
#     >> gpt.$PREMISE.vocab$VOCAB.log
# 
# 
# PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=$NGPUS \
#     examples/gpt/train.py \
#         --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
#         --layers 48 --hidden 5120 --heads 40 --seqlen 2048 --vocab $VOCAB \
#         --db-cache gpt_${GPU}_db.json \
#     >> gpt.$PREMISE.vocab$VOCAB.log
# 
# 
# PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=$NGPUS \
#     examples/gpt/train.py \
#         --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
#         --layers 48 --hidden 6144 --heads 48 --seqlen 2048 --vocab $VOCAB \
#         --db-cache gpt_${GPU}_db.json \
#     >> gpt.$PREMISE.vocab$VOCAB.log
# 
# 
# PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
#     --nproc_per_node=$NGPUS \
#     examples/gpt/train.py \
#         --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
#         --layers 48 --hidden 5120 --heads 64 --seqlen 2048 --vocab $VOCAB \
#         --db-cache gpt_${GPU}_db.json \
#     >> gpt.$PREMISE.vocab$VOCAB.log
# 