
set -ex

GPU=V100
PREMISE=piper


#=============== micro-bench test =============#

# 8 layer
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --layers 8 --hidden 4096 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json \
    >> gpt.$PREMISE.hidden4096.heads32.log

# 9 layer
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --layers 9 --hidden 4096 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json \
    >> gpt.$PREMISE.hidden4096.heads32.log

# 10 layer
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --layers 10 --hidden 4096 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json \
    >> gpt.$PREMISE.hidden4096_heads32.log

# 11 layer
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --layers 11 --hidden 4096 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json \
    >> gpt.$PREMISE.hidden4096.heads32.log

# 12 layer
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --layers 12 --hidden 4096 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json \
    >> gpt.$PREMISE.hidden4096.heads32.log


# 13 layer
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --layers 13 --hidden 4096 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json \
    >> gpt.$PREMISE.hidden4096.heads32.log

# 14 layer
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --layers 14 --hidden 4096 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json \
    >> gpt.$PREMISE.hidden4096.heads32.log

# 15 layer
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --layers 15 --hidden 4096 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json \
    >> gpt.$PREMISE.hidden4096.heads32.log


# 16 layer
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise $PREMISE --recompute \
        --layers 16 --hidden 4096 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json \
    >> gpt.$PREMISE.hidden4096.heads32.log

#===============================================#




PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 1 --premise piper --recompute \
        --layers 24 --hidden 2560 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json


######### not work ###############

PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 16 --premise piper --recompute \
        --layers 32 --hidden 2560 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json


PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 16 --premise mshape --recompute \
        --layers 32 --hidden 2560 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json

###################################


PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise piper --recompute \
        --layers 8 --hidden 4096 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json \
    >> gpt_hidden4096_heads32.log


PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/gpt/train.py \
        --fp16 --mbs 1 --gbs 64 --premise mshape --recompute \
        --layers 8 --hidden 4096 --heads 32 --seqlen 2048 --vocab 512000 \
        --db-cache gpt_${GPU}_db.json \
    > gpt_hidden4096_heads32.log