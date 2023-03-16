

set -ex

mkdir -p runtime_figures

# single device test
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 --nnodes=1 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 1 --dp 1 --premise vshape \
        --resolution 640 --window-size 40 \
        --layers 18 --hidden 192 --heads 8

# ========================== hidden 192 ================================

# swin-? (except heads 6->8): 640x640:40
VISUALIZE_PLAN=1 \
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 --nnodes=1 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 6 --dp 1 --premise piper \
        --resolution 640 --window-size 40 \
        --layers 18 --hidden 192 --heads 8 --save runtime_figures \
    > runtime_figures/swin.vshape.4dev.log


# ========================== hidden 384 ================================

# Swin-Large (except heads 6->8): 640x640:40
VISUALIZE_PLAN=1 \
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 --nnodes=1 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 6 --dp 1 --premise piper \
        --resolution 640 --window-size 40 \
        --layers 18 --hidden 384 --heads 16 --save runtime_figures \
    > runtime_figures/swin.vshape.4dev.log

VISUALIZE_PLAN=1 \
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 --nnodes=1 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 6 --dp 1 --premise piper \
        --resolution 1536 --window-size 48 \
        --layers 18 --hidden 384 --heads 16 --save runtime_figures \
    > runtime_figures/swin.vshape.4dev.log


# ========================== hidden 512 ================================

# swin-? (except heads 6->8): 640x640:40
VISUALIZE_PLAN=1 \
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 --nnodes=1 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 6 --dp 1 --premise piper \
        --resolution 640 --window-size 40 \
        --layers 18 --hidden 512 --heads 16 --save runtime_figures \
    > runtime_figures/swin.mshape.4dev.log
