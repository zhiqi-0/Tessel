
set -ex

GPU=V100

# hidden 192 | heads 8 | image-window 640-40
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 1 --dp 1 --premise piper \
        --resolution 640 --window-size 40 \
        --layers 2 --hidden 192 --heads 8 \
        --db-cache swin_${GPU}_db.json

# hidden 192 | heads 8 | image-window 1536-48
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 1 --dp 1 --premise piper \
        --resolution 1536 --window-size 48 \
        --layers 2 --hidden 192 --heads 8 \
        --db-cache swin_${GPU}_db.json


# hidden 384 | heads 16 | image-window 640-40
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 1 --dp 1 --premise piper \
        --resolution 640 --window-size 40 \
        --layers 2 --hidden 384 --heads 16 \
        --db-cache swin_${GPU}_db.json


# hidden 384 | heads 16 | image-window 1536-48
# profile needs to add special rule for window_attn partition (num=4)
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 1 --dp 1 --premise piper \
        --resolution 1536 --window-size 48 \
        --layers 2 --hidden 384 --heads 16 \
        --db-cache swin_${GPU}_db.json


# hidden 512 | heads 16 | image-window 640-40
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 1 --dp 1 --premise piper \
        --resolution 640 --window-size 40 \
        --layers 2 --hidden 512 --heads 16 \
        --db-cache swin_${GPU}_db.json


# hidden 512 | heads 16 | image-window 1536-48
# profile needs to add special rule for window_attn partition (num=4)
PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/swin/train.py \
        --fp16 --mbs 1 --gbs 1 --dp 1 --premise piper \
        --resolution 1536 --window-size 48 \
        --layers 2 --hidden 512 --heads 16 \
        --db-cache swin_${GPU}_db.json
