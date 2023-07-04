
set -ex

GPU=V100
VOCAB=1024000

# default vocab size: 51.2K
# if name == '350M':
#     embed_dim, layers, attention_heads = 1024, 24, 16
# elif name == '760M':
#     embed_dim, layers, attention_heads = 1536, 24, 16
# elif name == '1.3B':
#     embed_dim, layers, attention_heads = 2048, 24, 32
# elif name == '2.6B':
#     embed_dim, layers, attention_heads = 2560, 32, 32
# elif name == '6.7B':
#     embed_dim, layers, attention_heads = 4096, 32, 32
# elif name == '15B':
#     embed_dim, layers, attention_heads = 5120, 48, 40
# elif name == '39B':
#     embed_dim, layers, attention_heads = 8192, 48, 64
# elif name == '175B':
#     embed_dim, layers, attention_heads = 12288, 96, 96


PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/mt5/train.py \
        --fp16 --mbs 1 --gbs 1 --premise 1f1b --recompute \
        --layers 2 --hidden 2560 --heads 32 --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json


PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/mt5/train.py \
        --fp16 --mbs 1 --gbs 1 --premise 1f1b --recompute \
        --layers 2 --hidden 4096 --heads 32 --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json


PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/mt5/train.py \
        --fp16 --mbs 1 --gbs 1 --premise 1f1b --recompute \
        --layers 2 --hidden 5120 --heads 32 --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json


PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/mt5/train.py \
        --fp16 --mbs 1 --gbs 1 --premise 1f1b --recompute \
        --layers 2 --hidden 6144 --heads 48 --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json


PYTHONPATH=.:$PYTHONPATH OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    examples/mt5/train.py \
        --fp16 --mbs 1 --gbs 1 --premise 1f1b --recompute \
        --layers 2 --hidden 8192 --heads 64 --seqlen 1024 --vocab $VOCAB \
        --db-cache mt5_${GPU}_db.json