#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/simulator
mkdir -p $LOGS

# VShape
python examples/simulator/cases_tetris.py \
    --premise vshape --ndevs 4 --nmicros 4 --inflight 4 \
2>&1 | tee ${LOGS}/tetris.vshape.train.log

python examples/simulator/cases_tetris.py \
    --premise vshape --ndevs 4 --nmicros 4 --inflight 4 --infer \
2>&1 | tee ${LOGS}/tetris.vshape.infer.log

# XShape
python examples/simulator/cases_tetris.py \
    --premise xshape --ndevs 4 --nmicros 3 --inflight 4 \
2>&1 | tee ${LOGS}/tetris.xshape.train.log

python examples/simulator/cases_tetris.py \
    --premise xshape --ndevs 4 --nmicros 3 --inflight 4 --infer \
2>&1 | tee ${LOGS}/tetris.xshape.infer.log

# NNShape
python examples/simulator/cases_tetris.py \
    --premise nnshape --ndevs 4 --nmicros 6 --inflight 6 \
2>&1 | tee ${LOGS}/tetris.nnshape.train.log

python examples/simulator/cases_tetris.py \
    --premise nnshape --ndevs 4 --nmicros 6 --inflight 4 --infer \
2>&1 | tee ${LOGS}/tetris.nnshape.infer.log

# MShape
python examples/simulator/cases_tetris.py \
    --premise mshape --ndevs 4 --nmicros 6 --inflight 6 \
2>&1 | tee ${LOGS}/tetris.mshape.train.log

python examples/simulator/cases_tetris.py \
    --premise mshape --ndevs 4 --nmicros 6 --inflight 4 --infer \
2>&1 | tee ${LOGS}/tetris.mshape.infer.log

# K-Shape
python examples/simulator/cases_tetris.py \
    --premise kshape --ndevs 4 --nmicros 3 --inflight 6 \
2>&1 | tee ${LOGS}/tetris.kshape.train.log

python examples/simulator/cases_tetris.py \
    --premise kshape --ndevs 4 --nmicros 2 --inflight 4 --infer \
2>&1 | tee ${LOGS}/tetris.kshape.infer.log


# ==================================================

# # VShape
# python examples/simulator/cases_full.py \
#     --premise vshape --ndevs 4 --nmicros 4 --inflight 4 \
# 2>&1 | tee ${LOGS}/naive.vshape.log
# 
# # XShape
# python examples/simulator/cases_full.py \
#     --premise xshape --ndevs 4 --nmicros 3 --inflight 4 \
# 2>&1 | tee ${LOGS}/naive.xshape.log
# 
# # MShape
# python examples/simulator/cases_full.py \
#     --premise mshape --ndevs 4 --nmicros 6 --inflight 6 \
# 2>&1 | tee ${LOGS}/naive.mshape.log
# 
# # >|Shape
# python examples/simulator/cases_full.py \
#     --premise yshape --ndevs 4 --nmicros 2 --inflight 4 \
# 2>&1 | tee ${LOGS}/naive.yshape.log