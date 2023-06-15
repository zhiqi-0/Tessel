#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/simulator
mkdir -p $LOGS

# VShape
python examples/simulator/cases_tetris.py \
    --premise vshape --ndevs 4 --nmicros 4 --inflight 4 \
2>&1 | tee ${LOGS}/tetris.vshape.log

# XShape
python examples/simulator/cases_tetris.py \
    --premise xshape --ndevs 4 --nmicros 3 --inflight 4 \
2>&1 | tee ${LOGS}/tetris.xshape.log

# MShape
python examples/simulator/cases_tetris.py \
    --premise mshape --ndevs 4 --nmicros 6 --inflight 6 \
2>&1 | tee ${LOGS}/tetris.mshape.log

# >|Shape
python examples/simulator/cases_tetris.py \
    --premise yshape --ndevs 4 --nmicros 2 --inflight 4 \
2>&1 | tee ${LOGS}/tetris.yshape.log


# ==================================================

# VShape
python examples/simulator/cases_full.py \
    --premise vshape --ndevs 4 --nmicros 4 --inflight 4 \
2>&1 | tee ${LOGS}/naive.vshape.log

# XShape
python examples/simulator/cases_full.py \
    --premise xshape --ndevs 4 --nmicros 3 --inflight 4 \
2>&1 | tee ${LOGS}/naive.xshape.log

# MShape
python examples/simulator/cases_full.py \
    --premise mshape --ndevs 4 --nmicros 6 --inflight 6 \
2>&1 | tee ${LOGS}/naive.mshape.log

# >|Shape
python examples/simulator/cases_full.py \
    --premise yshape --ndevs 4 --nmicros 2 --inflight 4 \
2>&1 | tee ${LOGS}/naive.yshape.log