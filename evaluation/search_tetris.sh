#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/simulator
mkdir -p $LOGS

set -x

premise="vshape kshape mshape nnshape xshape"
# premise="nnshape_eager"

# VShape
if echo $premise | grep -qw 'vshape'; then

    python examples/simulator/cases_tetris.py \
        --premise vshape --ndevs 4 --nmicros 4 --memory 4 \
    2>&1 | tee ${LOGS}/tetris.vshape.train.log

    python examples/simulator/cases_tetris.py \
        --premise vshape --ndevs 4 --nmicros 4 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tetris.vshape.infer.log

fi

# XShape
if echo $premise | grep -qw 'xshape'; then

    python examples/simulator/cases_tetris.py \
        --premise xshape --ndevs 4 --nmicros 3 --memory 4 \
    2>&1 | tee ${LOGS}/tetris.xshape.train.log

    python examples/simulator/cases_tetris.py \
        --premise xshape --ndevs 4 --nmicros 3 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tetris.xshape.infer.log

fi

# NNShape
if echo $premise | grep -qw 'nnshape'; then

    python examples/simulator/cases_tetris.py \
        --premise nnshape --ndevs 4 --nmicros 6 --memory 16 \
    2>&1 | tee ${LOGS}/tetris.nnshape.train.log

    python examples/simulator/cases_tetris.py \
        --premise nnshape --ndevs 4 --nmicros 6 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tetris.nnshape.infer.log

fi

# NNShape-eager
if echo $premise | grep -qw 'nnshape_eager'; then

    python examples/simulator/cases_tetris.py \
        --premise nnshape_eager --ndevs 4 --nmicros 6 --memory 6 \
    2>&1 | tee ${LOGS}/tetris.nnshape_eager.train.log

    python examples/simulator/cases_tetris.py \
        --premise nnshape --ndevs 4 --nmicros 6 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tetris.nnshape_eager.infer.log

fi

# MShape
if echo $premise | grep -qw 'mshape'; then

    python examples/simulator/cases_tetris.py \
        --premise mshape --ndevs 4 --nmicros 6 --memory 11 \
    2>&1 | tee ${LOGS}/tetris.mshape.train.log

    python examples/simulator/cases_tetris.py \
        --premise mshape --ndevs 4 --nmicros 6 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tetris.mshape.infer.log

fi

# K-Shape
if echo $premise | grep -qw 'kshape'; then

    python examples/simulator/cases_tetris.py \
        --premise kshape --ndevs 4 --nmicros 3 --memory 6 \
    2>&1 | tee ${LOGS}/tetris.kshape.train.log

    python examples/simulator/cases_tetris.py \
        --premise kshape --ndevs 4 --nmicros 2 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tetris.kshape.infer.log

fi
