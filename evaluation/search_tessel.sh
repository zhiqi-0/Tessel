#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/simulator1013-lazy
mkdir -p $LOGS

set -x

premise="vshape kshape mshape nnshape xshape"
# premise="nnshape_eager"

# VShape
if echo $premise | grep -qw 'vshape'; then

    python examples/simulator/cases_tessel.py \
        --premise vshape --ndevs 4 --nmicros 4 --memory 4 \
    2>&1 | tee ${LOGS}/tessel.vshape.train.log

    python examples/simulator/cases_tessel.py \
        --premise vshape --ndevs 4 --nmicros 4 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tessel.vshape.infer.log

fi

# XShape
if echo $premise | grep -qw 'xshape'; then

    python examples/simulator/cases_tessel.py \
        --premise xshape --ndevs 4 --nmicros 3 --memory 4 \
    2>&1 | tee ${LOGS}/tessel.xshape.train.log

    python examples/simulator/cases_tessel.py \
        --premise xshape --ndevs 4 --nmicros 3 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tessel.xshape.infer.log

fi

# NNShape
if echo $premise | grep -qw 'nnshape'; then

    # all block memory 1 and -1: --memory 36
    python examples/simulator/cases_tessel.py \
        --premise nnshape --ndevs 4 --nmicros 6 --memory 6 \
    2>&1 | tee ${LOGS}/tessel.nnshape.train.log

    python examples/simulator/cases_tessel.py \
        --premise nnshape --ndevs 4 --nmicros 6 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tessel.nnshape.infer.log

fi

# NNShape-eager
if echo $premise | grep -qw 'nnshape_eager'; then

    python examples/simulator/cases_tessel.py \
        --premise nnshape_eager --ndevs 4 --nmicros 6 --memory 6 \
    2>&1 | tee ${LOGS}/tessel.nnshape_eager.train.log

    python examples/simulator/cases_tessel.py \
        --premise nnshape --ndevs 4 --nmicros 6 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tessel.nnshape_eager.infer.log

fi

# MShape
if echo $premise | grep -qw 'mshape'; then

    # all block memory 1 and -1: --memory 15
    python examples/simulator/cases_tessel.py \
        --premise mshape --ndevs 4 --nmicros 6 --memory 6 \
    2>&1 | tee ${LOGS}/tessel.mshape.train.log

    python examples/simulator/cases_tessel.py \
        --premise mshape --ndevs 4 --nmicros 6 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tessel.mshape.infer.log

fi

# K-Shape
if echo $premise | grep -qw 'kshape'; then

    python examples/simulator/cases_tessel.py \
        --premise kshape --ndevs 4 --nmicros 3 --memory 6 \
    2>&1 | tee ${LOGS}/tessel.kshape.train.log

    python examples/simulator/cases_tessel.py \
        --premise kshape --ndevs 4 --nmicros 2 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tessel.kshape.infer.log

fi
