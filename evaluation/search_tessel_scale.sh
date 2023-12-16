#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/simulator1214
mkdir -p $LOGS

set -x

# premise="vshape kshape mshape nnshape xshape"
premise="mshape"
# premise="nnshape_eager"

# VShape
if echo $premise | grep -qw 'vshape'; then

    python examples/simulator/cases_tessel.py \
        --placement vshape --ndevs 4 --nmicros 4 --memory 4 \
    2>&1 | tee ${LOGS}/tessel.vshape.4devs.lazy.log
    
    python examples/simulator/cases_tessel.py \
        --placement vshape --ndevs 5 --nmicros 5 --memory 5 \
    2>&1 | tee ${LOGS}/tessel.vshape.5devs.lazy.log
    
    python examples/simulator/cases_tessel.py \
        --placement vshape --ndevs 6 --nmicros 6 --memory 6 \
    2>&1 | tee ${LOGS}/tessel.vshape.6devs.lazy.log
    
    python examples/simulator/cases_tessel.py \
        --placement vshape --ndevs 7 --nmicros 7 --memory 7 \
    2>&1 | tee ${LOGS}/tessel.vshape.7devs.lazy.log
    
    python examples/simulator/cases_tessel.py \
        --placement vshape --ndevs 8 --nmicros 8 --memory 8 \
    2>&1 | tee ${LOGS}/tessel.vshape.8devs.lazy.log

    # =======================================

    # python examples/simulator/cases_tessel.py \
    #     --placement vshape --ndevs 4 --nmicros 4 --memory 20 --fast-search \
    # 2>&1 | tee ${LOGS}/tessel.vshape.4devs.fast.log
    # 
    # python examples/simulator/cases_tessel.py \
    #     --placement vshape --ndevs 5 --nmicros 5 --memory 20 --fast-search \
    # 2>&1 | tee ${LOGS}/tessel.vshape.5devs.fast.log
    # 
    # python examples/simulator/cases_tessel.py \
    #     --placement vshape --ndevs 6 --nmicros 6 --memory 20 --fast-search \
    # 2>&1 | tee ${LOGS}/tessel.vshape.6devs.fast.log
    # 
    # python examples/simulator/cases_tessel.py \
    #     --placement vshape --ndevs 7 --nmicros 7 --memory 20 --fast-search \
    # 2>&1 | tee ${LOGS}/tessel.vshape.7devs.fast.log
    # 
    # python examples/simulator/cases_tessel.py \
    #     --placement vshape --ndevs 8 --nmicros 8 --memory 20 --fast-search \
    # 2>&1 | tee ${LOGS}/tessel.vshape.8devs.fast.log

fi

# XShape
if echo $premise | grep -qw 'xshape'; then

    python examples/simulator/cases_tessel.py \
        --placement xshape --ndevs 4 --nmicros 3 --memory 4 \
    2>&1 | tee ${LOGS}/tessel.xshape.train.log

    python examples/simulator/cases_tessel.py \
        --placement xshape --ndevs 4 --nmicros 3 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tessel.xshape.infer.log

fi

# NNShape
if echo $premise | grep -qw 'nnshape'; then

    # all block memory 1 and -1: --memory 36
    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 4 --nmicros 6 --memory 6 \
    2>&1 | tee ${LOGS}/tessel.nnshape.train.log

    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 4 --nmicros 6 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tessel.nnshape.infer.log

fi

# NNShape-eager
if echo $premise | grep -qw 'nnshape_eager'; then

    python examples/simulator/cases_tessel.py \
        --placement nnshape_eager --ndevs 4 --nmicros 6 --memory 6 \
    2>&1 | tee ${LOGS}/tessel.nnshape_eager.train.log

    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 4 --nmicros 6 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tessel.nnshape_eager.infer.log

fi

# MShape
if echo $premise | grep -qw 'mshape'; then

    # all block memory 1 and -1: --memory 15
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 4 --nmicros 6 --memory 6 \
    2>&1 | tee ${LOGS}/tessel.mshape.4devs.lazy.log

    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 5 --nmicros 7 --memory 7 \
    2>&1 | tee ${LOGS}/tessel.mshape.5devs.lazy.log

    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 6 --nmicros 8 --memory 8 \
    2>&1 | tee ${LOGS}/tessel.mshape.6devs.lazy.log

    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 7 --nmicros 9 --memory 9 \
    2>&1 | tee ${LOGS}/tessel.mshape.7devs.lazy.log

    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 8 --nmicros 10 --memory 10 \
    2>&1 | tee ${LOGS}/tessel.mshape.8devs.lazy.log

fi

# K-Shape
if echo $premise | grep -qw 'kshape'; then

    python examples/simulator/cases_tessel.py \
        --placement kshape --ndevs 4 --nmicros 3 --memory 6 \
    2>&1 | tee ${LOGS}/tessel.kshape.train.log

    python examples/simulator/cases_tessel.py \
        --placement kshape --ndevs 4 --nmicros 2 --memory 4 --infer \
    2>&1 | tee ${LOGS}/tessel.kshape.infer.log

fi
