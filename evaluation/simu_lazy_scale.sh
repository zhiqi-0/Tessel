#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/scale
mkdir -p $LOGS

set -x

premise="vshape kshape mshape nnshape xshape"
# premise="mshape"

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

# MShape
if echo $premise | grep -qw 'mshape'; then

    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 4 --nmicros 6 --memory 20 \
    2>&1 | tee ${LOGS}/tessel.mshape.4devs.lazy.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 5 --nmicros 7 --memory 20 \
    2>&1 | tee ${LOGS}/tessel.mshape.5devs.lazy.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 6 --nmicros 9 --memory 20 \
    2>&1 | tee ${LOGS}/tessel.mshape.6devs.lazy.log

    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 7 --nmicros 10 --memory 20 \
    2>&1 | tee ${LOGS}/tessel.mshape.7devs.lazy.log

    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 8 --nmicros 12 --memory 20 \
    2>&1 | tee ${LOGS}/tessel.mshape.8devs.lazy.log

fi

# NNShape
if echo $premise | grep -qw 'nnshape'; then

    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 4 --nmicros 6 --memory 20 \
    2>&1 | tee ${LOGS}/tessel.nnshape.4devs.lazy.log

    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 5 --nmicros 7 --memory 20 \
    2>&1 | tee ${LOGS}/tessel.nnshape.5devs.lazy.log

    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 6 --nmicros 10 --memory 20 \
    2>&1 | tee ${LOGS}/tessel.nnshape.6devs.lazy.log

    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 7 --nmicros 11 --memory 20 \
    2>&1 | tee ${LOGS}/tessel.nnshape.7devs.lazy.log
    
    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 8 --nmicros 12 --memory 20 \
    2>&1 | tee ${LOGS}/tessel.nnshape.8devs.lazy.log

fi


# K-Shape
if echo $premise | grep -qw 'kshape'; then

    python examples/simulator/cases_tessel.py \
        --placement kshape --ndevs 4 --nmicros 2 --memory 20 --infer \
    2>&1 | tee ${LOGS}/tessel.kshape.infer.log

    python examples/simulator/cases_tessel.py \
        --placement kshape --ndevs 4 --nmicros 3 --memory 20 --infer \
    2>&1 | tee ${LOGS}/tessel.kshape.infer.log

    python examples/simulator/cases_tessel.py \
        --placement kshape --ndevs 4 --nmicros 4 --memory 20 --infer \
    2>&1 | tee ${LOGS}/tessel.kshape.infer.log

fi
