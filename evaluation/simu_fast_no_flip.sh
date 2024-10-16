#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/no-flip
mkdir -p $LOGS

set -x

premise="kshape mshape nnshape"
# premise="nnshape"

# MShape
if echo $premise | grep -qw 'mshape'; then

    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 4 --memory 20 --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.mshape.4devs.fast.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 5 --memory 20 --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.mshape.5devs.fast.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 6 --memory 20 --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.mshape.6devs.fast.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 7 --memory 20 --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.mshape.7devs.fast.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 8 --memory 20 --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.mshape.8devs.fast.log
fi


# NNShape
if echo $premise | grep -qw 'nnshape'; then

    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 4 --memory 20 --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.nnshape.4devs.fast.log
    
    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 5 --memory 20 --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.nnshape.5devs.fast.log
    
    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 6 --memory 20 --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.nnshape.6devs.fast.log
    
    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 7 --memory 20 --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.nnshape.7devs.fast.log

    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 8 --memory 20 --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.nnshape.8devs.fast.log

fi


# K-Shape
if echo $premise | grep -qw 'kshape'; then

    python examples/simulator/cases_tessel.py \
        --placement kshape --ndevs 4 --memory 20 --infer --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.kshape.4devs.fast.log

    python examples/simulator/cases_tessel.py \
        --placement kshape --ndevs 6 --memory 20 --infer --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.kshape.6devs.fast.log

    python examples/simulator/cases_tessel.py \
        --placement kshape --ndevs 8 --memory 20 --infer --fast-search --fast-no-flip \
    2>&1 | tee ${LOGS}/tessel.kshape.8devs.fast.log

fi
