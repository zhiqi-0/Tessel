#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/fast_ratio
mkdir -p $LOGS

set -x

# premise="kshape mshape nnshape"
premise="mshape"


# NNShape
if echo $premise | grep -qw 'nnshape'; then

    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 4 --memory 20 --fast-search \
    2>&1 | tee ${LOGS}/tessel.nnshape.4devs.fast.log
    
    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 5 --memory 20 --fast-search \
    2>&1 | tee ${LOGS}/tessel.nnshape.5devs.fast.log
    
    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 6 --memory 20 --fast-search \
    2>&1 | tee ${LOGS}/tessel.nnshape.6devs.fast.log
    
    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 7 --memory 20 --fast-search \
    2>&1 | tee ${LOGS}/tessel.nnshape.7devs.fast.log

    python examples/simulator/cases_tessel.py \
        --placement nnshape --ndevs 8 --memory 20 --fast-search \
    2>&1 | tee ${LOGS}/tessel.nnshape.8devs.fast.log

fi


# MShape
if echo $premise | grep -qw 'mshape'; then

    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 6 --memory 20 --fast-search --fast-ratio 0.02 \
    2>&1 | tee ${LOGS}/mshape.6devs.fast.ratio2.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 6 --memory 20 --fast-search --fast-ratio 0.03 \
    2>&1 | tee ${LOGS}/mshape.6devs.fast.ratio3.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 6 --memory 20 --fast-search --fast-ratio 0.04 \
    2>&1 | tee ${LOGS}/mshape.6devs.fast.ratio4.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 6 --memory 20 --fast-search --fast-ratio 0.05 \
    2>&1 | tee ${LOGS}/mshape.6devs.fast.ratio5.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 6 --memory 20 --fast-search --fast-ratio 0.10 \
    2>&1 | tee ${LOGS}/mshape.6devs.fast.ratio10.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 6 --memory 20 --fast-search --fast-ratio 0.15 \
    2>&1 | tee ${LOGS}/mshape.6devs.fast.ratio15.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 6 --memory 20 --fast-search --fast-ratio 0.20 \
    2>&1 | tee ${LOGS}/mshape.6devs.fast.ratio20.log
    
    python examples/simulator/cases_tessel.py \
        --placement mshape --ndevs 6 --memory 20 --fast-search --fast-ratio 0.25 \
    2>&1 | tee ${LOGS}/mshape.6devs.fast.ratio25.log
fi
