#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH
LOGS=logs/simulator
mkdir -p $LOGS

set -ex


nmicros=(1 2 4 6)
premise=(vshape xshape kshape mshape nnshape)

for m in ${nmicros[@]}; do
    for p in ${premise[@]}; do
        python examples/simulator/cases_full.py \
            --premise $p --ndevs 4 --nmicros $m --memory 16 \
        2>&1 | tee ${LOGS}/naive.$p.nmicros$m.train.log
    done
done

for m in ${nmicros[@]}; do
    for p in ${premise[@]}; do
        python examples/simulator/cases_full.py \
            --premise $p --ndevs 4 --nmicros $m --memory 16 --infer \
        2>&1 | tee ${LOGS}/naive.$p.nmicros$m.infer.log
    done
done