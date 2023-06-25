#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH
LOGS=logs/simulator
mkdir -p $LOGS

set -ex


nmicros=(1 2 4 8)
premise=(nnshape mshape kshape)

for m in ${nmicros[@]}; do
    for p in ${premise[@]}; do
        echo > ${LOGS}/naive.$p.nmicros$m.log

        python examples/simulator/cases_full.py \
            --premise $p --ndevs 4 --nmicros $m --inflight 10 \
        2>&1 | tee ${LOGS}/naive.$p.nmicros$m.train.log
    done
done

for m in ${nmicros[@]}; do
    for p in ${premise[@]}; do
        echo > ${LOGS}/naive.$p.nmicros$m.log

        python examples/simulator/cases_full.py \
            --premise $p --ndevs 4 --nmicros $m --inflight 10 --infer \
        2>&1 | tee ${LOGS}/naive.$p.nmicros$m.infer.log
    done
done