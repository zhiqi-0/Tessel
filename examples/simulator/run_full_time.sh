#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH
LOGS=logs/simulator
mkdir -p $LOGS

set -ex


nmicros=(1 2 3 4 5 6)
premise=(vshape xshape mshape yshape)

for m in ${nmicros[@]}; do
    for p in ${premise[@]}; do
        echo > ${LOGS}/naive.$p.nmicros$m.log

        python examples/simulator/cases_full.py \
            --premise $p --ndevs 4 --nmicros $m --inflight 10 \
        2>&1 | tee -a ${LOGS}/naive.$p.nmicros$m.log
    done
done