#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/simulator
mkdir -p $LOGS

set -ex

memory=(1 2 3 4 5 6)
premise=(vshape xshape mshape yshape)

for p in ${premise[@]}; do
    for m in ${memory[@]}; do
        echo  > ${LOGS}/abalation.tetris.$p.mem$m.log

        python examples/simulator/cases_tetris.py \
            --premise $p --ndevs 4 --nmicros 6 --inflight $m \
        2>&1 | tee -a ${LOGS}/abalation.tetris.$p.mem$m.log
    done
done
