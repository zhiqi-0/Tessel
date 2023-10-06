#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/simulator
mkdir -p $LOGS

set -ex

nmicros=(1 2 3 4 5 6)
premise=(vshape xshape kshape mshape nnshape)
# premise=(nnshape)

for m in ${nmicros[@]}; do
    for p in ${premise[@]}; do
        python examples/simulator/cases_tetris.py \
            --premise $p --ndevs 4 --nmicros $m --memory 16 \
        2>&1 | tee ${LOGS}/abalation.tetris.$p.nmicros$m.log
    done
done
