#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/simulator
mkdir -p $LOGS

set -ex

memory=(1 2 3 4 5 6)

premise=(vshape)
nmicros=4

premise=(xshape)
nmicros=3

premise=(nnshape mshape)
nmicros=6

premise=(kshape)
nmicros=3


for p in ${premise[@]}; do
    for m in ${memory[@]}; do
        python examples/simulator/cases_tetris.py \
            --premise $p --ndevs 4 --nmicros $nmicros --inflight $m \
        2>&1 | tee ${LOGS}/abalation.tetris.$p.mem$m.log
    done
done
