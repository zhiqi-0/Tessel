#!/bin/bash

export PYTHONPATH=.:$PYTHONPATH

LOGS=logs/simulator
mkdir -p $LOGS

set -x

memory=(1 2 3 4 5 6)

# premise=(vshape)
# nmicros=4

# premise=(xshape)
# nmicros=3

premise=(nnshape mshape)
memory=(3 4 5 6 7 8 9 10 11 12 13 14 15 16)
nmicros=6

# premise=(kshape)
# nmicros=3


for p in ${premise[@]}; do
    for m in ${memory[@]}; do
        python examples/simulator/cases_tessel.py \
            --premise $p --ndevs 4 --nmicros $nmicros --memory $m \
        2>&1 | tee ${LOGS}/abalation.tessel.$p.mem$m.log
    done
done
