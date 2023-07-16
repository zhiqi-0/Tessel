#!/bin/bash

set -x

TETRIS_HOME=/workspace
WORKERS=(worker-1 worker-2 worker-3)

for WORKER in ${WORKERS[@]}; do
    echo "send tetris to worker ${WORKER}"
    scp -r ${TETRIS_HOME}/Tetris ${WORKER}:${TETRIS_HOME}
done
