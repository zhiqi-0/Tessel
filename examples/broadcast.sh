#!/bin/bash

set -x

tessel_HOME=/workspace
WORKERS=(worker-1 worker-2 worker-3)

for WORKER in ${WORKERS[@]}; do
    echo "send Tessel to worker ${WORKER}"
    scp -r ${tessel_HOME}/Tessel ${WORKER}:${tessel_HOME}
done
