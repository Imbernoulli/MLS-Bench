#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/image-deblur
python harness.py --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} --surface loss --blur-type el --label el --solution solution/losskind.py --iters ${DEBLUR_ITERS:-1500} --seed ${SEED:-42}
