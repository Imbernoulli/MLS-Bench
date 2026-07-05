#!/bin/bash
# stereo-disparity-range: a FIXED small GC-Net-style stereo net is trained for a
# short schedule on a deterministic SYNTHETIC rectified stereo dataset (a textured
# left image + an exactly-known disparity field; the right image is the left
# shifted LEFTWARD by the disparity, so GT disparity and EPE are exact).
# The agent designs ONLY the max disparity (number of levels) the cost volume
# covers. The features, cost volume, 3D aggregation, soft-argmin readout, loss and
# schedule are fixed. This is the EASY difficulty setting (disparities up to ~16 px).
# Metric: validation EPE (disparity end-point error, px; LOWER is better).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/stereo-matching

python harness.py \
    --task drange \
    --severity easy \
    --solution solution/drange.py \
    --label easy \
    --seed ${SEED:-42} \
    --steps ${STEREO_STEPS:-1200}
