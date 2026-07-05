#!/bin/bash
# stereo-cost-upsampling: train the FIXED small GC-Net/PSMNet-style stereo net with the agent's
# design (surface `upsampling`) on the easy difficulty setting (disparity
# range), then report validation EPE (LOWER is better).
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/stereo-matching

# NOTE: this surface ships at a LONGER schedule (3000 steps, not the package
# default 1200 steps) -- at 1200 steps the hard-severity order between
# nearest/trilinear is not yet resolved (see vendor/stereo-matching/anchors/
# README.md); hardcoded here regardless of the STEREO_STEPS package env var.
python harness.py \
    --task upsampling \
    --severity easy \
    --solution solution/upsampling.py \
    --label easy \
    --seed ${SEED:-42} \
    --steps 3000
