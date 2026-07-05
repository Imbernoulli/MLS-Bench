#!/bin/bash
# stereo-disparity-refinement: train the FIXED small GC-Net/PSMNet-style stereo net with the agent's
# design (surface `refine`) on the medium difficulty setting (disparity
# range), then report validation EPE (LOWER is better).
set -e
# NOTE: do NOT hardcode CUDA_VISIBLE_DEVICES here. The verifier
# (score_task.py) runs this task's test_cmds concurrently within the same
# group and assigns each one a DISTINCT physical GPU via CUDA_VISIBLE_DEVICES
# in the subprocess env (see _allocate_group_gpu_assignments/_run_eval_wave).
# Hardcoding =0 here overrides that assignment and forces concurrent runs
# onto the same physical GPU, causing OOM collisions. Inherit whatever the
# harness set.
cd /workspace/stereo-matching

# NOTE: this surface ships at a LONGER schedule (3000 steps, not the package
# default 1200 steps) -- at 1200 steps the easy/hard none/residual order was
# inverted on real Middlebury data (see vendor/stereo-matching/anchors/
# README.md); at 3000 steps it cleanly resolves on both seeds 42/123.
# Hardcoded here regardless of the STEREO_STEPS package env var.
python harness.py \
    --task refine \
    --severity medium \
    --solution solution/refine.py \
    --label medium \
    --seed ${SEED:-42} \
    --steps 3000
