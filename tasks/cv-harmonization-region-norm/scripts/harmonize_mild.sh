#!/bin/bash
# cv-harmonization-region-norm: train the harmonizer (composite+mask -> harmonized) with
# the agent-designed backbone on the MILD appearance-mismatch setting, then score the
# FOREGROUND-region PSNR of the harmonized output vs the real GT on the held-out split.
set -e
# NOTE: do NOT hardcode CUDA_VISIBLE_DEVICES here. The verifier
# (score_task.py) runs this task's test_cmds concurrently within the same
# group and assigns each one a DISTINCT physical GPU via CUDA_VISIBLE_DEVICES
# in the subprocess env (see _allocate_group_gpu_assignments/_run_eval_wave).
# Hardcoding =0 here overrides that assignment and forces concurrent runs
# onto the same physical GPU, causing OOM collisions. Inherit whatever the
# harness set.
cd /workspace/image-harmonization
python harness.py \
    --data-root ${HARMONY_DATA_ROOT:-/data/image-harmonization} \
    --surface network \
    --severity mild \
    --label mild \
    --solution solution/network.py \
    --iters ${HARMONY_ITERS:-500} \
    --seed ${SEED:-42}
