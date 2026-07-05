#!/bin/bash
# cv-count-architecture (medium scene): train the agent-designed full image->density
# counter on the MEDIUM crowd-density scene, score counting MAE on its higher-count val.
set -e
# NOTE: do NOT hardcode CUDA_VISIBLE_DEVICES here. The verifier
# (score_task.py) runs this task's test_cmds concurrently within the same
# group and assigns each one a DISTINCT physical GPU via CUDA_VISIBLE_DEVICES
# in the subprocess env (see _allocate_group_gpu_assignments/_run_eval_wave).
# Hardcoding =0 here overrides that assignment and forces concurrent runs
# onto the same physical GPU, causing OOM collisions. Inherit whatever the
# harness set.
cd /workspace/crowd-counting
python harness.py \
    --data-root ${COUNT_DATA_ROOT:-/data/crowd-counting}/medium \
    --surface arch \
    --label medium \
    --solution solution/arch.py \
    --iters ${COUNT_ITERS:-300} \
    --seed ${SEED:-42}
