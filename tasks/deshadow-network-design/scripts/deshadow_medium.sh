#!/bin/bash
# deshadow-mask-guidance (setting: medium cast shadow): train the residual deshadower with the
# agent-designed backbone, then score SHADOW-REGION PSNR on the held-out val split.
set -e
# NOTE: do NOT hardcode CUDA_VISIBLE_DEVICES here. The verifier
# (score_task.py) runs this task's test_cmds concurrently within the same
# group and assigns each one a DISTINCT physical GPU via CUDA_VISIBLE_DEVICES
# in the subprocess env (see _allocate_group_gpu_assignments/_run_eval_wave).
# Hardcoding =0 here overrides that assignment and forces concurrent runs
# onto the same physical GPU, causing OOM collisions. Inherit whatever the
# harness set.
cd /workspace/image-deshadow
python harness.py \
    --data-root ${DESHADOW_DATA_ROOT:-/data/image-deshadow}/medium \
    --surface network \
    --label medium \
    --solution solution/network.py \
    --iters ${DESHADOW_ITERS:-400} \
    --seed ${SEED:-42}
