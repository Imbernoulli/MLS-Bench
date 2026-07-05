#!/bin/bash
# cv-matting-arch: train the matting net with the agent's 'arch' surface, then score alpha SAD in
# the trimap UNKNOWN band on the 'wide' trimap-width val setting.
set -e
# NOTE: do NOT hardcode CUDA_VISIBLE_DEVICES here. The verifier
# (score_task.py) runs this task's test_cmds concurrently within the same
# group and assigns each one a DISTINCT physical GPU via CUDA_VISIBLE_DEVICES
# in the subprocess env (see _allocate_group_gpu_assignments/_run_eval_wave).
# Hardcoding =0 here overrides that assignment and forces concurrent runs
# onto the same physical GPU, causing OOM collisions. Inherit whatever the
# harness set.
cd /workspace/image-matting
python harness.py \
    --data-root ${MATTING_DATA_ROOT:-/data/image-matting/composites} \
    --surface arch \
    --trimap-width wide \
    --label wide \
    --solution solution/arch.py \
    --iters 400 \
    --seed ${SEED:-42}
