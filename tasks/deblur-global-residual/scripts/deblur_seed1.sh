#!/bin/bash
# DEPRECATED hidden-seed script -- not referenced by config.json. The scored settings are
# the three severities rs/rm/rl (deblur_rs.sh / deblur_rm.sh / deblur_rl.sh). Kept as a
# harmless seed-1 run of the primary (rm) setting if invoked directly.
set -e
# NOTE: do NOT hardcode CUDA_VISIBLE_DEVICES here. The verifier
# (score_task.py) runs this task's test_cmds concurrently within the same
# group and assigns each one a DISTINCT physical GPU via CUDA_VISIBLE_DEVICES
# in the subprocess env (see _allocate_group_gpu_assignments/_run_eval_wave).
# Hardcoding =0 here overrides that assignment and forces concurrent runs
# onto the same physical GPU, causing OOM collisions. Inherit whatever the
# harness set.
cd /workspace/image-deblur
python harness.py \
    --data-root ${DEBLUR_DATA_ROOT:-/data/image-deblur} \
    --surface residual \
    --blur-type rm \
    --label rm \
    --solution solution/residual.py \
    --iters ${DEBLUR_ITERS:-1500} \
    --seed 1
