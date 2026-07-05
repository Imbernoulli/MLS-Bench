#!/bin/bash
# deblur-global-residual [rl = mild motion blur, larger severity]: train the fixed deblur
# net with the agent-designed global-residual configuration, score deblur PSNR on the
# held-out val split.
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
    --blur-type rl \
    --label rl \
    --solution solution/residual.py \
    --iters ${DEBLUR_ITERS:-1500} \
    --seed ${SEED:-42}
