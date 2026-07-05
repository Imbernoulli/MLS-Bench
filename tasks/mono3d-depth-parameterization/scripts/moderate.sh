#!/bin/bash
# mono3d-depth-parameterization [moderate difficulty tier]: train the fixed mono-3D model with the
# agent's DEPTH parameterization (solution/depth_param.py), then score AP3D / 3D-IoU on the
# TEST objects in KITTI's official 'moderate' difficulty tier. Training is on the full fixed
# train split; only scoring is sliced to this tier.
set -e
# NOTE: do NOT hardcode CUDA_VISIBLE_DEVICES here. The verifier
# (score_task.py) runs this task's test_cmds concurrently within the same
# group and assigns each one a DISTINCT physical GPU via CUDA_VISIBLE_DEVICES
# in the subprocess env (see _allocate_group_gpu_assignments/_run_eval_wave).
# Hardcoding =0 here overrides that assignment and forces concurrent runs
# onto the same physical GPU, causing OOM collisions. Inherit whatever the
# harness set.
cd /workspace/mono3d-detection

python harness.py \
    --task depth \
    --solution solution/depth_param.py \
    --label moderate \
    --setting moderate \
    --seed ${SEED:-42}
