#!/bin/bash
# reg-similarity-loss: the registration method is FIXED to the learned dense
# deformable field (VoxelMorph U-Net). The agent designs ONLY the image-similarity
# term that drives the field (MSE vs local NCC). The harness sweeps THREE
# deformation magnitudes (small / medium / large) and reports per-setting
# warped-moving vs fixed PSNR (dB; HIGHER better). Data, U-Net, smoothness weight,
# optimiser, steps, seed and eval are all fixed.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/deformable-registration

python harness.py \
    --task similarity \
    --solution solution/similarity.py \
    --label similarity \
    --seed ${SEED:-42} \
    --steps ${REG_STEPS:-800} \
    --data-root ${REG_DATA_ROOT:-/data/deformable-registration}
