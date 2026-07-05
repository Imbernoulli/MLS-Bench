#!/bin/bash
# reg-smoothness-weight: the registration method is FIXED to the learned dense
# deformable field (VoxelMorph U-Net + NCC similarity). The agent designs ONLY the
# smoothness-regulariser weight lambda on the displacement field
# (loss = NCC + lambda * grad_smoothness). The harness sweeps THREE deformation
# magnitudes (small / medium / large) and reports per-setting warped-moving PSNR
# (dB; HIGHER better) AND the folding fraction (non-diffeomorphic pixels; LOWER
# better). Data, U-Net, similarity, optimiser, steps, seed and eval are fixed.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/deformable-registration

python harness.py \
    --task smoothness \
    --solution solution/smoothness.py \
    --label smoothness \
    --seed ${SEED:-42} \
    --steps ${REG_STEPS:-800} \
    --data-root ${REG_DATA_ROOT:-/data/deformable-registration}
