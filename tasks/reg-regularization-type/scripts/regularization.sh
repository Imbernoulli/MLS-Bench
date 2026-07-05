#!/bin/bash
# reg-regularization-type: a FIXED learned dense deformable registration pipeline (VoxelMorph U-Net)
# aligns a MOVING image to a FIXED image on a deterministic SYNTHETIC dataset
# (a clean richly-textured fixed image warped by an exactly-known smooth non-rigid
# deformation, so warped-moving vs fixed PSNR and landmark TRE are exact). The
# agent designs ONLY the regularization surface (build_reg_type). The harness sweeps
# THREE deformation magnitudes (small / medium / large) and reports per-setting
# warped-moving PSNR (dB; HIGHER better), landmark TRE (px; LOWER), folding
# fraction and NCC. Everything else (data, similarity, optimiser, seed, eval) is
# fixed.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/deformable-registration

python harness.py \
    --task regularization \
    --solution solution/regularization.py \
    --label regularization \
    --seed ${SEED:-42} \
    --steps ${REG_STEPS:-800} \
    --data-root ${REG_DATA_ROOT:-/data/deformable-registration}
