#!/bin/bash
# Mamba Appendix E.1 selective-copying protocol; Delta finalization is the only variable.
set -euo pipefail

cd /workspace/mamba/harness
python design_harness.py \
    --solution ../solution/delta_softplus.py --hook finalize_dt \
    --protocol mamba_selective_copy_paper_e1_v1 --label paper_e1 \
    --L 4096 --M 16 --A 16 --d_model 64 --d_state 16 --n_layer 2 \
    --steps 400000 --batch 64 --lr 1e-4 \
    --optimizer adam --weight-decay 0 --grad-clip 1 --eval-batches 16 \
    --seed "${SEED:-42}"
