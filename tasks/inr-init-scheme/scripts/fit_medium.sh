#!/bin/bash
# inr-init-scheme (MEDIUM-frequency signal): fit a fixed-capacity coordinate MLP to the REAL Kodak MEDIUM-frequency target (kodim07, flowers + window lattice) with the agent's chosen SIREN INITIALIZATION / w0, then score PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

python harness.py \
    --solution solution/init_scheme.py \
    --signal medium \
    --seed "${SEED:-0}" \
    --label medium
