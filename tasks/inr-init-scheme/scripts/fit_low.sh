#!/bin/bash
# inr-init-scheme (LOW-frequency signal): fit a fixed-capacity coordinate MLP to the REAL Kodak LOW-frequency target (kodim10, calm dockside scene) with the agent's chosen SIREN INITIALIZATION / w0, then score PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

python harness.py \
    --solution solution/init_scheme.py \
    --signal low \
    --seed "${SEED:-0}" \
    --label low
