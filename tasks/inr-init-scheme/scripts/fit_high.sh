#!/bin/bash
# inr-init-scheme (HIGH-frequency signal): fit a fixed-capacity coordinate MLP to the REAL Kodak HIGH-frequency target (kodim13, forest/rock/mountain texture) with the agent's chosen SIREN INITIALIZATION / w0, then score PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

python harness.py \
    --solution solution/init_scheme.py \
    --signal high \
    --seed "${SEED:-0}" \
    --label high
