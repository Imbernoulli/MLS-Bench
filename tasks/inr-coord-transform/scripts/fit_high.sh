#!/bin/bash
# inr-coord-transform (HIGH-frequency signal): fit a fixed coordinate MLP to the high target with the
# agent's chosen design surface (COORDINATE NORMALIZATION / transform before the encoding (inflate vs unit vs identity).), then score full-grid reconstruction PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

python harness.py \
    --solution solution/coord_transform.py \
    --signal high \
    --seed "${SEED:-0}" \
    --label high
