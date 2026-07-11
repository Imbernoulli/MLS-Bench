#!/bin/bash
# inr-coord-transform (LOW-frequency signal): fit a fixed coordinate MLP to the low target with the
# agent's chosen design surface (COORDINATE NORMALIZATION / transform before the encoding (inflate vs unit vs identity).), then score full-grid reconstruction PSNR.
set -euo pipefail
cd /workspace/inr-signal-fitting

python harness.py \
    --solution solution/coord_transform.py \
    --signal low \
    --seed "${SEED:-0}" \
    --label low
