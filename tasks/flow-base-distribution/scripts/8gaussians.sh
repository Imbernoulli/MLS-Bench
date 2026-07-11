#!/bin/bash
set -euo pipefail
cd /workspace/normflows-density || exit 111

exec python harness_flow.py \
    --solution solution/base_dist.py \
    --surface base_distribution \
    --target 8gaussians \
    --seed "${SEED:-42}" \
    --steps 20000 \
    --batch-size 512 \
    --lr 5e-4 \
    --n-train 30000 \
    --n-test 30000
