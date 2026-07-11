#!/bin/bash
# flow-coupling-transform: build a normalizing flow from the agent's coupling
# literal design in solution/coupling.py, train it for a fixed budget on the
# 2-D checkerboard density, then report exact held-out NLL (nats, lower is better).
set -euo pipefail
# The rendered H20 verifier serializes all settings on its assigned GPU.
# Inherit CUDA_VISIBLE_DEVICES from the scheduler instead of overriding it.
cd /workspace/normflows-density || exit 111

exec python harness_flow.py \
    --solution solution/coupling.py \
    --surface coupling_transform \
    --target checkerboard \
    --seed "${SEED:-42}" \
    --steps 20000 \
    --batch-size 512 \
    --lr 5e-4 \
    --n-train 30000 \
    --n-test 30000
