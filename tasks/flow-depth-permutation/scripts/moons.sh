#!/bin/bash
# flow-depth-permutation: build a normalizing flow from the agent's literal depth
# choice with fixed swap mixing, then train it for a fixed
# budget on the 2-D two-moons density, then report exact held-out NLL (lower better).
set -euo pipefail
cd /workspace/normflows-density || exit 111

exec python harness_flow.py \
    --solution solution/depth.py \
    --surface depth \
    --target moons \
    --seed "${SEED:-42}" \
    --steps 20000 \
    --batch-size 512 \
    --lr 5e-4 \
    --n-train 30000 \
    --n-test 30000
