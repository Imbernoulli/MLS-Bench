#!/bin/bash
# flow-arch-family: build a normalizing flow from the agent's architecture-family
# literal choice in solution/arch.py, train it for a fixed budget on the 2-D
# checkerboard density, then report exact held-out NLL (nats, lower is better).
set -euo pipefail
cd /workspace/normflows-density || exit 111

exec python harness_flow.py \
    --solution solution/arch.py \
    --surface architecture \
    --target checkerboard \
    --seed "${SEED:-42}" \
    --steps 20000 \
    --batch-size 512 \
    --lr 5e-4 \
    --n-train 30000 \
    --n-test 30000
