#!/bin/bash
set -euo pipefail

cd /workspace/compressai
test -f harness_zoo_entropy.py
test -f harness_zoo_policy.py
test -f policy_parser.py
test -f /data/compressai-zoo/protocol.json
test -d /data/compressai-zoo/kodak
test -d /data/compressai-zoo/checkpoints

/opt/conda/bin/python -I harness_zoo_policy.py \
  --solution solution/high_rate_policy.py \
  --surface-name high_rate_policy \
  --mode quality \
  --data-root /data/compressai-zoo/kodak \
  --checkpoint-root /data/compressai-zoo/checkpoints \
  --protocol /data/compressai-zoo/protocol.json \
  --protocol-sha256 4b84d6ac0f8af07206b674824608ddbf1ff6e05037f363048521c5869bc525c9
