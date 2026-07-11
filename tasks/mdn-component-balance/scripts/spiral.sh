#!/bin/bash
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "VERIFICATION_FAILED task=mdn-component-balance rc=%s\\n" "$rc"; fi' EXIT
cd /workspace/mdn-density

python harness_mdn.py \
    --task mdn-component-balance \
    --solution solution/component_balance.py \
    --surface component_balance \
    --target spiral \
    --seed "${SEED:-42}" \
    --steps 4000 \
    --batch-size 512 \
    --lr 1e-3 \
    --n-train 20000 \
    --n-test 20000
