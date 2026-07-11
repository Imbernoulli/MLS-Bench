#!/bin/bash
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "OOD_FAILURE task=ood-temperature rc=$rc"; fi' EXIT

cd /workspace/ood-detection-lab
DATA_ROOT="${OOD_DATA:-/data/ood-detection-lab}"
python harness.py \
    --task ood-temperature \
    --solution solution/temperature_score.py \
    --data "$DATA_ROOT/ood_full_eval_uint8.npz" \
    --checkpoint "$DATA_ROOT/openood_resnet18_cifar10_seed0.pt" \
    --seed "${SEED:-42}"
