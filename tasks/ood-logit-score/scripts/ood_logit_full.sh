#!/bin/bash
set -euo pipefail
trap 'rc=$?; if [ "$rc" -ne 0 ]; then echo "OOD_FAILURE task=ood-logit-score rc=$rc"; fi' EXIT

cd /workspace/ood-detection-lab
DATA=/data/ood-detection-lab/ood_full_eval_uint8.npz
CHECKPOINT=/data/ood-detection-lab/openood_resnet18_cifar10_seed0.pt
test -f "$DATA"
test -f "$CHECKPOINT"

python harness_full_logit.py \
    --solution solution/logit_score.py \
    --data "$DATA" \
    --checkpoint "$CHECKPOINT" \
    --seed "${SEED:-42}"
