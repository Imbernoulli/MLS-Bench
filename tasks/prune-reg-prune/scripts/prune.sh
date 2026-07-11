#!/bin/bash
# prune-reg-prune: prune ResNet-18 on REAL CIFAR-10 (--surface reg_prune),
# enforced budget, 160-epoch recovery; label=cifar10 seed=42.
set -euo pipefail
cd /workspace/prune-lab
python harness.py \
    --data-root ${PRUNE_DATA_ROOT:-/data/prune-lab/cifar} \
    --dense-ckpt ${PRUNE_DENSE_CKPT:-/data/prune-lab/dense_resnet18_cifar10.pt} \
    --manifest ${PRUNE_MANIFEST:-/data/prune-lab/manifest.json} \
    --task-id prune-reg-prune \
    --surface reg_prune \
    --solution solution/reg_prune.py \
    --label cifar10 \
    --sparsity 0.9 \
    --recovery-epochs 160 \
    --seed 42
