#!/bin/bash
# prune-second-order: prune ResNet-18 on REAL CIFAR-10 (--surface second_order),
# enforced budget, 160-epoch recovery; label=cifar10 seed=42.
set -euo pipefail
cd /workspace/prune-lab
python harness.py \
    --data-root ${PRUNE_DATA_ROOT:-/data/prune-lab/cifar} \
    --dense-ckpt ${PRUNE_DENSE_CKPT:-/data/prune-lab/dense_resnet18_cifar10.pt} \
    --manifest ${PRUNE_MANIFEST:-/data/prune-lab/manifest.json} \
    --task-id prune-second-order \
    --surface second_order \
    --solution solution/second_order.py \
    --label cifar10 \
    --sparsity 0.9 \
    --recovery-epochs 160 \
    --seed 42
