#!/bin/bash
# prune-reinit: prune ResNet-18 on REAL CIFAR-10 (--surface reinit),
# enforced budget, 160-epoch recovery; label=cifar10_seed1 seed=1.
set -euo pipefail
cd /workspace/prune-lab
python harness.py \
    --data-root ${PRUNE_DATA_ROOT:-/data/prune-lab/cifar} \
    --dense-ckpt ${PRUNE_DENSE_CKPT:-/data/prune-lab/dense_resnet18_cifar10.pt} \
    --manifest ${PRUNE_MANIFEST:-/data/prune-lab/manifest.json} \
    --task-id prune-reinit \
    --surface reinit \
    --solution solution/reinit.py \
    --label cifar10_seed1 \
    --sparsity 0.9 \
    --recovery-epochs 160 \
    --seed 1
