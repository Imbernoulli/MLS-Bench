#!/bin/bash
# Meta-learning inner-loop optimizer evaluation: CIFAR-FS 5-way 5-shot
set -e

cd /workspace
# Retrofit the bookkeeping-cache flock fix onto stale image-baked scaffolds.
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"
ENV=cifar_fs_5shot SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u learn2learn/custom_maml.py
