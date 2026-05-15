#!/bin/bash
set -e
cd /workspace/easy-few-shot-learning

# Create symlinks for CIFAR-FS images and specs
mkdir -p data/cifar_fs
ln -sfn /data/cifar_fs/images data/cifar_fs/images 2>/dev/null || true
# Symlink JSON specs from data directory if they exist there
for spec in train.json val.json test.json; do
    [ -f "/data/cifar_fs/$spec" ] && ln -sfn "/data/cifar_fs/$spec" "data/cifar_fs/$spec" 2>/dev/null || true
done

ENV=cifar_fs SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u custom_fewshot.py
