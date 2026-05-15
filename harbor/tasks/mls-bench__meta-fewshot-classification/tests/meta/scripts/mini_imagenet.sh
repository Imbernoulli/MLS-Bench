#!/bin/bash
set -e
cd /workspace/easy-few-shot-learning

# Create symlinks so easyfsl specs can find images
# miniImageNet: CSV root maps to data/mini_imagenet/<class_name>/<image.JPEG>
mkdir -p data/mini_imagenet
ln -sfn /data/mini_imagenet/images data/mini_imagenet/images 2>/dev/null || true
# Also symlink the flat class dirs at the root (MiniImageNet uses root/<class>/<img>)

ENV=mini_imagenet SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u custom_fewshot.py
