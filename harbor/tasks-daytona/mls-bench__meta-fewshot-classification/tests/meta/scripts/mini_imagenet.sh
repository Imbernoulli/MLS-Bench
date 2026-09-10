#!/bin/bash
set -e
cd /workspace/easy-few-shot-learning

# Create symlinks so easyfsl specs can find images
# miniImageNet: CSV root maps to data/mini_imagenet/<class_name>/<image.JPEG>
# `ln -sfn` is unlink()+symlink() — NOT atomic — and the concurrent
# (label, seed) eval runs share this workspace while dereferencing this
# link for every image open, so re-linking while a sibling reads opens a
# transient-ENOENT window. Therefore: (a) skip the ln entirely when the link
# already points at the right target, and (b) serialize first-run creation
# under an inter-process flock on a workspace lockfile.
setup_data_links() {
    mkdir -p data/mini_imagenet
    [ "$(readlink data/mini_imagenet/images 2>/dev/null)" = "/data/mini_imagenet/images" ] \
        || ln -sfn /data/mini_imagenet/images data/mini_imagenet/images 2>/dev/null || true
    # Also symlink the flat class dirs at the root (MiniImageNet uses root/<class>/<img>)
}
if command -v flock >/dev/null 2>&1; then
    ( flock 9 && setup_data_links ) 9>>".mlsbench_data_links.lock"
else
    setup_data_links
fi

ENV=mini_imagenet SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u custom_fewshot.py
