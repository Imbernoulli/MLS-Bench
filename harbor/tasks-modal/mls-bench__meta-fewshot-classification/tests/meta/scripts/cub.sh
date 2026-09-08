#!/bin/bash
set -e
cd /workspace/easy-few-shot-learning

# Create symlinks for CUB images and specs.
# `ln -sfn` is unlink()+symlink() — NOT atomic — and the concurrent
# (label, seed) eval runs share this workspace while dereferencing these
# links for every image open, so re-linking while a sibling reads opens a
# transient-ENOENT window. Therefore: (a) skip the ln entirely when the link
# already points at the right target, and (b) serialize first-run creation
# under an inter-process flock on a workspace lockfile.
setup_data_links() {
    mkdir -p data/CUB
    [ "$(readlink data/CUB/images 2>/dev/null)" = "/data/CUB/images" ] \
        || ln -sfn /data/CUB/images data/CUB/images 2>/dev/null || true
    # Symlink JSON specs from data directory if they exist there
    for spec in train.json val.json test.json; do
        [ -f "/data/CUB/$spec" ] || continue
        [ "$(readlink "data/CUB/$spec" 2>/dev/null)" = "/data/CUB/$spec" ] \
            || ln -sfn "/data/CUB/$spec" "data/CUB/$spec" 2>/dev/null || true
    done
}
if command -v flock >/dev/null 2>&1; then
    ( flock 9 && setup_data_links ) 9>>".mlsbench_data_links.lock"
else
    setup_data_links
fi

ENV=CUB SEED=${SEED:-42} OUTPUT_DIR=${OUTPUT_DIR:-./output} \
    python -u custom_fewshot.py
