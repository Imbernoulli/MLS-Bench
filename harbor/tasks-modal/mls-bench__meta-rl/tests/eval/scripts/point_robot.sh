#!/bin/bash
set -e
cd "${MLSBENCH_PKG_DIR:-.}"
# Retrofit the concurrent-seed log-dir fix onto stale image-baked launchers.
source "$(dirname "${BASH_SOURCE[0]}")/_runtime_patch.sh"
python launch_custom.py --env point-robot --gpu 0 --seed ${SEED:-42}
