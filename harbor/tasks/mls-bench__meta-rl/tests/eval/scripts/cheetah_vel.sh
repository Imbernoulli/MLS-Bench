#!/bin/bash
set -e
cd "${MLSBENCH_PKG_DIR:-.}"
python launch_custom.py --env cheetah-vel --gpu 0 --seed ${SEED:-42}
