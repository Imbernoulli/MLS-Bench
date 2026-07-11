#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    exit 64
fi

surface="$1"
solution="$2"

cd /workspace/image-inpainting
data_root="${MLSBENCH_VERIFIER_DATA_ROOT:?missing verifier data root}/data/image-inpainting/community_v1"
test -f "${data_root}/protocol_manifest.json"
test -f "${data_root}/train/manifest.json"
test -f "${data_root}/val/manifest.json"

python harness.py \
    --data-root "${data_root}" \
    --surface "${surface}" \
    --solution "${solution}" \
    --seed 42

printf 'INPAINT_VERIFICATION scope=full status=ok\n'
