#!/bin/bash
# mt-repetition-penalty [de_en]: translate the complete pinned de_en test split with a FROZEN OPUS-MT model
# using the agent's editable decode surface (solution/reppen.py), then score corpus
# sacreBLEU / chrF on the fixed English references (higher is better).
set -euo pipefail
: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
export MT_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/machine-translation/data"
export MT_DIR=de_en
cd /workspace/machine-translation

python harness_reppen.py \
    --solution solution/reppen.py \
    --seed ${SEED:-42}

echo "MT_SETTING_COMPLETE direction=${MT_DIR}"
