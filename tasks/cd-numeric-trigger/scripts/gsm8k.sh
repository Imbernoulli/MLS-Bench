#!/bin/bash
# decode a FROZEN Qwen2.5-0.5B-Instruct under the agent's constrained-
# decoding policy (solution/decoder_trigger.py). Scores joint validity + correctness.
set -euo pipefail
: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
: "${CD_MODEL:?CD_MODEL is required}"
export CD_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/constrained-decoding"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
printf '%s  %s\n' fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe "${CD_MODEL}/model.safetensors" | sha256sum -c -
printf '%s  %s\n' 02ed017f9052a9e70777d01f388ba30153d04ccf5d06503e4d76a86005d8114e "${CD_DATA}/gsm8k.json" | sha256sum -c -
cd /workspace/constrained-decoding-lab

python harness_numeric.py \
    --solution solution/decoder_trigger.py \
    --seed ${SEED:-42} \
    --n 1319
