#!/bin/bash
# decode a FROZEN Qwen2.5-0.5B-Instruct under the agent's constrained-
# decoding policy (solution/decoder_choice_reasoning.py). Scores joint validity + correctness.
set -euo pipefail
: "${MLSBENCH_VERIFIER_DATA_ROOT:?MLSBENCH_VERIFIER_DATA_ROOT is required}"
: "${CD_MODEL:?CD_MODEL is required}"
export CD_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/constrained-decoding"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
printf '%s  %s\n' fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe "${CD_MODEL}/model.safetensors" | sha256sum -c -
printf '%s  %s\n' 33645f5c37148a6b05003c8fbbd8994b1c863a7642e9f1a52fda70dff6aa8a4e "${CD_DATA}/classification.json" | sha256sum -c -
cd /workspace/constrained-decoding-lab

python harness_choice.py \
    --solution solution/decoder_choice_reasoning.py \
    --task-id cd-choice-reasoning \
    --surface decoder_choice_reasoning \
    --seed ${SEED:-42} \
    --n 7600
