#!/usr/bin/env bash
set -euo pipefail

: "${MT_NUM_BEAMS:?MT_NUM_BEAMS is required}"
: "${MT_NOREP:?MT_NOREP is required}"
case "${MT_NUM_BEAMS}:${MT_NOREP}" in
    1:0|5:0|5:3) ;;
    *) echo "unsupported MT anchor cell ${MT_NUM_BEAMS}:${MT_NOREP}" >&2; exit 2 ;;
esac

solution="/tmp/mt_beam_${MT_NUM_BEAMS}_${MT_NOREP}.py"
printf 'def build_beam_config():\n    return {"num_beams": %s, "no_repeat_ngram_size": %s}\n' \
    "${MT_NUM_BEAMS}" "${MT_NOREP}" > "${solution}"

echo "MT_ANCHOR_START beams=${MT_NUM_BEAMS} norep=${MT_NOREP} timestamp=$(date -Is) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /workspace/machine-translation
for direction in de_en fr_en ru_en; do
    export MT_DIR="${direction}"
    echo "MT_SETTING_START direction=${direction} beams=${MT_NUM_BEAMS} norep=${MT_NOREP}"
    python harness_beam.py --solution "${solution}" --seed 42
    echo "MT_SETTING_DONE direction=${direction} beams=${MT_NUM_BEAMS} norep=${MT_NOREP}"
done
echo "MT_ANCHOR_DONE beams=${MT_NUM_BEAMS} norep=${MT_NOREP} timestamp=$(date -Is)"
