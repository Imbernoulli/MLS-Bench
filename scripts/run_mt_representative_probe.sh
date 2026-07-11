#!/usr/bin/env bash
set -Eeuo pipefail

STAGE_ROOT=${1:?usage: run_mt_representative_probe.sh STAGE_ROOT OUTPUT_DIR}
OUTPUT_DIR=${2:?usage: run_mt_representative_probe.sh STAGE_ROOT OUTPUT_DIR}
HARNESS="${STAGE_ROOT}/vendor/machine-translation/harness_beam.py"
SOLUTION="${STAGE_ROOT}/vendor/machine-translation/solution/beam.py"

for path in "${HARNESS}" "${SOLUTION}"; do
    if [[ ! -f "${path}" ]]; then
        echo "missing representative probe input: ${path}" >&2
        exit 66
    fi
done
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "refusing to reuse representative output: ${OUTPUT_DIR}" >&2
    exit 73
fi
mkdir -p "${OUTPUT_DIR}"
printf 'running\n' > "${OUTPUT_DIR}/status"
date -Iseconds > "${OUTPUT_DIR}/STARTED"

finish() {
    rc=$?
    trap - EXIT HUP INT TERM
    printf '%s\n' "${rc}" > "${OUTPUT_DIR}/rc"
    if [[ ${rc} -eq 0 ]]; then
        printf 'success\n' > "${OUTPUT_DIR}/status"
        date -Iseconds > "${OUTPUT_DIR}/SUCCESS"
    else
        printf 'failed\n' > "${OUTPUT_DIR}/status"
    fi
    exit "${rc}"
}
trap finish EXIT

export MT_DATA=/data/machine-translation/data
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
for direction in de_en fr_en ru_en; do
    case "${direction}" in
        de_en) model_dir=opus-mt-de-en ;;
        fr_en) model_dir=opus-mt-fr-en ;;
        ru_en) model_dir=opus-mt-ru-en ;;
        *) exit 70 ;;
    esac
    export MT_DIR="${direction}"
    export MT_MODEL="/data/machine-translation/models/${model_dir}"
    started=$(date +%s)
    set +e
    python "${HARNESS}" --solution "${SOLUTION}" --seed 42 \
        > "${OUTPUT_DIR}/${direction}.log" 2>&1
    setting_rc=$?
    set -e
    ended=$(date +%s)
    printf '%s\n' "${setting_rc}" > "${OUTPUT_DIR}/${direction}.rc"
    printf '%s\n' "$((ended - started))" > "${OUTPUT_DIR}/${direction}.wall_seconds"
    if [[ ${setting_rc} -ne 0 ]]; then
        echo "representative direction failed: ${direction} rc=${setting_rc}" >&2
        exit "${setting_rc}"
    fi
    line_count=$(wc -l < "${OUTPUT_DIR}/${direction}.log")
    direction_count=$(grep -c "direction=${direction}" "${OUTPUT_DIR}/${direction}.log" || true)
    if [[ ${line_count} -ne 5 || ${direction_count} -ne 5 ]]; then
        echo "representative protocol mismatch: expected five ${direction} records, got lines=${line_count} directions=${direction_count}" >&2
        exit 70
    fi
done
printf 'MT_REPRESENTATIVE_COMPLETE task=mt-decoding-beam settings=3\n'
