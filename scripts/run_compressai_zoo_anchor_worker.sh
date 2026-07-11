#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 RUN_ID FAMILY" >&2
    exit 64
fi

RUN_ID=$1
FAMILY=$2
STAGE=${COMPRESS_STAGE:-/stage}
RUN=${STAGE}/anchors/${RUN_ID}
HARNESS=${COMPRESS_HARNESS:-${STAGE}/source/harness_zoo_entropy.py}
SOLUTION=${RUN}/entropy_model.py
DATA=/data/compressai-zoo

[[ ${RUN_ID} =~ ^[A-Za-z0-9._-]+$ ]] || exit 64
case "${FAMILY}" in
    factorized|hyperprior_scale|meanscale) ;;
    *) echo "unsupported family: ${FAMILY}" >&2; exit 64 ;;
esac
test -s "${HARNESS}"
test -s "${DATA}/protocol.json"
test -s "${DATA}/protocol.sha256"
test -d "${RUN}"
cd "${RUN}" || exit 111
if ! mkdir "${RUN}/.worker-claim"; then
    echo "refusing to reuse claimed CompressAI anchor run: ${RUN}" >&2
    exit 73
fi
exec > >(tee -a "${RUN}/worker.log") 2>&1

finish() {
    rc=$?
    trap - EXIT HUP INT TERM
    printf '%s\n' "${rc}" > "${RUN}/rc"
    date -Iseconds > "${RUN}/FINISHED"
    if [[ ${rc} -eq 0 ]] && [[ -s ${RUN}/proof.log ]] && [[ -s ${RUN}/proof.sha256 ]]; then
        date -Iseconds > "${RUN}/SUCCESS"
    else
        rm -f "${RUN}/SUCCESS"
    fi
    exit "${rc}"
}
trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

date -Iseconds > "${RUN}/STARTED"
echo "COMPRESS_ANCHOR_WORKER host=$(hostname) run=${RUN_ID} family=${FAMILY}"
sha256sum "$0" "${HARNESS}" > "${RUN}/source.sha256"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python - <<'PY'
import compressai, torch
assert str(compressai.__version__) == "1.2.8", compressai.__version__
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print("COMPRESS_RUNTIME", compressai.__version__, torch.__version__, torch.version.cuda)
PY

cat > "${SOLUTION}" <<PY
"""Pinned measured ${FAMILY} anchor surface."""
from __future__ import annotations


def entropy_model() -> str:
    return "${FAMILY}"
PY

protocol_sha=$(cut -d' ' -f1 "${DATA}/protocol.sha256")
[[ ${protocol_sha} =~ ^[0-9a-f]{64}$ ]]
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/nonexistent-network-cache

command_start_ns=$(/opt/conda/bin/python -c 'import time; print(time.time_ns())')
set +e
/opt/conda/bin/python -I "${HARNESS}" \
    --solution "${SOLUTION}" \
    --data-root "${DATA}/kodak" \
    --checkpoint-root "${DATA}/checkpoints" \
    --protocol "${DATA}/protocol.json" \
    --protocol-sha256 "${protocol_sha}" \
    2>&1 | tee "${RUN}/proof.log"
harness_rc=${PIPESTATUS[0]}
set -e
command_end_ns=$(/opt/conda/bin/python -c 'import time; print(time.time_ns())')
COMMAND_START_NS=${command_start_ns} COMMAND_END_NS=${command_end_ns} \
    /opt/conda/bin/python - <<'PY' > "${RUN}/command.time"
import os

start = int(os.environ["COMMAND_START_NS"])
end = int(os.environ["COMMAND_END_NS"])
if end <= start:
    raise SystemExit("invalid command timing interval")
print(f"ANCHOR_COMMAND start_ns={start} end_ns={end} elapsed={(end-start)/1e9:.9f}")
PY
printf '%s\n' "${harness_rc}" > "${RUN}/harness.rc"
[[ ${harness_rc} -eq 0 ]]
[[ $(grep -c '^COMPRESS_FINAL ' "${RUN}/proof.log") -eq 1 ]]
[[ $(grep -c '^COMPRESS_CASE ' "${RUN}/proof.log") -eq 192 ]]
[[ $(grep -c '^COMPRESS_MODEL ' "${RUN}/proof.log") -eq 8 ]]
[[ $(grep -c '^COMPRESS_RATE ' "${RUN}/proof.log") -eq 32 ]]
[[ $(grep -c '^COMPRESS_SETTING ' "${RUN}/proof.log") -eq 4 ]]
sha256sum "${RUN}/proof.log" > "${RUN}/proof.sha256"
echo "COMPRESS_ANCHOR_READY family=${FAMILY} proof_sha256=$(cut -d' ' -f1 "${RUN}/proof.sha256")"
