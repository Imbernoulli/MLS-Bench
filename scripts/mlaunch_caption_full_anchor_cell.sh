#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "usage: $0 RUN_ID {sample|greedy|beam} ZONE IMAGE [PRIORITY]" >&2
    exit 2
fi
RUN_ID=$1
BASELINE=$2
ZONE=$3
IMAGE=$4
PRIORITY=${5:-599}
STAGE_ROOT=${CAPTION_STAGE:-/mnt/moonfs/lvbohan-b0/image-captioning-full-v1}
STAGE_MOUNT_ROOT=${CAPTION_STAGE_MOUNT_ROOT:-$(dirname "${STAGE_ROOT}")}
RC_GROUP=${CAPTION_RC_GROUP:-}
RUN="${STAGE_ROOT}/anchors/${RUN_ID}/${BASELINE}"

if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "invalid caption anchor RUN_ID: ${RUN_ID}" >&2
    exit 2
fi
case "${BASELINE}" in
    sample|greedy|beam) ;;
    *) echo "invalid caption decoding baseline: ${BASELINE}" >&2; exit 2 ;;
esac
if [[ ! "${IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "caption anchor image must be pinned by digest: ${IMAGE}" >&2
    exit 2
fi
if [[ ! -d "${STAGE_MOUNT_ROOT}" ]]; then
    echo "caption anchor stage mount root does not exist: ${STAGE_MOUNT_ROOT}" >&2
    exit 2
fi
if [[ -n "${RC_GROUP}" && ! "${RC_GROUP}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "invalid caption anchor RC group: ${RC_GROUP}" >&2
    exit 2
fi
if [[ "${PRIORITY}" != "preemptible" && ! "${PRIORITY}" =~ ^([0-9]+|High|Medium|Low)$ ]]; then
    echo "invalid caption anchor priority mode: ${PRIORITY}" >&2
    exit 2
fi
if [[ -e "${RUN}" ]]; then
    echo "refusing to reuse caption anchor cell: ${RUN}" >&2
    exit 2
fi
mkdir -p "$(dirname "${RUN}")"
mkdir "${RUN}"
printf 'queued\n' > "${RUN}/status"
printf '125\n' > "${RUN}/rc"
printf '%s\n' \
    "task=caption-decoding-strategy" \
    "baseline=${BASELINE}" \
    "zone=${ZONE}" \
    "rc_group=${RC_GROUP:-default}" \
    "priority_mode=${PRIORITY}" \
    "gpu_count=1" \
    "cell_uses_single_gpu=true" \
    "protocol=flickr8k_official_v1" \
    "train_images=6000" \
    "train_pairs=30000" \
    "eval_images=1000" \
    "epochs=10" \
    "batch_size=40" \
    "optimizer_steps=7500" \
    "runtime_install=false" \
    "runtime_download=false" \
    "stage_mount_root=${STAGE_MOUNT_ROOT}" \
    "image=${IMAGE}" \
    > "${RUN}/launch-request.txt"
date -Iseconds > "${RUN}/SUBMITTED"

rc_group_args=()
if [[ -n "${RC_GROUP}" ]]; then
    rc_group_args=(-g "${RC_GROUP}")
fi
priority_args=(--priority "${PRIORITY}")
if [[ "${PRIORITY}" == "preemptible" ]]; then
    priority_args=(--preemptible=yes)
fi

set +e
output=$(mlaunch -d \
    "${rc_group_args[@]}" \
    -z "${ZONE}" \
    --gpu=1 \
    "${priority_args[@]}" \
    --preemption-policy-never=false \
    --max-wait-duration 48h \
    --max-idle-duration 12h \
    --i-know-i-am-wasting-resource \
    --enable-sshd=false \
    --comment "caption full v3 ${BASELINE} 6000x5 10ep 7500step single GPU" \
    --image "${IMAGE}" \
    --image-pull-policy IfNotPresent \
    --volume "${STAGE_MOUNT_ROOT}:${STAGE_MOUNT_ROOT}" \
    -w "${RUN}" \
    -- bash -lc "
set +e
cd '${RUN}' || exit 111
date -Iseconds > STARTED
printf 'running\\n' > status
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python - <<'PY'
import importlib.metadata as metadata
import torch

expected = {
    'transformers': '4.53.2',
    'pycocoevalcap': '1.2',
    'jdk4py': '17.0.9.2',
}
observed = {name: metadata.version(name) for name in expected}
assert observed == expected, observed
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print('CAPTION_ANCHOR_RUNTIME', observed, torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY
runtime_rc=\$?
if [[ \${runtime_rc} -eq 0 ]]; then
    python /opt/mlsbench-caption/repo/scripts/run_caption_full_protocol_anchor.py \\
        --root /opt/mlsbench-caption/repo \\
        --data-root /data \\
        --task caption-decoding-strategy \\
        --baseline ${BASELINE} \\
        --output \"\$PWD/output\" \\
        > worker.log 2>&1
    worker_rc=\$?
else
    worker_rc=\${runtime_rc}
fi
printf '%s\\n' \"\${worker_rc}\" > rc
if [[ \${worker_rc} -eq 0 ]]; then
    printf 'success\\n' > status
    date -Iseconds > SUCCESS
else
    printf 'failed\\n' > status
fi
exit \${worker_rc}
" 2>&1)
launch_rc=$?
set -e

printf '%s\n' "${output}" > "${RUN}/mlaunch.log"
if [[ ${launch_rc} -ne 0 ]]; then
    printf '%s\n' "${launch_rc}" > "${RUN}/rc"
    printf 'launch_failed\n' > "${RUN}/status"
    date -Iseconds > "${RUN}/LAUNCH_FAILED"
    echo "CAPTION_ANCHOR_LAUNCH_FAILED baseline=${BASELINE} rc=${launch_rc}" >&2
    exit "${launch_rc}"
fi
worker=$(printf '%s\n' "${output}" | tail -n 1)
printf '%s\n' "${worker}" > "${RUN}/worker.name"
echo "CAPTION_ANCHOR_SUBMITTED baseline=${BASELINE} worker=${worker} zone=${ZONE} gpu=1 run=${RUN}"
