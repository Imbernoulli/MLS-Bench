#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 RUN_ID ZONE IMAGE [PRIORITY]" >&2
    exit 2
fi

RUN_ID=$1
ZONE=$2
IMAGE=$3
PRIORITY=${4:-599}
STAGE_ROOT=${CAPTION_RENDER_STAGE:-/home/lvbohan/image-captioning-anchor-v3}
OVERLAY=${CAPTION_RENDER_OVERLAY:-${STAGE_ROOT}/current-overlay-r8}
DATA_ROOT=${CAPTION_RENDER_DATA_ROOT:-/home/lvbohan/image-captioning-full-v1/data_root-v2}
MOUNT_ROOT=${CAPTION_RENDER_MOUNT_ROOT:-/home/lvbohan}
RUN=${STAGE_ROOT}/renders/${RUN_ID}
OUTPUT=${RUN}/output

if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "invalid Caption render RUN_ID: ${RUN_ID}" >&2
    exit 2
fi
if [[ ! "${ZONE}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "invalid Caption render zone: ${ZONE}" >&2
    exit 2
fi
if [[ ! "${IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "Caption render image must be pinned by digest: ${IMAGE}" >&2
    exit 2
fi
if [[ "${PRIORITY}" != "preemptible" && ! "${PRIORITY}" =~ ^([0-9]+|High|Medium|Low)$ ]]; then
    echo "invalid Caption render priority: ${PRIORITY}" >&2
    exit 2
fi
for path in "${OVERLAY}" "${DATA_ROOT}" "${MOUNT_ROOT}"; do
    if [[ ! -d "${path}" ]]; then
        echo "required Caption render directory does not exist: ${path}" >&2
        exit 2
    fi
done
if [[ -e "${RUN}" ]]; then
    echo "refusing to reuse Caption render directory: ${RUN}" >&2
    exit 2
fi

mkdir -p "$(dirname "${RUN}")"
mkdir "${RUN}"
printf 'queued\n' > "${RUN}/status"
printf '125\n' > "${RUN}/rc"
printf '%s\n' \
    "zone=${ZONE}" \
    "gpu_count=1" \
    "gpu_backend=h20" \
    "h20_serial=true" \
    "runtime_install=false" \
    "runtime_download=false" \
    "overlay=${OVERLAY}" \
    "data_root=${DATA_ROOT}" \
    "output=${OUTPUT}" \
    "image=${IMAGE}" \
    > "${RUN}/launch-request.txt"
date -Iseconds > "${RUN}/SUBMITTED"

priority_args=(--priority "${PRIORITY}")
if [[ "${PRIORITY}" == "preemptible" ]]; then
    priority_args=(--preemptible=yes)
fi

set +e
launch_output=$(mlaunch -d \
    -z "${ZONE}" \
    --gpu=1 \
    "${priority_args[@]}" \
    --preemption-policy-never=false \
    --max-wait-duration 48h \
    --max-idle-duration 12h \
    --i-know-i-am-wasting-resource \
    --enable-sshd=false \
    --comment "Caption fresh fail-closed render: 10 siblings" \
    --image "${IMAGE}" \
    --image-pull-policy IfNotPresent \
    --volume "${MOUNT_ROOT}:${MOUNT_ROOT}" \
    -w "${RUN}" \
    -- bash -lc "
set +e
cd '${RUN}' || exit 111
date -Iseconds > STARTED
printf 'running\\n' > status
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MLSBENCH_DATA_ROOT='${DATA_ROOT}'
export PYTHONPATH='${OVERLAY}/pytest_site:${OVERLAY}/render_site:${OVERLAY}/harbor_adapter/src:${OVERLAY}/src'
python -m pytest -q '${OVERLAY}/tests/test_image_captioning_fullscale_literal.py' \\
    > focused-tests.log 2>&1
test_rc=\$?
if [[ \${test_rc} -ne 0 ]]; then
    printf '%s\\n' \"\${test_rc}\" > rc
    printf 'failed\\n' > status
    exit \${test_rc}
fi
python -m pytest -q '${OVERLAY}/harbor_adapter/tests/test_score_task.py' \\
    -k 'run_evals_fails_closed_when_command_prints_metric_then_exits_nonzero or validate_eval_summary_rejects_standard_failure_markers or failure_marker_does_not_match_diagnostic_identifiers or validate_eval_summary_rejects_other_failure_markers or verifier_shell_nonzero_eval_rc_forces_exact_zero' \\
    > failclosed-tests.log 2>&1
test_rc=\$?
if [[ \${test_rc} -ne 0 ]]; then
    printf '%s\\n' \"\${test_rc}\" > rc
    printf 'failed\\n' > status
    exit \${test_rc}
fi
python -m mls_bench.main \\
    --mls-bench-root '${OVERLAY}' \\
    --output-dir '${OUTPUT}' \\
    --overwrite \\
    --mangrove \\
    --gpu-backend h20 \\
    --h20-serial \\
    --task-ids \\
        caption-decoding-strategy \\
        caption-visual-mapping \\
        caption-training-objective \\
        caption-feature-prep \\
        caption-mapping-init \\
        caption-train-sampling \\
        caption-optimizer \\
        caption-prompt-format \\
        caption-feature-augment \\
        caption-token-weighting \\
    > render.log 2>&1
worker_rc=\$?
if [[ \${worker_rc} -eq 0 ]]; then
    python '${OVERLAY}/scripts/audit_caption_render.py' \\
        --output '${OUTPUT}' \\
        --source '${OVERLAY}' \\
        --report audit-report.json \\
        > audit.log 2>&1
    worker_rc=\$?
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

printf '%s\n' "${launch_output}" > "${RUN}/mlaunch.log"
if [[ ${launch_rc} -ne 0 ]]; then
    printf '%s\n' "${launch_rc}" > "${RUN}/rc"
    printf 'launch_failed\n' > "${RUN}/status"
    date -Iseconds > "${RUN}/LAUNCH_FAILED"
    echo "CAPTION_RENDER_LAUNCH_FAILED rc=${launch_rc}" >&2
    exit "${launch_rc}"
fi

worker=$(printf '%s\n' "${launch_output}" | tail -n 1)
printf '%s\n' "${worker}" > "${RUN}/worker.name"
echo "CAPTION_RENDER_SUBMITTED worker=${worker} zone=${ZONE} gpu=1 run=${RUN}"
