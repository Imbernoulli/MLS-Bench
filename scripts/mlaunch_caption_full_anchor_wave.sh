#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 RUN_ID ZONE IMAGE [RC_GROUP]" >&2
    exit 2
fi

RUN_ID=$1
ZONE=$2
IMAGE=$3
RC_GROUP=${4:-alignment-joint}
PRIORITY=${CAPTION_WAVE_PRIORITY:-599}
WAVE_PROFILE=${CAPTION_WAVE_PROFILE:-decoding3}
STAGE_ROOT=${CAPTION_STAGE:-/mnt/moonfs/lvbohan-b0/image-captioning-full-v1}
STAGE_MOUNT_ROOT=${CAPTION_STAGE_MOUNT_ROOT:-$(dirname "${STAGE_ROOT}")}
RUN="${STAGE_ROOT}/anchor-waves/${RUN_ID}"

if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "invalid caption anchor wave RUN_ID: ${RUN_ID}" >&2
    exit 2
fi
if [[ ! "${ZONE}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "invalid caption anchor wave zone: ${ZONE}" >&2
    exit 2
fi
if [[ ! "${RC_GROUP}" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "invalid caption anchor wave RC group: ${RC_GROUP}" >&2
    exit 2
fi
if [[ "${PRIORITY}" != "preemptible" && ! "${PRIORITY}" =~ ^([0-9]+|High|Medium|Low)$ ]]; then
    echo "invalid caption anchor wave priority mode: ${PRIORITY}" >&2
    exit 2
fi
case "${WAVE_PROFILE}" in
    decoding3)
        ACTIVE_CELLS=3
        CELL_DECLARATION='caption-decoding-strategy/sample:0,caption-decoding-strategy/beam:1,caption-decoding-strategy/greedy:2'
        ;;
    sibling5)
        ACTIVE_CELLS=5
        CELL_DECLARATION='caption-visual-mapping/linear:0,caption-visual-mapping/mlp:1,caption-training-objective/ce:2,caption-training-objective/labelsmooth:3,caption-feature-prep/l2:4'
        ;;
    *)
        echo "invalid caption anchor wave profile: ${WAVE_PROFILE}" >&2
        exit 2
        ;;
esac
if [[ ! "${IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "caption anchor wave image must be pinned by digest: ${IMAGE}" >&2
    exit 2
fi
if [[ ! -d "${STAGE_MOUNT_ROOT}" ]]; then
    echo "caption anchor wave stage mount root does not exist: ${STAGE_MOUNT_ROOT}" >&2
    exit 2
fi
if [[ -e "${RUN}" ]]; then
    echo "refusing to reuse caption anchor wave: ${RUN}" >&2
    exit 2
fi

mkdir -p "$(dirname "${RUN}")"
mkdir "${RUN}"
printf 'queued\n' > "${RUN}/status"
printf '125\n' > "${RUN}/rc"
printf '%s\n' \
    "zone=${ZONE}" \
    "rc_group=${RC_GROUP}" \
    "priority_mode=${PRIORITY}" \
    "allocation_gpu_count=8" \
    "wave_profile=${WAVE_PROFILE}" \
    "active_cells=${ACTIVE_CELLS}" \
    "gpu_per_cell=1" \
    "cells=${CELL_DECLARATION}" \
    "protocol=flickr8k_official_v1" \
    "train_images_per_cell=6000" \
    "train_pairs_per_cell=30000" \
    "eval_images_per_cell=1000" \
    "epochs_per_cell=10" \
    "batch_size_per_cell=40" \
    "optimizer_steps_per_cell=7500" \
    "runtime_install=false" \
    "runtime_download=false" \
    "stage_mount_root=${STAGE_MOUNT_ROOT}" \
    "image=${IMAGE}" \
    > "${RUN}/launch-request.txt"
date -Iseconds > "${RUN}/SUBMITTED"

priority_args=(--priority "${PRIORITY}")
if [[ "${PRIORITY}" == "preemptible" ]]; then
    priority_args=(--preemptible=yes)
fi

set +e
output=$(mlaunch -d \
    -g "${RC_GROUP}" \
    -z "${ZONE}" \
    --gpu=8 \
    "${priority_args[@]}" \
    --preemption-policy-never=false \
    --max-wait-duration 48h \
    --max-idle-duration 12h \
    --i-know-i-am-wasting-resource \
    --enable-sshd=false \
    --comment "caption full v3 3x single-GPU 6000x5 10ep 7500step" \
    --image "${IMAGE}" \
    --image-pull-policy IfNotPresent \
    -e "CAPTION_RUN=${RUN}" \
    -e "CAPTION_WAVE_PROFILE=${WAVE_PROFILE}" \
    --volume "${STAGE_MOUNT_ROOT}:${STAGE_MOUNT_ROOT}" \
    -w "${RUN}" \
    -- bash -lc '
set +e
cd "${CAPTION_RUN}" || exit 111
date -Iseconds > STARTED
printf "running\n" > status
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python - <<"PY"
import importlib.metadata as metadata
import torch

expected = {
    "transformers": "4.53.2",
    "pycocoevalcap": "1.2",
    "jdk4py": "17.0.9.2",
}
observed = {name: metadata.version(name) for name in expected}
assert observed == expected, observed
assert torch.cuda.is_available() and torch.cuda.device_count() == 8
for index in range(8):
    print(f"CAPTION_WAVE_GPU index={index} name={torch.cuda.get_device_name(index)}", flush=True)
print("CAPTION_WAVE_RUNTIME", observed, torch.__version__, flush=True)
PY
runtime_rc=$?
if [[ ${runtime_rc} -ne 0 ]]; then
    printf "%s\n" "${runtime_rc}" > rc
    printf "failed\n" > status
    exit "${runtime_rc}"
fi

mkdir cells
pids=()
case "${CAPTION_WAVE_PROFILE}" in
    decoding3)
        tasks=(caption-decoding-strategy caption-decoding-strategy caption-decoding-strategy)
        baselines=(sample beam greedy)
        ;;
    sibling5)
        tasks=(caption-visual-mapping caption-visual-mapping caption-training-objective caption-training-objective caption-feature-prep)
        baselines=(linear mlp ce labelsmooth l2)
        ;;
    *)
        echo "invalid worker CAPTION_WAVE_PROFILE=${CAPTION_WAVE_PROFILE}" >&2
        exit 2
        ;;
esac
cell_count=${#tasks[@]}
for ((gpu=0; gpu<cell_count; gpu++)); do
    task=${tasks[$gpu]}
    baseline=${baselines[$gpu]}
    key=${task}__${baseline}
    mkdir "cells/${key}"
    printf "running\n" > "cells/${key}/status"
    (
        CUDA_VISIBLE_DEVICES=${gpu} python /opt/mlsbench-caption/repo/scripts/run_caption_full_protocol_anchor.py \
            --root /opt/mlsbench-caption/repo \
            --data-root /data \
            --task "${task}" \
            --baseline "${baseline}" \
            --output "$PWD/cells/${key}/output" \
            > "cells/${key}/worker.log" 2>&1
        cell_rc=$?
        printf "%s\n" "${cell_rc}" > "cells/${key}/rc"
        if [[ ${cell_rc} -eq 0 ]]; then
            printf "success\n" > "cells/${key}/status"
            date -Iseconds > "cells/${key}/SUCCESS"
        else
            printf "failed\n" > "cells/${key}/status"
        fi
        exit "${cell_rc}"
    ) &
    pids+=("$!")
done

overall=0
for ((index=0; index<cell_count; index++)); do
    if ! wait "${pids[$index]}"; then
        overall=1
    fi
done
printf "%s\n" "${overall}" > rc
if [[ ${overall} -eq 0 ]]; then
    printf "success\n" > status
    date -Iseconds > SUCCESS
else
    printf "failed\n" > status
fi
exit "${overall}"
' 2>&1)
launch_rc=$?
set -e

printf '%s\n' "${output}" > "${RUN}/mlaunch.log"
if [[ ${launch_rc} -ne 0 ]]; then
    printf '%s\n' "${launch_rc}" > "${RUN}/rc"
    printf 'launch_failed\n' > "${RUN}/status"
    date -Iseconds > "${RUN}/LAUNCH_FAILED"
    echo "CAPTION_ANCHOR_WAVE_LAUNCH_FAILED rc=${launch_rc}" >&2
    exit "${launch_rc}"
fi
worker=$(printf '%s\n' "${output}" | tail -n 1)
printf '%s\n' "${worker}" > "${RUN}/worker.name"
echo "CAPTION_ANCHOR_WAVE_SUBMITTED worker=${worker} zone=${ZONE} allocation_gpu=8 cell_gpu=1 run=${RUN}"
