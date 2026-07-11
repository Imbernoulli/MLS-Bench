#!/usr/bin/env bash
set -uo pipefail

stage=/mnt/moonfs/lvbohan-b0/image-captioning-full-v1
repo="${stage}/repo"
data_root="${stage}/data_root-v2"
wave="${CAPTION_WAVE:-official-v1-wave1}"
run="${stage}/anchors/${wave}"

if ! mkdir "${run}"; then
    echo "refusing to double-write existing anchor wave: ${run}" >&2
    exit 2
fi
exec > >(tee "${run}/job.log") 2>&1

echo "CAPTION_WORKER_START host=$(hostname) date=$(date -Iseconds)"
printf '%s\n' "$(date -Iseconds)" > "${run}/STARTED"
(cd "${repo}" && sha256sum -c "${stage}/repo.sha256") \
    > "${run}/stage.sha256.log"
stage_rc=$?
printf '%s\n' "${stage_rc}" > "${run}/stage.sha256.rc"
if [[ ${stage_rc} -ne 0 ]]; then
    printf '%s\n' "${stage_rc}" > "${run}/wave.rc"
    exit "${stage_rc}"
fi
echo "CAPTION_STAGE_SHA_OK files=$(wc -l < "${stage}/repo.sha256")"
python - <<'PY' 2>&1 | tee "${run}/gpu_check.log"
import torch
count = torch.cuda.device_count()
print(f"CAPTION_TORCH version={torch.__version__} cuda={torch.version.cuda} devices={count}")
for index in range(count):
    print(f"CAPTION_GPU index={index} name={torch.cuda.get_device_name(index)}")
if not torch.cuda.is_available() or count != 8:
    raise SystemExit(f"caption wave requires exactly 8 visible CUDA devices, got {count}")
PY
gpu_rc=${PIPESTATUS[0]}
printf '%s\n' "${gpu_rc}" > "${run}/gpu_check.rc"
if [[ ${gpu_rc} -ne 0 ]]; then
    printf '%s\n' "${gpu_rc}" > "${run}/wave.rc"
    exit "${gpu_rc}"
fi

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="${stage}/hf-cache-v2"
export HUGGINGFACE_HUB_CACHE="${stage}/hf-cache-v2/hub"
export FLICKR8K_CANONICAL_ARCHIVE="${stage}/canonical/caption_datasets.zip"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${data_root}" "${HF_HOME}"

python -m pip install --no-cache-dir \
    --index-url https://mirrors.ivolces.com/pypi/simple/ \
    --extra-index-url https://pypi.msh.team/simple/ \
    'open_clip_torch==3.3.0' \
    'transformers==4.53.2' \
    'tokenizers>=0.21,<0.22' \
    'datasets==3.6.0' \
    'huggingface_hub>=0.30,<0.34' \
    'pycocoevalcap==1.2' \
    'jdk4py==17.0.9.2' \
    'protobuf<6' ftfy regex pillow \
    2>&1 | tee "${run}/bootstrap.log"
bootstrap_rc=${PIPESTATUS[0]}
printf '%s\n' "${bootstrap_rc}" > "${run}/bootstrap.rc"
if [[ ${bootstrap_rc} -ne 0 ]]; then
    printf '%s\n' "${bootstrap_rc}" > "${run}/wave.rc"
    exit "${bootstrap_rc}"
fi

exec 9>"${stage}/data-prepare.lock"
flock 9
CUDA_VISIBLE_DEVICES=0 python \
    "${repo}/vendor/data_scripts/image-captioning/prepare_data.py" \
    --data-root "${data_root}" \
    2>&1 | tee "${run}/prepare.worker.log"
prepare_rc=${PIPESTATUS[0]}
flock -u 9
printf '%s\n' "${prepare_rc}" > "${run}/prepare.rc"
if [[ ${prepare_rc} -ne 0 ]]; then
    printf '%s\n' "${prepare_rc}" > "${run}/wave.rc"
    exit "${prepare_rc}"
fi
sha256sum "${data_root}/image-captioning/source_manifest.json" \
    > "${run}/source_manifest.sha256"

matrix_profile="${CAPTION_MATRIX_PROFILE:-}"
if [[ -z "${matrix_profile}" ]]; then
    case "${wave}" in
        official-v1-wave1j-*) matrix_profile=initialization ;;
        official-v1-wave1k-*) matrix_profile=optimization ;;
        *) matrix_profile=core ;;
    esac
fi

case "${matrix_profile}" in
    decoding)
        # One complete single-GPU run per decoding strategy. The allocation may
        # be a whole node, but no caption cell ever spans multiple GPUs.
        matrix=(
            '0|caption-decoding-strategy|sample'
            '1|caption-decoding-strategy|greedy'
            '2|caption-decoding-strategy|beam'
        )
        ;;
    core)
        matrix=(
            '0|caption-decoding-strategy|sample'
            '1|caption-decoding-strategy|greedy'
            '2|caption-decoding-strategy|beam'
            '3|caption-visual-mapping|linear'
            '4|caption-visual-mapping|mlp'
            '5|caption-training-objective|ce'
            '6|caption-training-objective|labelsmooth'
            '7|caption-feature-prep|none'
        )
        ;;
    initialization)
        matrix=(
            '0|caption-decoding-strategy|sample'
            '1|caption-decoding-strategy|greedy'
            '2|caption-decoding-strategy|beam'
            '3|caption-feature-prep|l2'
            '4|caption-feature-prep|standardize'
            '5|caption-mapping-init|default'
            '6|caption-mapping-init|xavier'
            '7|caption-mapping-init|caption_mean'
        )
        ;;
    optimization)
        matrix=(
            '0|caption-decoding-strategy|sample'
            '1|caption-decoding-strategy|greedy'
            '2|caption-decoding-strategy|beam'
            '3|caption-train-sampling|uniform'
            '4|caption-train-sampling|bucketed'
            '5|caption-optimizer|sgd'
            '6|caption-optimizer|adamw'
            '7|caption-prompt-format|photo_prefix'
        )
        ;;
    *)
        echo "unknown CAPTION_MATRIX_PROFILE: ${matrix_profile}" >&2
        printf '2\n' > "${run}/wave.rc"
        exit 2
        ;;
esac
echo "CAPTION_MATRIX profile=${matrix_profile} cells=${#matrix[@]}"
cell_count=${#matrix[@]}

pids=()
keys=()
for entry in "${matrix[@]}"; do
    IFS='|' read -r gpu task baseline <<<"${entry}"
    key="${task}__${baseline}"
    keys+=("${key}")
    echo "CAPTION_CELL_START gpu=${gpu} key=${key}"
    (
        set +e
        CUDA_VISIBLE_DEVICES="${gpu}" python \
            "${repo}/scripts/run_caption_full_protocol_anchor.py" \
            --root "${repo}" \
            --data-root "${data_root}" \
            --task "${task}" \
            --baseline "${baseline}" \
            --output "${run}/${key}" \
            > "${run}/${key}.client.log" 2>&1
        rc=$?
        printf '%s\n' "${rc}" > "${run}/${key}.cell.rc"
        exit "${rc}"
    ) &
    pids+=("$!")
done

while :; do
    alive=0
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            alive=$((alive + 1))
        fi
    done
    echo "CAPTION_WAVE_PROGRESS alive=${alive} completed=$((cell_count - alive))/${cell_count} date=$(date -Iseconds)"
    [[ ${alive} -eq 0 ]] && break
    sleep 60
done

overall=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "CAPTION_CELL_DONE key=${keys[$index]} rc=0"
    else
        rc=$?
        echo "CAPTION_CELL_DONE key=${keys[$index]} rc=${rc}"
        overall=1
    fi
done
printf '%s\n' "${overall}" > "${run}/wave.rc"
echo "CAPTION_WORKER_DONE rc=${overall} date=$(date -Iseconds)"
exit "${overall}"
