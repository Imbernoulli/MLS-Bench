#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 RUN_ID BASE_IMAGE TARGET_IMAGE" >&2
    exit 64
fi

RUN_ID=$1
BASE_IMAGE=$2
TARGET_IMAGE=$3
STAGE=${COMPRESS_STAGE:-/stage}
RUN=${STAGE}/builds/${RUN_ID}
PREP=${STAGE}/scripts/prepare_compressai_zoo_bundle.py
CRANE=/launchpad/crane

[[ ${RUN_ID} =~ ^[A-Za-z0-9._-]+$ ]] || exit 64
[[ ${TARGET_IMAGE} == *:* ]] || exit 64
test -x "${PREP}"
test -x "${CRANE}"
test -d "${RUN}"
cd "${RUN}" || exit 111
if ! mkdir "${RUN}/.worker-claim"; then
    echo "refusing to reuse claimed CompressAI image-build run: ${RUN}" >&2
    exit 73
fi
exec > >(tee -a "${RUN}/worker.log") 2>&1

finish() {
    rc=$?
    trap - EXIT HUP INT TERM
    printf '%s\n' "${rc}" > "${RUN}/rc"
    date -Iseconds > "${RUN}/FINISHED"
    if [[ ${rc} -eq 0 ]] && [[ -s ${RUN}/image.ref ]] && [[ -s ${RUN}/protocol.sha256 ]]; then
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
echo "COMPRESS_IMAGE_WORKER host=$(hostname) run=${RUN_ID}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || \
    echo "COMPRESS_IMAGE_WORKER gpu=none build_phase=cpu"
python - <<'PY'
import compressai, torch
assert str(compressai.__version__) == "1.2.8", compressai.__version__
print(
    "COMPRESS_RUNTIME",
    compressai.__version__,
    torch.__version__,
    torch.version.cuda,
    "cuda_available=", torch.cuda.is_available(),
    "devices=", torch.cuda.device_count(),
)
PY

export http_proxy=${http_proxy:-http://proxy.msh.work:3128}
export https_proxy=${https_proxy:-http://proxy.msh.work:3128}
export HTTP_PROXY=${HTTP_PROXY:-${http_proxy}}
export HTTPS_PROXY=${HTTPS_PROXY:-${https_proxy}}
export no_proxy=${no_proxy:-localhost,127.0.0.1,msh.team,msh.work,launchpad,svc,ksyun.cn,volces.com,aliyun.com,ksyuncs.com,ksyuncs.cn,aliyuncs.com}
export NO_PROXY=${NO_PROXY:-${no_proxy}}
echo "COMPRESS_IMAGE_WORKER network=proxy registry_bypass=enabled"
export TORCH_HOME="${STAGE}/cache/torch"
export PYTHONUNBUFFERED=1
mkdir -p "${TORCH_HOME}"

ROOT=${RUN}/root
DATA=${ROOT}/data/compressai-zoo
python "${PREP}" \
    --kodak-root /data/compressai/kodak \
    --output-root "${DATA}" \
    2>&1 | tee "${RUN}/prepare.log"
cp "${DATA}/protocol.sha256" "${RUN}/protocol.sha256"

base_digest=$("${CRANE}" digest "${BASE_IMAGE}")
[[ ${base_digest} =~ ^sha256:[0-9a-f]{64}$ ]]
printf '%s@%s\n' "${BASE_IMAGE%%@*}" "${base_digest}" > "${RUN}/base.ref"

LAYER=${RUN}/compressai-zoo.layer.tar
LC_ALL=C tar \
    --format=ustar \
    --sort=name \
    --mtime=@0 \
    --owner=0 \
    --group=0 \
    --numeric-owner \
    -C "${ROOT}" \
    -cf "${LAYER}" data/compressai-zoo
sha256sum "${LAYER}" > "${RUN}/layer.sha256"

"${CRANE}" append \
    --base "${BASE_IMAGE}@${base_digest}" \
    --new_layer "${LAYER}" \
    --new_tag "${TARGET_IMAGE}" \
    --set-base-image-annotations \
    2>&1 | tee "${RUN}/push.log"
image_digest=$("${CRANE}" digest "${TARGET_IMAGE}")
[[ ${image_digest} =~ ^sha256:[0-9a-f]{64}$ ]]
"${CRANE}" validate --remote "${TARGET_IMAGE}@${image_digest}" --fast
"${CRANE}" manifest "${TARGET_IMAGE}@${image_digest}" > "${RUN}/image.manifest.json"
"${CRANE}" config "${TARGET_IMAGE}@${image_digest}" > "${RUN}/image.config.json"
printf '%s\n' "${image_digest}" > "${RUN}/image.digest"
printf '%s@%s\n' "${TARGET_IMAGE%%:*}" "${image_digest}" > "${RUN}/image.ref"
echo "COMPRESS_IMAGE_READY image=$(cat "${RUN}/image.ref") protocol_sha256=$(cut -d' ' -f1 "${RUN}/protocol.sha256")"
