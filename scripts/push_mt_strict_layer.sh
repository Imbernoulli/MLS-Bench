#!/usr/bin/env bash
set -Eeuo pipefail

LAYER_DIR=${1:?usage: push_mt_strict_layer.sh LAYER_DIR RUN_ID}
RUN_ID=${2:?usage: push_mt_strict_layer.sh LAYER_DIR RUN_ID}
BASE_IMAGE="msai-cn-beijing.cr.volces.com/public/bohanlyu2022/mlsbench-harbor-machine-translation@sha256:8dfc00ac296d6c5404e482af44ad862fb8a24c60b54029bd340c13a49076efba"
IMAGE_REPO="msai-cn-beijing.cr.volces.com/public/bohanlyu2022/mlsbench-harbor-machine-translation"
IMAGE="${IMAGE_REPO}:${RUN_ID}"
CRANE=/launchpad/crane

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid MT image run id: ${RUN_ID}" >&2
    exit 2
fi
for path in \
    "${LAYER_DIR}/LAYER_READY" \
    "${LAYER_DIR}/strict-layer.tar.gz" \
    "${LAYER_DIR}/strict-layer.sha256" \
    "${CRANE}"; do
    if [[ ! -s "${path}" ]]; then
        echo "missing MT push input: ${path}" >&2
        exit 66
    fi
done
exec > >(tee -a "${LAYER_DIR}/push.log") 2>&1

expected_layer_digest=$(cut -d' ' -f1 "${LAYER_DIR}/strict-layer.sha256")
actual_layer_digest=$(sha256sum "${LAYER_DIR}/strict-layer.tar.gz" | cut -d' ' -f1)
if [[ ! "${expected_layer_digest}" =~ ^[0-9a-f]{64}$ || \
      "${actual_layer_digest}" != "${expected_layer_digest}" ]]; then
    echo "MT strict layer digest mismatch" >&2
    exit 70
fi
base_digest=$("${CRANE}" digest "${BASE_IMAGE}")
if [[ "${base_digest}" != "sha256:8dfc00ac296d6c5404e482af44ad862fb8a24c60b54029bd340c13a49076efba" ]]; then
    echo "base image digest mismatch: ${base_digest}" >&2
    exit 70
fi
"${CRANE}" append \
    --base "${BASE_IMAGE}" \
    --new_layer "${LAYER_DIR}/strict-layer.tar.gz" \
    --new_tag "${IMAGE}" \
    --set-base-image-annotations
digest=$("${CRANE}" digest "${IMAGE}")
if [[ ! "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "failed to resolve immutable MT image digest: ${digest}" >&2
    exit 70
fi
printf '%s@%s\n' "${IMAGE_REPO}" "${digest}" > "${LAYER_DIR}/image.ref"
printf '%s\n' "${digest}" > "${LAYER_DIR}/image.digest"
date -Iseconds > "${LAYER_DIR}/IMAGE_READY"
printf 'MT_STRICT_IMAGE_PUSHED image=%s@%s\n' "${IMAGE_REPO}" "${digest}"
