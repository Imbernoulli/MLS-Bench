#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ID=${1:?usage: create_mt_strict_layer.sh RUN_ID STAGE_ROOT}
STAGE_ROOT=${2:?usage: create_mt_strict_layer.sh RUN_ID STAGE_ROOT}
BUILD="${STAGE_ROOT}/layer-files-${RUN_ID}"

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid MT layer run id: ${RUN_ID}" >&2
    exit 2
fi
for path in \
    "${STAGE_ROOT}/scripts/write_mt_manifest_layer.py" \
    "${STAGE_ROOT}/vendor/machine-translation/common.py"; do
    if [[ ! -f "${path}" ]]; then
        echo "missing MT layer input: ${path}" >&2
        exit 66
    fi
done
if [[ -e "${BUILD}" ]]; then
    echo "refusing to reuse MT layer directory: ${BUILD}" >&2
    exit 73
fi
mkdir -p "${BUILD}/rootfs/data/machine-translation/models/opus-mt-de-en"
mkdir -p "${BUILD}/rootfs/data/machine-translation/models/opus-mt-fr-en"
mkdir -p "${BUILD}/rootfs/data/machine-translation/models/opus-mt-ru-en"
exec > >(tee -a "${BUILD}/build.log") 2>&1

finish() {
    rc=$?
    trap - EXIT HUP INT TERM
    printf '%s\n' "${rc}" > "${BUILD}/rc"
    if [[ ${rc} -eq 0 ]]; then
        printf 'success\n' > "${BUILD}/status"
        date -Iseconds > "${BUILD}/SUCCESS"
    else
        printf 'failed\n' > "${BUILD}/status"
    fi
    exit "${rc}"
}
trap finish EXIT
printf 'running\n' > "${BUILD}/status"
date -Iseconds > "${BUILD}/STARTED"

for model_dir in opus-mt-de-en opus-mt-fr-en opus-mt-ru-en; do
    touch "${BUILD}/rootfs/data/machine-translation/models/${model_dir}/.wh..cache"
    touch "${BUILD}/rootfs/data/machine-translation/models/${model_dir}/.wh..gitattributes"
    touch "${BUILD}/rootfs/data/machine-translation/models/${model_dir}/.wh.README.md"
done
touch "${BUILD}/rootfs/data/machine-translation/models/opus-mt-fr-en/.wh.pytorch_model.bin"

python "${STAGE_ROOT}/scripts/write_mt_manifest_layer.py" \
    --common "${STAGE_ROOT}/vendor/machine-translation/common.py" \
    --layer-root "${BUILD}/rootfs"
touch "${BUILD}/rootfs/data/machine-translation/data/.wh.flores_de_en_test.jsonl"
tar --numeric-owner --owner=0 --group=0 -czf "${BUILD}/strict-layer.tar.gz" \
    -C "${BUILD}/rootfs" .
tar -tzf "${BUILD}/strict-layer.tar.gz" > "${BUILD}/layer-inventory.txt"
sha256sum "${BUILD}/strict-layer.tar.gz" > "${BUILD}/strict-layer.sha256"
date -Iseconds > "${BUILD}/LAYER_READY"
printf 'MT_STRICT_LAYER_FILES_READY path=%s\n' "${BUILD}/strict-layer.tar.gz"
