#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 RUN_ID SOURCE_RUN_ID TARGET_IMAGE" >&2
    exit 64
fi

RUN_ID=$1
SOURCE_RUN_ID=$2
TARGET_IMAGE=$3
STAGE=${COMPRESS_STAGE:-/stage}
RUN=${STAGE}/builds/${RUN_ID}
SOURCE=${STAGE}/builds/${SOURCE_RUN_ID}
CRANE=/launchpad/crane

[[ ${RUN_ID} =~ ^[A-Za-z0-9._-]+$ ]] || exit 64
[[ ${SOURCE_RUN_ID} =~ ^[A-Za-z0-9._-]+$ ]] || exit 64
[[ ${TARGET_IMAGE} == *:* ]] || exit 64
test -d "${RUN}"
cd "${RUN}" || exit 111
if ! mkdir "${RUN}/.worker-claim"; then
    echo "refusing to reuse claimed CompressAI push run: ${RUN}" >&2
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
echo "COMPRESS_PUSH_WORKER host=$(hostname) run=${RUN_ID} source=${SOURCE_RUN_ID}"
test -x "${CRANE}"
test -s /root/.docker/config.json
test -s "${SOURCE}/FINISHED"
test "$(cat "${SOURCE}/rc")" = 1
test -s "${SOURCE}/compressai-zoo.layer.tar"
test -s "${SOURCE}/layer.sha256"
test -s "${SOURCE}/protocol.sha256"
test -s "${SOURCE}/base.ref"
test "$(grep -c '^COMPRESS_PREP_CHECKPOINT ' "${SOURCE}/prepare.log")" = 24
test "$(grep -c '^COMPRESS_PREP_COMPLETE ' "${SOURCE}/prepare.log")" = 1
sha256sum -c "${SOURCE}/layer.sha256"
cp "${SOURCE}/protocol.sha256" "${RUN}/protocol.sha256"
cp "${SOURCE}/layer.sha256" "${RUN}/layer.sha256"
cp "${SOURCE}/base.ref" "${RUN}/base.ref"

BASE_REF=$(cat "${SOURCE}/base.ref")
"${CRANE}" append \
    --base "${BASE_REF}" \
    --new_layer "${SOURCE}/compressai-zoo.layer.tar" \
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
