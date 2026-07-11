#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    exit 64
fi

RUN_ID=$1
SOURCE_RUN_ID=$2
TARGET_IMAGE=$3
STAGE=${COMPRESS_STAGE:-/stage}
SECRET_DIR=${STAGE}/secrets/${RUN_ID}
SECRET=${SECRET_DIR}/moongate.token
OWNER=${SECRET_DIR}/owner

[[ ${RUN_ID} =~ ^[A-Za-z0-9._-]+$ ]] || exit 64
test -f "${SECRET}"
test -f "${OWNER}"
test "$(stat -c %a "${SECRET}")" = 600 || exit 114
test "$(stat -c %u:%g "${SECRET}")" = "$(cat "${OWNER}")" || exit 115
IFS= read -r MOONGATE_ACCESS_TOKEN < "${SECRET}"
test -n "${MOONGATE_ACCESS_TOKEN}" || exit 116
: > "${SECRET}"
rm -f "${SECRET}" "${OWNER}"
rmdir "${SECRET_DIR}"
test ! -e "${SECRET_DIR}" || exit 117
export MOONGATE_ACCESS_TOKEN

cd "${STAGE}/builds/${RUN_ID}" || exit 111
exec "${STAGE}/scripts/run_compressai_zoo_push_worker.sh" \
    "${RUN_ID}" "${SOURCE_RUN_ID}" "${TARGET_IMAGE}"
