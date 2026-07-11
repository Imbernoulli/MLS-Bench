#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
    exit 64
fi

RUN_ID=$1
STAGE=${COMPRESS_STAGE:-/stage}
RUN=${STAGE}/anchors/${RUN_ID}
cd "${RUN}" || exit 111
sha256sum -c entry.sha256
export COMPRESS_HARNESS=${RUN}/harness_zoo_entropy.py
exec "${RUN}/anchor_worker.sh" "$@"
