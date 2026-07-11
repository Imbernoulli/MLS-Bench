#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 RUN_ID FAMILY PROOF" >&2
    exit 64
fi

RUN_ID=$1
FAMILY=$2
PROOF=$3
STAGE=${COMPRESS_STAGE:-/stage}
RUN=${STAGE}/validation/${RUN_ID}

test -d "${RUN}"
cd "${RUN}" || exit 111
mkdir .worker-claim
exec > >(tee -a "${RUN}/worker.log") 2>&1

finish() {
    rc=$?
    trap - EXIT HUP INT TERM
    printf '%s\n' "${rc}" > "${RUN}/rc"
    date -Iseconds > "${RUN}/FINISHED"
    if [[ ${rc} -eq 0 ]] && [[ -s ${RUN}/validation.json ]]; then
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
sha256sum -c entry.sha256
/opt/conda/bin/python - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print("COMPRESS_REPLAY_GPU", torch.cuda.get_device_name(0))
PY
/opt/conda/bin/python -I "${RUN}/validate_anchor.py" \
    --parser "${RUN}/parser.py" \
    --proof "${PROOF}" \
    --family "${FAMILY}" \
    --output "${RUN}/validation.json"
