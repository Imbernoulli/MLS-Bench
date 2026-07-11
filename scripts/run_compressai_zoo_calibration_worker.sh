#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    exit 64
fi

RUN_ID=$1
STAGE=${COMPRESS_STAGE:-/stage}
RUN=${STAGE}/calibration/${RUN_ID}

test -d "${RUN}"
cd "${RUN}" || exit 111
mkdir .worker-claim
exec > >(tee -a "${RUN}/worker.log") 2>&1

finish() {
    rc=$?
    trap - EXIT HUP INT TERM
    printf '%s\n' "${rc}" > "${RUN}/rc"
    date -Iseconds > "${RUN}/FINISHED"
    if [[ ${rc} -eq 0 ]] && [[ -s ${RUN}/output/files.sha256 ]]; then
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
print("COMPRESS_CALIBRATION_GPU", torch.cuda.get_device_name(0))
PY
/opt/conda/bin/python -I "${RUN}/calibrate.py" \
    --parser "${RUN}/parser.py" \
    --factorized /stage/anchors/zoo-anchor-factorized-20260711b-k1/proof.log \
    --hyperprior-scale /stage/anchors/zoo-anchor-hyperprior_scale-20260711b-k1/proof.log \
    --meanscale /stage/anchors/zoo-anchor-meanscale-20260711c-k1/proof.log \
    --output "${RUN}/output"
(cd "${RUN}/output" && sha256sum -c files.sha256)
