#!/bin/bash
# simp-length-control: simplify each of the THREE FIXED test settings (asset / turk /
# wiki) with a FROZEN T5-base simplifier (beam 5 FIXED) using the agent's LENGTH /
# COMPRESSION decode config (solution/length.py -> build_length_config), then score
# corpus SARI per setting (higher is better).
set -euo pipefail
trap 'rc=$?; if [[ ${rc} -ne 0 ]]; then printf "VERIFICATION_FAILED text-simplification rc=%s\\n" "${rc}" >&2; fi' EXIT
cd /workspace/text-simplification

python harness_length.py \
    --solution solution/length.py \
    --seed "${SEED:-42}"
