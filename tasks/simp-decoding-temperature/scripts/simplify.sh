#!/bin/bash
# simp-decoding-temperature: simplify each of the THREE FIXED test settings (asset /
# turk / wiki) with a FROZEN T5-base simplifier using SAMPLING (do_sample=True,
# num_beams=1 FIXED) at the agent's TEMPERATURE (solution/temperature.py ->
# build_temperature), then score corpus SARI per setting (higher is better).
set -euo pipefail
trap 'rc=$?; if [[ ${rc} -ne 0 ]]; then printf "VERIFICATION_FAILED text-simplification rc=%s\\n" "${rc}" >&2; fi' EXIT
cd /workspace/text-simplification

python harness_temperature.py \
    --solution solution/temperature.py \
    --seed "${SEED:-42}"
