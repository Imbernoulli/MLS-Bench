#!/bin/bash
# simp-decoding-beam: simplify each of the THREE FIXED test settings (asset / turk /
# wiki) with a FROZEN T5-base simplifier using the agent's BEAM / REPETITION decode
# config (solution/beam.py -> build_beam_config), then score corpus SARI per setting
# (higher is better).
set -euo pipefail
trap 'rc=$?; if [[ ${rc} -ne 0 ]]; then printf "VERIFICATION_FAILED text-simplification rc=%s\\n" "${rc}" >&2; fi' EXIT
cd /workspace/text-simplification

python harness_beam.py \
    --solution solution/beam.py \
    --seed "${SEED:-42}"
