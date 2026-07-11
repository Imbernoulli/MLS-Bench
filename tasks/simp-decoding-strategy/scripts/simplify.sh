#!/bin/bash
# simp-decoding-strategy: simplify each of the THREE FIXED test settings (asset /
# turk / wiki) with a FROZEN T5-base simplifier using the agent's DECODING STRATEGY
# (solution/strategy.py -> build_strategy: "sample" / "topp" / "beam"), then score
# corpus SARI per setting (higher is better).
set -euo pipefail
trap 'rc=$?; if [[ ${rc} -ne 0 ]]; then printf "VERIFICATION_FAILED text-simplification rc=%s\\n" "${rc}" >&2; fi' EXIT
cd /workspace/text-simplification

python harness_strategy.py \
    --solution solution/strategy.py \
    --seed "${SEED:-42}"
