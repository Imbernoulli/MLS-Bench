#!/bin/bash
# simp-minlen-floor: simplify each of the THREE FIXED test settings (asset / turk /
# wiki) with a FROZEN T5-base simplifier under a FIXED beam width / length-penalty /
# max_length, varying ONLY the agent's decoder-side min-length FLOOR
# (solution/minlen.py -> build_min_length), then score corpus SARI per setting
# (higher is better).
set -euo pipefail
trap 'rc=$?; if [[ ${rc} -ne 0 ]]; then printf "VERIFICATION_FAILED text-simplification rc=%s\\n" "${rc}" >&2; fi' EXIT
cd /workspace/text-simplification

python harness_minlen.py \
    --solution solution/minlen.py \
    --seed "${SEED:-42}"
