#!/bin/bash
# simp-model-capacity: simplify each of the THREE FIXED test settings (asset /
# turk / wiki) with the agent's choice of FROZEN pretrained simplifier
# (solution/capacity.py -> build_model_choice), under a FIXED strong beam decode
# (num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0), then score corpus
# SARI per setting (higher is better).
set -euo pipefail
trap 'rc=$?; if [[ ${rc} -ne 0 ]]; then printf "VERIFICATION_FAILED text-simplification rc=%s\\n" "${rc}" >&2; fi' EXIT
cd /workspace/text-simplification

python harness_capacity.py \
    --solution solution/capacity.py \
    --seed "${SEED:-42}"
