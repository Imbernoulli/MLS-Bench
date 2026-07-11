#!/bin/bash
# simp-source-policy: rewrite each of the THREE FIXED simplification test settings
# (asset / turk / wiki) by the agent's SOURCE POLICY (solution/policy.py ->
# build_policy), then score corpus SARI per setting (higher is better).
set -euo pipefail
trap 'rc=$?; if [[ ${rc} -ne 0 ]]; then printf "VERIFICATION_FAILED text-simplification rc=%s\\n" "${rc}" >&2; fi' EXIT
cd /workspace/text-simplification

python harness_policy.py \
    --solution solution/policy.py \
    --seed "${SEED:-42}"
