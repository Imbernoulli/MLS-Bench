#!/bin/bash
# summ-decoding-length: decode the THREE FIXED domain settings (xsum / cnndm / samsum) with the
# FROZEN domain-matched summarizers, using the agent's config
# (solution/length.py -> build_length_config), then score corpus ROUGE-L F1 per setting (higher is
# better; the task score gmean's the 3 settings).
set -euo pipefail
TASK_ID="summ-decoding-length"
trap 'rc=$?; if (( rc != 0 )); then printf "VERIFICATION_FAILED task=%s rc=%d\n" "$TASK_ID" "$rc"; fi' EXIT
cd /workspace/abstractive-summarization || exit 111

python harness_length.py \
    --solution solution/length.py \
    --seed "${SEED:-42}"
