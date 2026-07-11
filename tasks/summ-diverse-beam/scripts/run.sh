#!/bin/bash
# summ-diverse-beam: decode the THREE FIXED domain settings (xsum / cnndm / samsum) with the
# FROZEN domain-matched summarizers, using the agent's config
# (solution/diverse.py -> build_diverse_config), then score mean per-example ROUGE-L F1 per setting (higher is
# better; the task score gmean's the 3 settings).
set -euo pipefail
TASK_ID="summ-diverse-beam"
trap 'rc=$?; if (( rc != 0 )); then printf "VERIFICATION_FAILED task=%s rc=%d\n" "$TASK_ID" "$rc"; fi' EXIT
cd /workspace/abstractive-summarization || exit 111

python harness_diverse.py \
    --solution solution/diverse.py \
    --seed "${SEED:-42}"
