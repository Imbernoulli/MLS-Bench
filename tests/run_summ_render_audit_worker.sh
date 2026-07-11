#!/bin/bash
set -euo pipefail

RUN=${1:?run directory required}
SRC=${2:?source root required}
ZONE=${3:?zone label required}

mkdir -p "$RUN"
date -Ins > "$RUN/STARTED"
cd "$RUN" || exit 111

set +e
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SRC/harbor_adapter/src" \
  python -m mls_bench.main \
  --output-dir "$RUN/rendered" \
  --overwrite \
  --mangrove \
  --gpu-backend h20 \
  --mls-bench-root "$SRC" \
  --task-ids \
    summ-beam-repetition \
    summ-beam-width \
    summ-decoding-length \
    summ-decoding-temperature \
    summ-diverse-beam \
    summ-norepeat-ngram \
    summ-nucleus-topp \
    summ-post-truncation \
    summ-sampling-vs-beam \
    summ-source-policy \
  > "$RUN/render.log" 2>&1
rc=$?

if [ "$rc" -eq 0 ]; then
  PYTHONDONTWRITEBYTECODE=1 python "$SRC/tests/audit_summ_rendered.py" \
    "$RUN/rendered" "$SRC" > "$RUN/audit.log" 2>&1
  rc=$?
fi

printf '%s\n' "$rc" > "$RUN/rc"
date -Ins > "$RUN/FINISHED"
if [ "$rc" -eq 0 ]; then
  touch "$RUN/SUCCESS"
  printf 'SUMM_RENDER_WORKER_SUCCESS zone=%s\n' "$ZONE" >> "$RUN/audit.log"
fi
exit "$rc"
