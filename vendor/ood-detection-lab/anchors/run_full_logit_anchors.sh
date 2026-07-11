#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s DATA CHECKPOINT OUTPUT_DIR\n' "$0" >&2
    exit 2
fi

DATA="$1"
CHECKPOINT="$2"
OUTPUT_DIR="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUTPUT_DIR"
test -f "$DATA"
test -f "$CHECKPOINT"
test "$(python -c 'import torch; print(torch.cuda.device_count())')" = 1

for baseline in msp energy pseudo_cosine; do
    started=$(date +%s)
    python "$ROOT/harness_full_logit.py" \
        --solution "$ROOT/baselines/$baseline.py" \
        --data "$DATA" \
        --checkpoint "$CHECKPOINT" \
        --seed 42 \
        > "$OUTPUT_DIR/$baseline.log" 2>&1
    elapsed=$(( $(date +%s) - started ))
    printf 'OOD_ANCHOR baseline=%s wall_seconds=%s status=ok\n' "$baseline" "$elapsed" \
        | tee -a "$OUTPUT_DIR/$baseline.log"
done
