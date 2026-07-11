#!/bin/bash
set -euo pipefail

DATA_ROOT=/data
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$DATA_ROOT/ood-detection-lab"
mkdir -p "$OUTPUT_DIR"

DATA="$OUTPUT_DIR/ood_full_eval_uint8.npz"
CHECKPOINT="$OUTPUT_DIR/openood_resnet18_cifar10_seed0.pt"
DATA_SHA256=796799c9a1c073784b02a3f42a00d0fc8b902387b1f3345b5b3d1f2631e0722d
CHECKPOINT_SHA256=8859e0ff484ab029fcd5bd0f85052c5679d82b8e36a7c066e922e2fdff62b7dc

if [[ ! -f "$DATA" ]]; then
    python "$SCRIPT_DIR/prepare_full_eval.py" --output "$DATA"
fi
test -f "$CHECKPOINT"
printf '%s  %s\n' "$DATA_SHA256" "$DATA" | sha256sum -c -
printf '%s  %s\n' "$CHECKPOINT_SHA256" "$CHECKPOINT" | sha256sum -c -
printf 'OOD_FULL_ASSETS_READY data=%s checkpoint=%s status=ok\n' "$DATA" "$CHECKPOINT"
