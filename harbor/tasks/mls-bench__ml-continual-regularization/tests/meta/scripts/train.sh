#!/bin/bash
set -e

SEED=${SEED:-42}
export OUTPUT_DIR=${OUTPUT_DIR:-./results}
mkdir -p "$OUTPUT_DIR"

echo "=== Running Split-MNIST (5 contexts, task-incremental) ==="

python main.py \
    --experiment splitMNIST \
    --scenario task \
    --contexts 5 \
    --weight-penalty \
    --importance-weighting fisher \
    --reg-strength 5000 \
    --iters 2000 \
    --seed "$SEED" \
    --results-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train_output.log"

echo ""
echo "=== Extracting metrics ==="

python -c "
import re, sys, os

log = open(os.environ['OUTPUT_DIR'] + '/train_output.log').read()

# Parse average accuracy: '=> average accuracy over all N contexts: X.XXXX'
avg_match = re.search(r'average accuracy over all \d+ contexts: ([\d.]+)', log)
avg_acc = avg_match.group(1) if avg_match else None

# Parse per-context accuracies: '=> accuracy: X.XXX' (printed per context)
ctx_accs = re.findall(r'=> accuracy: ([\d.]+)', log)

# TRAIN_METRICS: per-context detail
for i, acc in enumerate(ctx_accs):
    print(f'TRAIN_METRICS: context={i+1} accuracy={acc}', flush=True)

# TEST_METRICS: average accuracy (the main metric)
if avg_acc:
    print(f'TEST_METRICS: average_accuracy={avg_acc}', flush=True)
else:
    # Fallback: read from results file
    import glob, os
    acc_files = sorted(glob.glob(os.path.join('$OUTPUT_DIR', 'acc-*.txt')))
    if acc_files:
        with open(acc_files[-1]) as f:
            avg_acc = f.readline().strip()
        print(f'TEST_METRICS: average_accuracy={avg_acc}', flush=True)
    else:
        print('ERROR: No average accuracy found', file=sys.stderr)
        sys.exit(1)
"
