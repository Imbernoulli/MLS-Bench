#!/bin/bash
# Evaluate bandit policy on linear contextual bandit (K=5, d=10, T=10000)
#
# The fixed harness (SMPyBandits/mlsb_bandit_harness.py) owns the environment
# and runs BanditPolicy in a separate interpreter.  The only trusted result is
# the harness line stamped with a per-run nonce that the harness reads and
# deletes before any policy code starts; TEST_METRICS is printed here, after
# the harness has exited.
set -uo pipefail
cd /workspace

# A sourceless .pyc next to a module shadows it, and the edit-range guard does
# not hash .pyc files.  Remove any before the harness starts.
find SMPyBandits -maxdepth 1 -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find SMPyBandits -maxdepth 1 -name '*.pyc' -type f -delete 2>/dev/null

WORK="$(mktemp -d)"
chmod 700 "${WORK}"
head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "${WORK}/nonce"
NONCE="$(cat "${WORK}/nonce")"

TMPDIR="${WORK}" python -I -B SMPyBandits/mlsb_bandit_harness.py \
    --env contextual \
    --seed ${SEED:-42} \
    --nonce-file "${WORK}/nonce" \
    > "${WORK}/run.log" 2>&1
RC=$?

# Replay the harness log, minus anything that looks like a result line.
grep -a -v -e '^TEST_METRICS' -e '^HARNESS_RESULT' "${WORK}/run.log"
LINE="$(grep -a "^HARNESS_RESULT ${NONCE} " "${WORK}/run.log" | tail -1)"
rm -rf "${WORK}"

if [ "${RC}" -ne 0 ] || [ -z "${LINE}" ]; then
    echo "EVAL_INVALID env=contextual rc=${RC}: no trusted harness result" >&2
    exit 1
fi
echo "TEST_METRICS ${LINE#HARNESS_RESULT ${NONCE} }"
