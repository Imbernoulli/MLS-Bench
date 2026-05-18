#!/bin/bash
# Compute Solvable Pass Rate (SoPR) post-hoc for agent-tool-reasoning.
#
# NEW (row-oriented) design:
#   Each leaderboard row may have answer_ts_{deepseek,qwen72b,qwen7b}
#   columns populated by train.sh via the parser. For every (row, setting)
#   pair that has a non-empty answer_ts, this script locates the answer
#   directory at
#     $SAVE_PATH/agent-tool-reasoning/<exp>/seed_<seed>/<setting>/<answer_ts>/G1_instruction/
#   (globbed across workspace dirs — the exp dir name is not required),
#   runs StableToolBench's judge via OpenRouter, and writes the resulting
#   sopr_<suffix> / sopr_n_scored_<suffix> back into that SAME row.
#
#   Because each test_cmd invocation uses a unique timestamped subdir, old
#   rounds are preserved; SoPR is computed against the exact files that
#   produced each row's pass_rate / avg_queries / give_up_rate metrics.
#
# CLI flags:
#   --save-path PATH       override SAVE_PATH (default: read from config yaml)
#   --eval-model MODEL     judge model on OpenRouter (default: meta-llama/llama-3.3-70b-instruct)
#   --evaluate-times N     number of judge runs per task (default: 1)
#   --max-eval-threads N   parallel judge calls (default: 4)
#   --row-filter PATTERN   restrict to leaderboard rows whose `model` column
#                          matches PATTERN (glob; repeatable; default: all rows)
#   --setting LABEL        restrict to one setting (repeatable; default: all 3)
#   --no-leaderboard       print summary but don't touch leaderboard.csv
#
# Env-var overrides: MLSBENCH_SAVE_PATH, EVAL_MODEL, EVALUATE_TIMES,
# MAX_EVAL_THREADS, OPENROUTER_API_KEY_NEW.
set -e

# ── argument parsing ─────────────────────────────────────────────────
SAVE_PATH=""
EVAL_MODEL=""
EVALUATE_TIMES=""
MAX_EVAL_THREADS=""
WRITE_LEADERBOARD=1
USER_FILTERS=()
USER_SETTINGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --save-path) SAVE_PATH="$2"; shift 2 ;;
        --eval-model) EVAL_MODEL="$2"; shift 2 ;;
        --evaluate-times) EVALUATE_TIMES="$2"; shift 2 ;;
        --max-eval-threads) MAX_EVAL_THREADS="$2"; shift 2 ;;
        --row-filter) USER_FILTERS+=("$2"); shift 2 ;;
        --setting) USER_SETTINGS+=("$2"); shift 2 ;;
        --no-leaderboard) WRITE_LEADERBOARD=0; shift ;;
        -h|--help)
            sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PKG_ROOT="${REPO_ROOT}/vendor/external_packages/stabletoolbench"
TASK_DIR="${REPO_ROOT}/tasks/agent-tool-reasoning"
WORK_DIR="${TASK_DIR}/.sopr_work"
OR_KEY_FILE="${TASK_DIR}/.openrouter_key"
LEADERBOARD="${TASK_DIR}/leaderboard.csv"
CONFIG_FILE="${MLSBENCH_CONFIG:-${REPO_ROOT}/configs/config.yaml}"

# ── resolve SAVE_PATH ───────────────────────────────────────────────
if [ -z "${SAVE_PATH}" ]; then
    SAVE_PATH="${MLSBENCH_SAVE_PATH:-}"
fi
if [ -z "${SAVE_PATH}" ] && [ -f "${CONFIG_FILE}" ]; then
    SAVE_PATH="$(python3 -c "
import yaml, sys
try:
    with open('${CONFIG_FILE}') as f:
        cfg = yaml.safe_load(f) or {}
    print(cfg.get('save_path', ''))
except Exception as e:
    print('', file=sys.stderr)
    print(f'WARN: could not parse ${CONFIG_FILE}: {e}', file=sys.stderr)
    print('')
")"
fi
if [ -z "${SAVE_PATH}" ]; then
    echo "ERROR: SAVE_PATH not provided. Use --save-path or set MLSBENCH_SAVE_PATH or ensure ${CONFIG_FILE} has 'save_path:'." >&2
    exit 1
fi

EVAL_MODEL="${EVAL_MODEL:-meta-llama/llama-3.3-70b-instruct}"
EVALUATE_TIMES="${EVALUATE_TIMES:-1}"
MAX_EVAL_THREADS="${MAX_EVAL_THREADS:-4}"

# ── resolve judge key (env preferred, fall back to file / config yaml) ──
if [ -z "${OPENROUTER_API_KEY_NEW:-}" ] && [ -f "${OR_KEY_FILE}" ]; then
    OPENROUTER_API_KEY_NEW="$(tr -d '\n' < "${OR_KEY_FILE}")"
fi
if [ -z "${OPENROUTER_API_KEY_NEW:-}" ] && [ -f "${CONFIG_FILE}" ]; then
    OPENROUTER_API_KEY_NEW="$(python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get('providers', {}).get('openrouter', {}).get('api_key', ''))
" 2>/dev/null)"
fi
if [ -z "${OPENROUTER_API_KEY_NEW:-}" ]; then
    echo "ERROR: judge key not found. Set OPENROUTER_API_KEY_NEW, create ${OR_KEY_FILE}, or ensure providers.openrouter.api_key is set in ${CONFIG_FILE}." >&2
    exit 1
fi

mkdir -p "${WORK_DIR}/converted" "${WORK_DIR}/results" "${WORK_DIR}/logs"
API_POOL_FILE="${WORK_DIR}/api_pool.json"
cat > "${API_POOL_FILE}" <<EOF
[{"api_key": "${OPENROUTER_API_KEY_NEW}", "api_base": "https://openrouter.ai/api/v1"}]
EOF
chmod 600 "${API_POOL_FILE}"

export PATH="${HOME}/miniconda3/condabin:${PATH}"
CONDA_RUN="conda run -n mlsbench-stabletoolbench --no-capture-output"

# ── build task list from leaderboard (row × setting) ────────────────
# Output TSV lines: row_idx<TAB>suffix<TAB>setting_label<TAB>answer_ts<TAB>model<TAB>seed<TAB>answer_dir
TASK_LIST_FILE="${WORK_DIR}/task_list.tsv"
python3 <<PY > "${TASK_LIST_FILE}"
import csv, glob, os, fnmatch

LEADERBOARD = "${LEADERBOARD}"
SAVE_PATH = "${SAVE_PATH}"
USER_FILTERS = """${USER_FILTERS[@]}""".split() if """${USER_FILTERS[@]}""".strip() else []
USER_SETTINGS = """${USER_SETTINGS[@]}""".split() if """${USER_SETTINGS[@]}""".strip() else []

SETTINGS = [
    ("_deepseek", "I1-instruction-deepseek"),
    ("_qwen72b", "I1-instruction-qwen72b"),
    ("_qwen7b", "I1-instruction-qwen7b"),
]
if USER_SETTINGS:
    SETTINGS = [(s, l) for s, l in SETTINGS if l in USER_SETTINGS]

with open(LEADERBOARD) as f:
    rows = list(csv.reader(f))
header = rows[0]

def col_idx(name):
    try: return header.index(name)
    except ValueError: return -1

idx_model = col_idx("model")
idx_seed = col_idx("seed")

for i, r in enumerate(rows[1:], 1):
    row = dict(zip(header, r))
    model = row.get("model", "")
    seed = row.get("seed", "")
    if USER_FILTERS and not any(fnmatch.fnmatch(model, pat) for pat in USER_FILTERS):
        continue
    for suffix, setting_label in SETTINGS:
        ts = row.get(f"answer_ts{suffix}", "").strip()
        if not ts:
            continue
        # exp = workspace dir name. Agent workspaces embed timestamps; baseline
        # workspaces use the bare baseline key (e.g. "greedy_chain"). Glob
        # across all possible exp dirs for this setting/ts.
        pattern = f"{SAVE_PATH}/agent-tool-reasoning/*/seed_{seed}/{setting_label}/{ts}/G1_instruction"
        matches = glob.glob(pattern)
        if not matches:
            # Fallback: also try without seed subdir (defensive)
            pattern2 = f"{SAVE_PATH}/agent-tool-reasoning/*/seed_*/{setting_label}/{ts}/G1_instruction"
            matches = glob.glob(pattern2)
        if not matches:
            print(f"# MISSING answer dir for row {i} model={model} setting={setting_label} ts={ts}",
                  flush=True)
            continue
        # If multiple matches (unlikely — ts is unique), pick the one whose
        # exp dir contains the model name or is newest.
        if len(matches) > 1:
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        answer_dir = matches[0]
        print("\t".join([str(i), suffix, setting_label, ts, model, seed, answer_dir]))
PY

N_TASKS=$(grep -vc '^#' "${TASK_LIST_FILE}" || true)
echo "=== ${N_TASKS} (row × setting) pairs to evaluate ==="
grep '^#' "${TASK_LIST_FILE}" >&2 || true  # print MISSING warnings

# ── run convert + eval for each task ─────────────────────────────────
cd "${PKG_ROOT}/toolbench/tooleval"

# Map (row_idx, suffix) -> (sopr_value, n_scored)
RESULT_FILE="${WORK_DIR}/row_sopr_results.tsv"
: > "${RESULT_FILE}"

while IFS=$'\t' read -r ROW_IDX SUFFIX SETTING_LABEL TS MODEL SEED ANSWER_DIR; do
    [ -z "${ROW_IDX}" ] && continue
    [[ "${ROW_IDX}" == \#* ]] && continue

    # Build a safe name unique per (row, setting)
    SAFE_MODEL="${MODEL//[\/:]/_}"
    SAFE="row${ROW_IDX}_${SAFE_MODEL}_${SETTING_LABEL}_${TS}"
    CONV_DIR="${WORK_DIR}/converted/${SAFE}"
    RESULT_DIR="${WORK_DIR}/results/${SAFE}"
    mkdir -p "${CONV_DIR}" "${RESULT_DIR}"
    rm -f "${CONV_DIR}/G1_instruction.json"

    # Convert
    $CONDA_RUN python "${TASK_DIR}/scripts/convert_answers_local.py" \
        --answer_dir "${ANSWER_DIR}" \
        --output "${CONV_DIR}/G1_instruction.json" \
        > "${WORK_DIR}/logs/convert_${SAFE}.log" 2>&1 || {
            echo "CONVERT FAILED: row=${ROW_IDX} setting=${SETTING_LABEL} (see ${WORK_DIR}/logs/convert_${SAFE}.log)" >&2
            continue
        }

    # Evaluate
    API_POOL_FILE="${API_POOL_FILE}" EVAL_MODEL="${EVAL_MODEL}" \
    $CONDA_RUN python eval_pass_rate.py \
        --converted_answer_path "${WORK_DIR}/converted" \
        --save_path "${RESULT_DIR}" \
        --reference_model "${SAFE}" \
        --test_ids "${PKG_ROOT}/solvable_queries/test_query_ids" \
        --evaluator tooleval_gpt-3.5-turbo_default \
        --max_eval_threads "${MAX_EVAL_THREADS}" \
        --evaluate_times "${EVALUATE_TIMES}" \
        --test_set G1_instruction \
        --overwrite \
        > "${WORK_DIR}/logs/eval_${SAFE}.log" 2>&1 || {
            echo "EVAL FAILED: row=${ROW_IDX} setting=${SETTING_LABEL} (see ${WORK_DIR}/logs/eval_${SAFE}.log)" >&2
            continue
        }

    JSON_FILE="${RESULT_DIR}/G1_instruction_${SAFE}.json"
    read VAL N <<<$(python3 - <<PY
import json
d = json.load(open("${JSON_FILE}"))
total = len(d)
score = 0.0
for v in d.values():
    s = str(v.get('is_solved', {}))
    if 'AnswerStatus.Solved' in s:
        score += 1.0
    elif 'AnswerStatus.Unsure' in s:
        score += 0.5
print(f"{score/total:.4f}" if total else "NaN", total)
PY
)
    echo "[SoPR] row=${ROW_IDX} model=${MODEL} setting=${SETTING_LABEL} ts=${TS}: ${VAL} (n=${N})"
    printf '%s\t%s\t%s\t%s\n' "${ROW_IDX}" "${SUFFIX}" "${VAL}" "${N}" >> "${RESULT_FILE}"
done < "${TASK_LIST_FILE}"

echo
echo "=== SoPR SUMMARY (eval_model=${EVAL_MODEL}, evaluate_times=${EVALUATE_TIMES}) ==="
cat "${RESULT_FILE}" | while IFS=$'\t' read -r ROW_IDX SUFFIX VAL N; do
    echo "row${ROW_IDX}${SUFFIX}: sopr=${VAL} n_scored=${N}"
done

# ── update leaderboard ──────────────────────────────────────────────
if [ "${WRITE_LEADERBOARD}" -eq 1 ]; then
    python3 - <<PY
import csv

results = {}  # {row_idx(int): {suffix: (val, n)}}
with open("${RESULT_FILE}") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 4:
            continue
        row_idx, suffix, val, n = parts
        results.setdefault(int(row_idx), {})[suffix] = (val, n)

path = "${LEADERBOARD}"
with open(path) as f:
    rows = list(csv.reader(f))
header = rows[0]
data = rows[1:]

# Ensure per-setting sopr columns exist
for suffix in ("_deepseek", "_qwen72b", "_qwen7b"):
    for col in (f"sopr{suffix}", f"sopr_n_scored{suffix}"):
        if col not in header:
            header.append(col)

for r in data:
    while len(r) < len(header):
        r.append("")

updated = 0
for row_idx, per_suffix in results.items():
    data_idx = row_idx - 1  # rows list includes header; data starts at rows[1:]
    if data_idx < 0 or data_idx >= len(data):
        continue
    r = data[data_idx]
    for suffix, (val, n) in per_suffix.items():
        sopr_col = f"sopr{suffix}"
        n_col = f"sopr_n_scored{suffix}"
        try:
            r[header.index(sopr_col)] = f"{float(val):.4f}" if val not in ("", "NaN") else ""
            r[header.index(n_col)] = str(n)
            updated += 1
        except (ValueError, IndexError):
            pass

with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(data)

print(f"[leaderboard] updated {updated} (row, setting) cells in {path}")
PY
fi
