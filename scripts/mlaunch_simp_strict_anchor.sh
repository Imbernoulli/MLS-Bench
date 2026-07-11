#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "usage: $0 RUN_ID TASK ZONE IMAGE [PRIORITY]" >&2
    exit 2
fi

RUN_ID=$1
TASK=$2
ZONE=$3
IMAGE=$4
PRIORITY=${5:-599}
STAGE=${SIMP_STRICT_STAGE:-/home/lvbohan/simp-strict-20260711}
SOURCE=${SIMP_STRICT_SOURCE:-${STAGE}/source}
MOUNT_ROOT=${SIMP_STRICT_MOUNT_ROOT:-/home/lvbohan}
RUN=${STAGE}/anchors/${RUN_ID}/${TASK}

case "${TASK}" in
    simp-beam-width) HARNESS=harness_beamwidth.py ;;
    simp-decoding-beam) HARNESS=harness_beam.py ;;
    simp-decoding-strategy) HARNESS=harness_strategy.py ;;
    simp-decoding-temperature) HARNESS=harness_temperature.py ;;
    simp-input-truncation) HARNESS=harness_truncation.py ;;
    simp-length-control) HARNESS=harness_length.py ;;
    simp-minlen-floor) HARNESS=harness_minlen.py ;;
    simp-model-capacity) HARNESS=harness_capacity.py ;;
    simp-nucleus-sampling) HARNESS=harness_nucleus.py ;;
    simp-source-policy) HARNESS=harness_policy.py ;;
    *) echo "unknown simplification task: ${TASK}" >&2; exit 2 ;;
esac
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid run id: ${RUN_ID}" >&2
    exit 2
fi
if [[ ! "${ZONE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid zone: ${ZONE}" >&2
    exit 2
fi
if [[ ! "${IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "image must be pinned by digest: ${IMAGE}" >&2
    exit 2
fi
resume_prepared=false
if [[ -e "${RUN}" ]]; then
    [[ -d "${RUN}" ]] || { echo "run path is not a directory: ${RUN}" >&2; exit 2; }
    python - "${RUN}/launch-request.json" "${RUN_ID}" "${TASK}" "${ZONE}" "${IMAGE}" "${SOURCE}" <<'PY'
import json, sys
path, run_id, task, zone, image, source = sys.argv[1:]
try:
    request = json.load(open(path))
except (OSError, ValueError) as exc:
    raise SystemExit(f"cannot recover run with an invalid launch request: {exc}")
expected = {
    "run_id": run_id,
    "task": task,
    "zone": zone,
    "image": image,
    "source": source,
}
for key, value in expected.items():
    if request.get(key) != value:
        raise SystemExit(
            f"refusing to recover run with mismatched {key}: "
            f"recorded={request.get(key)!r} requested={value!r}"
        )
PY
    status=$(head -n 1 "${RUN}/status" 2>/dev/null || true)
    recorded_rc=$(head -n 1 "${RUN}/task.rc" 2>/dev/null || true)

    if [[ "${status}" == "success" || -e "${RUN}/SUCCESS" ]]; then
        if [[ "${recorded_rc}" != "0" ]]; then
            echo "inconsistent successful run has rc=${recorded_rc:-missing}: ${RUN}" >&2
            exit 1
        fi
        echo "SIMP_STRICT_ANCHOR_RECOVERED task=${TASK} state=success rc=0 run=${RUN}"
        exit 0
    fi
    if [[ "${status}" == "failed" || "${status}" == "launch_failed" || -e "${RUN}/LAUNCH_FAILED" ]]; then
        if [[ ! "${recorded_rc}" =~ ^[1-9][0-9]*$ || "${recorded_rc}" -gt 255 ]]; then
            echo "inconsistent failed run has rc=${recorded_rc:-missing}: ${RUN}" >&2
            exit 1
        fi
        echo "SIMP_STRICT_ANCHOR_RECOVERED task=${TASK} state=${status:-failed} rc=${recorded_rc} run=${RUN}" >&2
        exit "${recorded_rc}"
    fi
    if [[ "${recorded_rc}" =~ ^[0-9]+$ && "${recorded_rc}" != "125" ]]; then
        if [[ "${recorded_rc}" -eq 0 ]]; then
            echo "SIMP_STRICT_ANCHOR_RECOVERED task=${TASK} state=terminal rc=0 run=${RUN}"
            exit 0
        fi
        if [[ "${recorded_rc}" -le 255 ]]; then
            echo "SIMP_STRICT_ANCHOR_RECOVERED task=${TASK} state=terminal rc=${recorded_rc} run=${RUN}" >&2
            exit "${recorded_rc}"
        fi
        echo "invalid recorded terminal rc=${recorded_rc}: ${RUN}" >&2
        exit 1
    fi
    if [[ -s "${RUN}/worker.name" ]]; then
        worker=$(head -n 1 "${RUN}/worker.name")
        echo "SIMP_STRICT_ANCHOR_RECOVERED task=${TASK} state=${status:-submitted} worker=${worker} run=${RUN}"
        exit 0
    fi
    if [[ -e "${RUN}/STARTED" ]]; then
        echo "SIMP_STRICT_ANCHOR_RECOVERED task=${TASK} state=${status:-running} run=${RUN}"
        exit 0
    fi
    if [[ "${status}" == "prepared" && -e "${RUN}/PREPARED" && ! -e "${RUN}/LAUNCHING" && ! -e "${RUN}/SUBMITTED" ]]; then
        python - "${RUN}/launch-request.json" "${MOUNT_ROOT}" "${PRIORITY}" <<'PY'
import json, sys
path, mount_root, priority = sys.argv[1:]
request = json.load(open(path))
for key, value in {"mount_root": mount_root, "priority": priority}.items():
    if request.get(key) != value:
        raise SystemExit(
            f"refusing to resume prepared run with mismatched {key}: "
            f"recorded={request.get(key)!r} requested={value!r}"
        )
PY
        resume_prepared=true
    else
        echo "ambiguous existing run may already have been submitted; use a new RUN_ID: ${RUN}" >&2
        exit 75
    fi
fi

for path in "${SOURCE}" "${MOUNT_ROOT}"; do
    [[ -d "${path}" ]] || { echo "missing required directory: ${path}" >&2; exit 2; }
done

if [[ "${resume_prepared}" == false ]]; then
    mkdir -p "$(dirname "${RUN}")"
    mkdir "${RUN}"
    printf 'prepared\n' > "${RUN}/status"
    printf '125\n' > "${RUN}/task.rc"
    python - "${RUN}/launch-request.json" "${RUN_ID}" "${TASK}" "${ZONE}" "${IMAGE}" "${SOURCE}" "${MOUNT_ROOT}" "${PRIORITY}" <<'PY'
import json, sys
from pathlib import Path
path, run_id, task, zone, image, source, mount_root, priority = sys.argv[1:]
request = {
    "run_id": run_id,
    "task": task,
    "zone": zone,
    "image": image,
    "source": source,
    "mount_root": mount_root,
    "priority": priority,
    "gpu_count": 1,
    "seed": 42,
    "protocol": "gem-full-test-v2",
    "runtime_install": False,
    "runtime_download": False,
}
temporary = Path(path + ".tmp")
temporary.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
    date -Iseconds > "${RUN}/PREPARED"
fi

priority_args=(--priority "${PRIORITY}")
if [[ "${PRIORITY}" == "preemptible" ]]; then
    priority_args=(--preemptible=yes)
fi

date -Iseconds > "${RUN}/LAUNCHING"
printf 'launching\n' > "${RUN}/status"
set +e
launch_output=$(mlaunch -d \
    -z "${ZONE}" \
    --gpu=1 \
    "${priority_args[@]}" \
    --preemption-policy-never=false \
    --max-wait-duration 24h \
    --max-idle-duration 2h \
    --i-know-i-am-wasting-resource \
    --enable-sshd=false \
    --comment "strict full-split ${TASK}" \
    --image "${IMAGE}" \
    --image-pull-policy IfNotPresent \
    --volume "${MOUNT_ROOT}:${MOUNT_ROOT}" \
    -w "${RUN}" \
    -- bash -lc "
set +e
runtime_rc=125
finalize() {
    exit_rc=\$?
    if [[ \${runtime_rc} -eq 125 && \${exit_rc} -ne 125 ]]; then
        runtime_rc=\${exit_rc}
    fi
    printf '%s\\n' \"\${runtime_rc}\" > '${RUN}/task.rc'
    if [[ \${runtime_rc} -eq 0 ]]; then
        printf 'success\\n' > '${RUN}/status'
        date -Iseconds > '${RUN}/SUCCESS'
    else
        printf 'failed\\n' > '${RUN}/status'
    fi
}
trap finalize EXIT
cd '${RUN}' || { runtime_rc=111; exit \${runtime_rc}; }
date -Iseconds > STARTED || { runtime_rc=1; exit \${runtime_rc}; }
printf 'running\\n' > status || { runtime_rc=1; exit \${runtime_rc}; }
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SIMP_MODEL=/data/text-simplification/models/t5-base-finetuned-turk-text-simplification
export TASK_DIR='${SOURCE}/tasks/${TASK}'
export PYTHONPATH='${SOURCE}/src:${SOURCE}/vendor/text-simplification'
python - '${TASK}' '${ZONE}' '${IMAGE}' <<'PY' > runtime.json
import json, sys
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print(json.dumps({
    'event': 'SIMP_ANCHOR_RUNTIME',
    'task': sys.argv[1],
    'zone': sys.argv[2],
    'image': sys.argv[3],
    'gpu_count': torch.cuda.device_count(),
    'gpu_name': torch.cuda.get_device_name(0),
    'torch': torch.__version__,
}, sort_keys=True), flush=True)
PY
runtime_rc=\$?
if [[ \${runtime_rc} -eq 0 ]]; then
    mkdir surfaces
    runtime_rc=\$?
fi
if [[ \${runtime_rc} -eq 0 ]]; then
    python - '${SOURCE}/tasks/${TASK}/config.json' <<'PY' > baselines.txt
import json, sys
for baseline in json.load(open(sys.argv[1]))['baselines']:
    print(baseline)
PY
    runtime_rc=\$?
fi
if [[ \${runtime_rc} -eq 0 ]]; then
    python - '${SOURCE}/tasks/${TASK}/config.json' '${TASK}' '${HARNESS}' <<'PY' > protocol.files
import json, sys
from pathlib import PurePosixPath

config_path, task, harness = sys.argv[1:]
config = json.load(open(config_path))

def require_relative(path):
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or '..' in parsed.parts:
        raise ValueError(f'non-relative protocol path: {path!r}')
    return parsed.as_posix()

files = config.get('files')
if not isinstance(files, list) or len(files) != 1:
    raise ValueError('task must configure exactly one native solution file')
native = require_relative(files[0]['filename'])
if not native.startswith('text-simplification/solution/'):
    raise ValueError(f'unexpected native solution path: {native!r}')

paths = [
    'vendor/text-simplification/common.py',
    'vendor/text-simplification/sari.py',
    f'vendor/text-simplification/{harness}',
    f'tasks/{task}/parser.py',
    f'tasks/{task}/config.json',
    f'tasks/{task}/data/simp_asset_refs.jsonl',
    f'tasks/{task}/data/simp_turk_refs.jsonl',
    f'tasks/{task}/data/simp_wiki_refs.jsonl',
    'vendor/text-simplification/_simp_data/simp_asset_src.jsonl',
    'vendor/text-simplification/_simp_data/simp_turk_src.jsonl',
    'vendor/text-simplification/_simp_data/simp_wiki_src.jsonl',
    'scripts/materialize_simp_anchor_surface.py',
    f'vendor/{native}',
]
baselines = config.get('baselines')
if not isinstance(baselines, dict) or not baselines:
    raise ValueError('task has no configured baselines')
for name in sorted(baselines):
    entry = baselines[name]
    if not isinstance(entry, dict) or set(entry) != {'edit_ops'}:
        raise ValueError(f'malformed baseline: {name!r}')
    edit_ops = require_relative(entry['edit_ops'])
    paths.append(f'tasks/{task}/{edit_ops}')
print(*paths, sep='\n')
PY
    runtime_rc=\$?
fi
if [[ \${runtime_rc} -eq 0 ]]; then
    while IFS= read -r baseline; do
        python '${SOURCE}/scripts/materialize_simp_anchor_surface.py' \\
            --task '${TASK}' --baseline \"\${baseline}\" \\
            --output \"\$PWD/surfaces/\${baseline}.py\" \\
            > \"materialize.\${baseline}.log\" 2>&1
        runtime_rc=\$?
        if [[ \${runtime_rc} -ne 0 ]]; then break; fi
    done < baselines.txt
fi
if [[ \${runtime_rc} -eq 0 ]]; then
    (
        cd '${SOURCE}' || exit 111
        while IFS= read -r protocol_path; do
            sha256sum \"\${protocol_path}\" || exit \$?
        done < '${RUN}/protocol.files'
    ) > protocol.sha256
    runtime_rc=\$?
fi
if [[ \${runtime_rc} -eq 0 ]]; then
    anchor_rc=0
    while IFS= read -r baseline; do
        python '${SOURCE}/vendor/text-simplification/${HARNESS}' \\
            --solution \"\$PWD/surfaces/\${baseline}.py\" --seed 42 \\
            > \"\${baseline}.worker.log\" 2>&1
        cell_rc=\$?
        printf '%s\\n' \"\${cell_rc}\" > \"\${baseline}.rc\"
        sha256sum \"\${baseline}.worker.log\" > \"\${baseline}.worker.log.sha256\"
        log_sha_rc=\$?
        printf '125\\n' > \"\${baseline}.parser.rc\"
        if [[ \${cell_rc} -ne 0 ]]; then
            if [[ \${anchor_rc} -eq 0 ]]; then anchor_rc=\${cell_rc}; fi
            continue
        fi
        if [[ \${log_sha_rc} -ne 0 && \${anchor_rc} -eq 0 ]]; then
            anchor_rc=\${log_sha_rc}
        fi
        python - '${SOURCE}/tasks/${TASK}/parser.py' \"\${baseline}.worker.log\" <<'PY'
import importlib.util, sys
parser_path, log_path = sys.argv[1:]
spec = importlib.util.spec_from_file_location('strict_anchor_parser', parser_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.Parser().parse('simplify', open(log_path).read())
expected = {'sari_asset', 'bleu_asset', 'sari_turk', 'bleu_turk', 'sari_wiki', 'bleu_wiki'}
if set(result.metrics) != expected:
    raise SystemExit(result.feedback)
PY
        parser_rc=\$?
        printf '%s\\n' \"\${parser_rc}\" > \"\${baseline}.parser.rc\"
        if [[ \${parser_rc} -ne 0 && \${anchor_rc} -eq 0 ]]; then
            anchor_rc=\${parser_rc}
        fi
    done < baselines.txt
    runtime_rc=\${anchor_rc}
fi
exit \${runtime_rc}
" 2>&1)
launch_rc=$?
set -e

printf '%s\n' "${launch_output}" > "${RUN}/mlaunch.log"
if [[ ${launch_rc} -ne 0 ]]; then
    printf '%s\n' "${launch_rc}" > "${RUN}/task.rc"
    printf 'launch_failed\n' > "${RUN}/status"
    date -Iseconds > "${RUN}/LAUNCH_FAILED"
    exit "${launch_rc}"
fi
worker=$(printf '%s\n' "${launch_output}" | tail -n 1)
if [[ -z "${worker}" ]]; then
    printf 'submission_ambiguous\n' > "${RUN}/status"
    date -Iseconds > "${RUN}/SUBMISSION_AMBIGUOUS"
    echo "mlaunch returned success without a worker receipt; use a new RUN_ID: ${RUN}" >&2
    exit 75
fi
printf '%s\n' "${worker}" > "${RUN}/worker.name.tmp"
mv "${RUN}/worker.name.tmp" "${RUN}/worker.name"
date -Iseconds > "${RUN}/SUBMITTED"
echo "SIMP_STRICT_ANCHOR_SUBMITTED task=${TASK} worker=${worker} run=${RUN}"
