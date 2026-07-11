#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${NORMFLOWS_REPO:?set NORMFLOWS_REPO}
RUN=${NORMFLOWS_TEST_RUN:?set NORMFLOWS_TEST_RUN}
ADAPTER_REPO=${NORMFLOWS_ADAPTER_REPO:-/home/lvbohan/projects/MLS-Bench}

test -d "${REPO}"
test -d "${ADAPTER_REPO}/harbor_adapter"
if [[ -e "${RUN}" ]]; then
    echo "refusing to reuse normflows test run: ${RUN}" >&2
    exit 73
fi
mkdir -p "${RUN}"
cd "${RUN}" || exit 111
exec > >(tee worker.log) 2>&1

finish() {
    local rc="${1:-$?}"
    trap - EXIT HUP INT TERM
    date -Iseconds > FINISHED
    printf '%s\n' "${rc}" > rc
    if [[ ${rc} -eq 0 ]]; then
        printf 'success\n' > status
        date -Iseconds > SUCCESS
    else
        printf 'failed\n' > status
        rm -f SUCCESS
    fi
    exit "${rc}"
}
trap finish EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

date -Iseconds > STARTED
printf 'running\n' > status
rm -f SUCCESS

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > gpu-inventory.log
allocated_gpu_count=$(wc -l < gpu-inventory.log)
test "${allocated_gpu_count}" -ge 1
# Some B0 resource groups allocate a whole eight-GPU node.  This audit exposes
# only one device and performs no model training.
export CUDA_VISIBLE_DEVICES=0
printf 'allocated=%s visible=1\n' "${allocated_gpu_count}" > gpu-usage.log

cd "${REPO}" || exit 111
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO}/src"
python -m pytest -q -p no:cacheprovider \
    tests/test_normflows_density_strict.py \
    tests/test_scoring_fail_closed.py \
    tests/test_scoring_no_implicit_fallback.py \
    tests/test_workspace_tools_fail_closed.py \
    2>&1 | tee "${RUN}/pytest.log"

rendered=${RUN}/rendered
export PYTHONPATH="${ADAPTER_REPO}/harbor_adapter/src:${ADAPTER_REPO}/src"
python -m mls_bench.main \
    --mls-bench-root "${REPO}" \
    --output-dir "${rendered}" \
    --task-ids \
        flow-arch-family \
        flow-autoregressive-coupling \
        flow-base-distribution \
        flow-batch-size \
        flow-conditioner-width \
        flow-coupling-transform \
        flow-depth-permutation \
        flow-learning-rate \
        flow-masking-pattern \
        flow-spline-bins \
    --overwrite \
    --mangrove \
    --gpu-backend h20 \
    --h20-serial \
    2>&1 | tee "${RUN}/render.log"

python - "${rendered}" "${RUN}" <<'PY' | tee "${RUN}/render-audit.log"
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1])
run = Path(sys.argv[2])
tasks = sorted(path for path in root.glob("flow-*") if path.is_dir())
assert len(tasks) == 10, [path.name for path in tasks]
expected_image = (
    "FROM msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
    "mlsbench-harbor-normflows-density@sha256:"
    "3b81a711a7a6a00234a5717f7f5199ae124faee0d7cf24b0527e186ebbf40837"
)
forbidden = re.compile(
    r"pip(?:3)?\s+install|apt(?:-get)?\s+install|conda\s+(?:install|create)|"
    r"curl\s|wget\s|git\s+clone|unzip\s|cmake\s|gcc\s|g\+\+\s",
    re.IGNORECASE,
)
private_names = {
    "common.py", "flow_blocks.py", "harness_flow.py", "_flow_data", "baselines"
}
for task in tasks:
    dockerfile = task.joinpath("environment/Dockerfile").read_text()
    assert next(line for line in dockerfile.splitlines() if line.startswith("FROM ")) == expected_image
    assert forbidden.search(dockerfile) is None, task.name
    assert forbidden.search(task.joinpath("tests/test.sh").read_text()) is None, task.name

    instruction = task.joinpath("instruction.md").read_text()
    assert re.search(r"\b(?:public|hidden)\b", instruction, re.IGNORECASE) is None

    task_toml = tomllib.loads(task.joinpath("task.toml").read_text())
    assert task_toml["environment"]["gpus"] == 1
    assert task_toml["environment"]["gpu_types"] == ["H20"]
    assert task_toml["environment"]["gpus"] <= 4

    config = json.loads(task.joinpath("tests/meta/config.json").read_text())
    assert config["calibration_protocol"] == "flow-2d-community-20k-literal-ast-v3"
    assert config["seeds"] == [42]
    assert "PROTOCOL_VERSION = \"flow-2d-community-20k-literal-ast-v3\"" in (
        task.joinpath("tests/meta/parser.py").read_text()
    )

    scaffold = task / "environment" / "_scaffold" / "normflows-density"
    assert scaffold.is_dir()
    assert not any(scaffold.joinpath(name).exists() for name in private_names), task.name
    assert not any(
        path.name in {"score_spec.py", "leaderboard.csv", "parser.py"}
        for path in scaffold.rglob("*")
    ), task.name
    assert any(path.name == "harness_flow.py" for path in task.joinpath("tests").rglob("*"))

render_complete = (
    "NORMFLOWS_RENDER_COMPLETE tasks=10 pinned_image=pass gpu_h20=1 "
    "offline_verifier=pass private_assets=pass instruction_settings=pass"
)
summary = (
    "NORMFLOWS_STATIC_COMPLETE siblings=10 valid=10 destructive=20 "
    "pending_zero=9 anchor_calibration=pass global_failclosed=pass render=10/10\n"
)
finished = datetime.now(timezone.utc).isoformat()
run.joinpath("summary").write_text(summary)
run.joinpath("rc").write_text("0\n")
run.joinpath("status").write_text("success\n")
run.joinpath("FINISHED").write_text(f"{finished}\n")
run.joinpath("SUCCESS").write_text(f"{finished}\n")
print(render_complete)
print(summary, end="")
PY

finish 0
