#!/bin/bash
# Oracle solution: applies the strongest baseline edit_ops for cv-sample-weighting
# (balanced_softmax) to the workspace, then exits.
#
# Verification (run by Harbor verifier) will diff the workspace against
# /opt/mlsbench/original/, find only in-range modifications, run all eval
# scripts from /opt/mlsbench/eval/, and write the resulting combined_score
# to /logs/verifier/reward.txt.

set -euo pipefail

cd /workspace

python3 - <<'PY'
import json, sys
from pathlib import Path

# Harbor mounts the task's solution/ dir at /solution/ when running the oracle
# agent. baseline_edit_ops.json lives here; the agent NEVER has access to it.
ops_json = Path("/solution/baseline_edit_ops.json")
ops = json.loads(ops_json.read_text())

workdir = Path("/workspace")
for op in ops:
    rel = Path(op["file"])
    if rel.is_absolute() or ".." in rel.parts:
        print(f"unsafe op path: {op['file']}", file=sys.stderr)
        sys.exit(2)
    target = (workdir / rel).resolve()
    if workdir.resolve() not in [target, *target.parents]:
        print(f"op path escapes workdir: {op['file']}", file=sys.stderr)
        sys.exit(2)
    if op["op"] == "replace":
        lines = target.read_text().splitlines(keepends=True)
        start = int(op["start_line"]) - 1
        end = int(op["end_line"])
        if end == -1:
            end = len(lines)
        new = op["content"]
        if not new.endswith("\n"):
            new += "\n"
        target.write_text("".join(lines[:start]) + new + "".join(lines[end:]))
    elif op["op"] == "insert":
        lines = target.read_text().splitlines(keepends=True)
        after = int(op.get("after_line", 0))
        new = op["content"]
        if not new.endswith("\n"):
            new += "\n"
        target.write_text("".join(lines[:after]) + new + "".join(lines[after:]))
    elif op["op"] == "create":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(op["content"])
    elif op["op"] == "delete":
        lines = target.read_text().splitlines(keepends=True)
        start = int(op["start_line"]) - 1
        end = int(op.get("end_line", op["start_line"]))
        target.write_text("".join(lines[:start]) + "".join(lines[end:]))
    else:
        print(f"unknown op: {op['op']}", file=sys.stderr)
        sys.exit(2)

print(f"applied {len(ops)} baseline edit ops (balanced_softmax) to {workdir}")
PY
