"""Every rendered task's oracle baseline must satisfy the task's own guard.

The Harbor oracle replays the strongest declared baseline through
``solution/baseline_edit_ops.json`` and is then verified exactly like an
agent submission.  A baseline that touches lines outside the declared
editable ranges can therefore never score, which shows up as a Daytona/Harbor
"environment failure" even though the task definition is at fault
(cv-dbm-sampler's ``dbim_high_order`` baseline rewrote the protected
``sample_dbim`` signature).  Check all 140 rendered bundles in memory with
the verifier's own range check.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "harbor" / "tasks-docker"
TEMPLATE = REPO / "harbor_adapter" / "src" / "mls_bench" / "task-template" / "tests" / "score_task.py"

# Known task-definition bugs that are tracked but not fixed in this branch.
KNOWN_VIOLATIONS = {
    # CFGpp baseline replaces past the trailing fixed segment of
    # latent_diffusion.py; not part of MLS-Bench-Lite.
    "mls-bench__cv-diffusion-efficiency",
}


def _score_task_module():
    spec = importlib.util.spec_from_file_location("score_task_for_tests", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_task_for_tests"] = module
    spec.loader.exec_module(module)
    return module


def _violations(task_dir: Path, st) -> list[str]:
    ops_path = task_dir / "solution" / "baseline_edit_ops.json"
    if not ops_path.exists():
        return []
    ops = json.loads(ops_path.read_text())
    config = json.loads((task_dir / "tests" / "meta" / "config.json").read_text())
    editable = {
        f["filename"]: [st.EditRange(int(r["start"]), int(r["end"])) for r in (f.get("edit") or [])]
        for f in config.get("files", [])
    }
    pristine_root = task_dir / "tests" / "meta" / "pristine"
    files: dict[str, list[str] | None] = {}
    for op in ops:
        rel = op["file"]
        pristine = pristine_root / rel
        if rel not in files:
            files[rel] = pristine.read_text().splitlines(keepends=True) if pristine.exists() else None
        if files[rel] is None:
            continue
        if op["op"] == "replace":
            start, end = int(op["start_line"]) - 1, int(op["end_line"])
            content = op["content"] if op["content"].endswith("\n") else op["content"] + "\n"
            files[rel] = files[rel][:start] + content.splitlines(keepends=True) + files[rel][end:]
        elif op["op"] == "create":
            files[rel] = op["content"].splitlines(keepends=True)
    problems: list[str] = []
    for rel, lines in files.items():
        if lines is None:
            continue
        pristine = pristine_root / rel
        ranges = editable.get(rel)
        text = "".join(lines)
        if ranges is None:
            if pristine.exists() and pristine.read_text() != text:
                problems.append(f"{rel}: modified by the baseline but not declared editable")
            continue
        if not ranges:
            if pristine.read_text() != text:
                problems.append(f"{rel}: declared read-only but modified by the baseline")
            continue
        with tempfile.NamedTemporaryFile("w", suffix=Path(rel).suffix, delete=False) as handle:
            handle.write(text)
            current = Path(handle.name)
        ok, reason = st._check_editable_only(pristine, current, ranges)
        current.unlink(missing_ok=True)
        if not ok:
            problems.append(f"{rel}: {reason}")
    return problems


@pytest.mark.skipif(not TASKS.is_dir(), reason="rendered harbor/tasks not present")
def test_oracle_baselines_respect_editable_ranges():
    st = _score_task_module()
    failures = {}
    for task_dir in sorted(TASKS.glob("mls-bench__*")):
        problems = _violations(task_dir, st)
        if problems and task_dir.name not in KNOWN_VIOLATIONS:
            failures[task_dir.name] = problems
    assert not failures, json.dumps(failures, indent=2)


@pytest.mark.skipif(not TASKS.is_dir(), reason="rendered harbor/tasks not present")
def test_known_violations_are_still_violations():
    """Drop an entry from KNOWN_VIOLATIONS once the task is fixed."""
    st = _score_task_module()
    stale = [name for name in KNOWN_VIOLATIONS if (TASKS / name).is_dir() and not _violations(TASKS / name, st)]
    assert not stale, f"fixed tasks still listed as known violations: {stale}"


def test_scaffold_and_guard_baseline_agree_for_every_declared_file():
    """The workspace an agent starts from is the image's ``_scaffold`` copy; the
    guard diffs submissions against ``tests/meta/pristine``. When the two drift
    (ai4bio-protein-structure-repr: the scaffold gained a 17-line cache gate
    in a fixed region, the pristine copy did not) every submission — the oracle
    included — fails the guard with "fixed segment not found" and scores 0.
    Check all three rendered trees."""
    for variant in ("tasks-docker", "tasks-daytona", "tasks-modal"):
        root = REPO / "harbor" / variant
        if not root.is_dir():
            continue
        drift = []
        for task_dir in sorted(root.glob("mls-bench__*")):
            config = json.loads((task_dir / "tests" / "meta" / "config.json").read_text())
            manifest = json.loads((task_dir / "tests" / "meta" / "pristine_manifest.json").read_text())
            for f in config.get("files", []):
                if not f.get("edit"):
                    continue
                rel = f["filename"]
                scaffold = task_dir / "environment" / "_scaffold" / rel
                pristine = task_dir / "tests" / "meta" / "pristine" / rel
                if scaffold.exists() and pristine.exists():
                    if scaffold.read_bytes() != pristine.read_bytes():
                        drift.append(f"{variant}/{task_dir.name}: {rel} scaffold != pristine")
                    import hashlib
                    if manifest.get(rel) and manifest[rel] != hashlib.sha256(pristine.read_bytes()).hexdigest():
                        drift.append(f"{variant}/{task_dir.name}: {rel} manifest hash != pristine")
        assert not drift, "\n".join(drift)
