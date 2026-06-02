from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import argparse
from pathlib import Path


def _load_score_task():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mls_bench"
        / "task-template"
        / "tests"
        / "score_task.py"
    )
    spec = importlib.util.spec_from_file_location("score_task_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_guard_fixture(tmp_path: Path, *, allow_create: bool) -> tuple[Path, Path, Path, Path]:
    task_meta = tmp_path / "meta"
    pristine = tmp_path / "pristine"
    workspace = tmp_path / "workspace"
    for root in (task_meta, pristine / "pkg", workspace / "pkg"):
        root.mkdir(parents=True)

    content = b"def f():\n    return 1\n"
    (pristine / "pkg" / "existing.py").write_bytes(content)
    (workspace / "pkg" / "existing.py").write_bytes(content)
    (task_meta / "config.json").write_text(json.dumps({
        "allow_create": allow_create,
        "files": [
            {
                "filename": "pkg/existing.py",
                "edit": [{"start": 1, "end": -1}],
            }
        ],
    }))
    (task_meta / "pristine_manifest.json").write_text(json.dumps({
        "pkg/existing.py": hashlib.sha256(content).hexdigest(),
    }))
    return task_meta, pristine, workspace, tmp_path / "violation.txt"


def test_allow_create_false_outside_prefix(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, pristine, workspace, violation = _write_guard_fixture(
        tmp_path,
        allow_create=False,
    )
    (workspace / "sitecustomize.py").write_text("print('bypass')\n")

    rc = score_task.cmd_guard(argparse.Namespace(
        task_meta=str(task_meta),
        pristine=str(pristine),
        workspace=str(workspace),
        violation_out=str(violation),
    ))

    assert rc == 10
    assert "created new file (allow_create=false): sitecustomize.py" in violation.read_text()


def test_verifier_task_dir_exempt(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, pristine, workspace, violation = _write_guard_fixture(
        tmp_path,
        allow_create=False,
    )
    task_dir = workspace / "_task"
    task_dir.mkdir()
    (task_dir / "config.json").write_text("{}\n")

    rc = score_task.cmd_guard(argparse.Namespace(
        task_meta=str(task_meta),
        pristine=str(pristine),
        workspace=str(workspace),
        violation_out=str(violation),
    ))

    assert rc == 0
    assert not violation.exists()


def test_allow_create_true_outside_prefix(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, pristine, workspace, violation = _write_guard_fixture(
        tmp_path,
        allow_create=True,
    )
    (workspace / "sitecustomize.py").write_text("print('allowed')\n")

    rc = score_task.cmd_guard(argparse.Namespace(
        task_meta=str(task_meta),
        pristine=str(pristine),
        workspace=str(workspace),
        violation_out=str(violation),
    ))

    assert rc == 0
    assert not violation.exists()


def test_edit_guard_rejects_deleted_fixed_separator_with_duplicate_in_editable(tmp_path: Path):
    score_task = _load_score_task()
    pristine = tmp_path / "pristine.py"
    current = tmp_path / "current.py"

    pristine.write_text(
        "header\n"
        "editable before\n"
        "===\n"
        "editable after\n"
        "===\n"
        "second editable\n"
        "tail\n"
    )
    current.write_text(
        "header\n"
        "editable before\n"
        "===\n"
        "editable after\n"
        "second editable\n"
        "tail\n"
    )

    ranges = [score_task.EditRange(2, 4), score_task.EditRange(6, 6)]
    ok, reason = score_task._check_editable_only(pristine, current, ranges)

    assert not ok
    assert reason is not None
    assert "only the declared editable range" in reason


def test_edit_guard_rejects_protected_line_with_open_ended_tail_range(tmp_path: Path):
    """Regression: a disjoint range list whose last range uses end=-1 (to-EOF)
    must NOT mark the whole file editable for the line-level backstop. Here the
    protected line (4) has a duplicate in the editable tail so the byte-anchor
    pass is satisfied — only the line-level check can catch the change."""
    score_task = _load_score_task()
    pristine = tmp_path / "pristine.py"
    current = tmp_path / "current.py"

    pristine.write_text(
        "header\n"      # 1 fixed (range starts at 2)
        "edit a\n"      # 2 editable
        "edit b\n"      # 3 editable
        "PROT\n"        # 4 PROTECTED (between the two ranges)
        "edit c\n"      # 5 editable (range 5..EOF)
        "PROT\n"        # 6 editable tail — same text as line 4
    )
    current.write_text(
        "header\n"
        "edit a\n"
        "edit b\n"
        "EVIL\n"        # 4 changed — must be rejected
        "edit c\n"
        "PROT\n"
    )

    ranges = [score_task.EditRange(2, 3), score_task.EditRange(5, -1)]
    ok, reason = score_task._check_editable_only(pristine, current, ranges)

    assert not ok
    assert reason is not None
    assert "only the declared editable range" in reason


def test_edit_guard_allows_legit_open_ended_tail_edit(tmp_path: Path):
    """Companion to the regression above: a legitimate edit confined to the
    open-ended tail range must still pass."""
    score_task = _load_score_task()
    pristine = tmp_path / "pristine.py"
    current = tmp_path / "current.py"

    pristine.write_text(
        "header\n"      # 1 fixed
        "edit a\n"      # 2 editable
        "edit b\n"      # 3 editable
        "PROT\n"        # 4 PROTECTED
        "edit c\n"      # 5 editable (5..EOF)
    )
    current.write_text(
        "header\n"
        "new a\n"       # 2 changed (editable) — ok
        "edit b\n"
        "PROT\n"        # 4 untouched
        "new c\n"       # 5 changed (editable) — ok
        "extra tail\n"  # appended within open-ended range — ok
    )

    ranges = [score_task.EditRange(2, 3), score_task.EditRange(5, -1)]
    ok, reason = score_task._check_editable_only(pristine, current, ranges)

    assert ok, reason


def test_metric_aggregation_coerces_strings_and_filters_nan():
    score_task = _load_score_task()

    mean = score_task._aggregate_metrics([
        {"acc": "0.5", "loss": float("nan")},
        {"acc": 1.0, "loss": 7.0},
    ])

    assert mean["acc"] == 0.75
    assert mean["loss"] == 7.0


def test_metric_aggregation_preserves_all_nan_when_no_finite_values():
    score_task = _load_score_task()

    mean = score_task._aggregate_metrics([
        {"acc": float("nan")},
        {"acc": "nan"},
    ])

    assert mean["acc"] != mean["acc"]


def test_sparse_seed_filter_drops_empty_and_elapsed_only_records():
    score_task = _load_score_task()

    valid = score_task._valid_seed_metric_records({
        1: {},
        2: {"elapsed_eval": 0.1},
        3: {"acc": "0.5", "elapsed_eval": 0.2},
    })

    assert valid == [{"acc": "0.5", "elapsed_eval": 0.2}]


def test_run_evals_records_elapsed_time(tmp_path: Path):
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    eval_root = tmp_path / "eval"
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    scripts = eval_root / "scripts"
    task_meta.mkdir()
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    (task_meta / "config.json").write_text(json.dumps({
        "test_cmds": [
            {
                "cmd": "scripts/eval.sh",
                "label": "eval",
                "package": "pkg",
                "time": "0:01:00",
                "compute": 1.0,
            }
        ],
        "seeds": [123],
    }))
    (task_meta / "package").write_text("pkg\n")
    (task_meta / "task_id").write_text("elapsed-task\n")
    script = scripts / "eval.sh"
    script.write_text("printf 'acc=0.5\\n'\n")

    rc = score_task.cmd_run_evals(argparse.Namespace(
        task_meta=str(task_meta),
        workspace=str(workspace),
        eval_root=str(eval_root),
        out_dir=str(tmp_path / "out"),
    ))
    summary = json.loads((tmp_path / "out" / "eval_summary.json").read_text())

    assert rc == 0
    assert summary[0]["logs"][0]["seed"] == 123
    assert isinstance(summary[0]["logs"][0]["elapsed"], float)
    assert summary[0]["logs"][0]["elapsed"] >= 0.0


def test_run_evals_applies_oracle_cmd_overrides(tmp_path: Path):
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    eval_root = tmp_path / "eval"
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    scripts = eval_root / "scripts"
    task_meta.mkdir()
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    (task_meta / "config.json").write_text(json.dumps({
        "use_cuda": False,
        "test_cmds": [
            {
                "cmd": "scripts/default.sh",
                "label": "eval",
                "package": "pkg",
                "time": "0:01:00",
            }
        ],
        "seeds": [123],
    }))
    (task_meta / "package").write_text("pkg\n")
    (task_meta / "task_id").write_text("mode1-oracle-task\n")
    (scripts / "default.sh").write_text("printf 'DEFAULT_SCRIPT\\n'\n")
    (scripts / "strong.sh").write_text("printf 'STRONG_BASELINE_SCRIPT\\n'\n")

    normal_out = tmp_path / "normal-out"
    normal_args = argparse.Namespace(
        task_meta=str(task_meta),
        workspace=str(workspace),
        eval_root=str(eval_root),
        out_dir=str(normal_out),
        oracle_cmd_overrides=None,
    )
    assert score_task.cmd_run_evals(normal_args) == 0
    normal_summary = json.loads((normal_out / "eval_summary.json").read_text())
    normal_log = Path(normal_summary[0]["logs"][0]["log"]).read_text()

    oracle_out = tmp_path / "oracle-out"
    oracle_args = argparse.Namespace(
        task_meta=str(task_meta),
        workspace=str(workspace),
        eval_root=str(eval_root),
        out_dir=str(oracle_out),
        oracle_cmd_overrides=json.dumps([
            {"label": "", "cmd": "scripts/strong.sh"},
        ]),
    )
    assert score_task.cmd_run_evals(oracle_args) == 0
    oracle_summary = json.loads((oracle_out / "eval_summary.json").read_text())
    oracle_log = Path(oracle_summary[0]["logs"][0]["log"]).read_text()

    assert "DEFAULT_SCRIPT" in normal_log
    assert "STRONG_BASELINE_SCRIPT" in oracle_log
    assert "DEFAULT_SCRIPT" not in oracle_log


def test_package_dir_matches_case_and_separators(tmp_path: Path):
    score_task = _load_score_task()
    workspace = tmp_path / "workspace"
    actual = workspace / "Nano-GPT"
    actual.mkdir(parents=True)

    resolved = score_task._package_dir(
        workspace,
        "fallback",
        {"package": "nano_gpt"},
    )

    assert resolved == actual
