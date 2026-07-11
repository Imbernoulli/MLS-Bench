from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


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


def test_eval_env_points_scripts_at_sanitized_verifier_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    score_task = _load_score_task()
    task_meta = tmp_path / "private-meta"
    eval_meta = tmp_path / "eval-meta"
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    out_dir = tmp_path / "out"
    for path in (task_meta, eval_meta / "data", package, out_dir):
        path.mkdir(parents=True)
    (task_meta / "package").write_text("pkg\n")
    (task_meta / "task_id").write_text("task\n")
    monkeypatch.setenv("MLSBENCH_VERIFIER_DATA_ROOT", "/agent-controlled")

    env = score_task._eval_env(
        task_meta=task_meta,
        eval_task_meta=eval_meta,
        out_dir=out_dir,
        workspace_root=workspace,
        pkg_dir=package,
        tc={"label": "case", "package": "pkg"},
        seed=42,
    )

    assert env["MLSBENCH_VERIFIER_DATA_ROOT"] == str(eval_meta / "data")


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


def test_guard_accepts_unchanged_workspace_for_failed_or_noop_agent(tmp_path: Path):
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    pristine = task_meta / "pristine"
    workspace = tmp_path / "workspace"
    task_meta.mkdir()
    pristine.mkdir()
    workspace.mkdir()
    config = {
        "files": [{
            "filename": "pkg/solution.py",
            "edit": [{"start": 2, "end": 2}],
        }],
    }
    (task_meta / "config.json").write_text(json.dumps(config))
    (task_meta / "pristine_manifest.json").write_text(json.dumps({
        "pkg/solution.py": "unused-by-declared-file-check",
    }))
    (pristine / "pkg").mkdir()
    (workspace / "pkg").mkdir()
    source = "fixed\nreturn 'weak default'\nfixed\n"
    (pristine / "pkg" / "solution.py").write_text(source)
    (workspace / "pkg" / "solution.py").write_text(source)
    violation = tmp_path / "violation.txt"

    rc = score_task.cmd_guard(argparse.Namespace(
        task_meta=str(task_meta),
        pristine=str(pristine),
        workspace=str(workspace),
        violation_out=str(violation),
    ))

    assert rc == 0
    assert not violation.exists()


def test_guard_accepts_edit_within_declared_range(tmp_path: Path):
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    pristine = task_meta / "pristine"
    workspace = tmp_path / "workspace"
    task_meta.mkdir()
    pristine.mkdir()
    workspace.mkdir()
    config = {
        "files": [{
            "filename": "pkg/solution.py",
            "edit": [{"start": 2, "end": 2}],
        }],
    }
    (task_meta / "config.json").write_text(json.dumps(config))
    (task_meta / "pristine_manifest.json").write_text(json.dumps({
        "pkg/solution.py": "unused-by-declared-file-check",
    }))
    (pristine / "pkg").mkdir()
    (workspace / "pkg").mkdir()
    (pristine / "pkg" / "solution.py").write_text("fixed\nreturn 'weak'\nfixed\n")
    (workspace / "pkg" / "solution.py").write_text("fixed\nreturn 'agent'\nfixed\n")
    violation = tmp_path / "violation.txt"

    rc = score_task.cmd_guard(argparse.Namespace(
        task_meta=str(task_meta),
        pristine=str(pristine),
        workspace=str(workspace),
        violation_out=str(violation),
    ))

    assert rc == 0
    assert not violation.exists()


def test_metric_aggregation_rejects_entire_matrix_on_nonfinite_value():
    score_task = _load_score_task()

    mean = score_task._aggregate_metrics([
        {"acc": "0.5", "loss": float("nan")},
        {"acc": 1.0, "loss": 7.0},
    ])

    assert mean == {}


def test_metric_aggregation_rejects_all_nan_values():
    score_task = _load_score_task()

    mean = score_task._aggregate_metrics([
        {"acc": float("nan")},
        {"acc": "nan"},
    ])

    assert mean == {}


def test_geometric_mean_does_not_lift_a_zero_setting_with_epsilon():
    from mlsbench.scoring.evaluate import _gmean

    assert _gmean([0.99, 0.0, 0.99]) == 0.0


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf"), -0.1, "not-a-number"],
)
def test_geometric_mean_rejects_invalid_values(invalid_value):
    from mlsbench.scoring.evaluate import _gmean

    assert _gmean([0.99, invalid_value]) == 0.0


@pytest.mark.parametrize(
    "metrics",
    [
        {"score": "not-a-number"},
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": True},
        {"score": 1.0, "ignored_std": float("nan")},
        {"score": 1.0, "elapsed_eval": float("inf")},
        [("score", 1.0)],
    ],
)
def test_parser_metric_validation_rejects_every_invalid_value(metrics):
    score_task = _load_score_task()

    assert score_task._parser_metrics_error(metrics) is not None



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
                "hidden": True,
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
    assert set(summary[0]) == {"label", "logs"}
    assert summary[0]["logs"][0]["seed"] == 123
    assert isinstance(summary[0]["logs"][0]["elapsed"], float)
    assert summary[0]["logs"][0]["elapsed"] >= 0.0


def test_run_evals_fails_closed_when_command_prints_metric_then_exits_nonzero(tmp_path: Path):
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
        "test_cmds": [{
            "cmd": "scripts/eval.sh",
            "label": "eval",
            "package": "pkg",
            "time": "0:01:00",
            "compute": 1.0,
        }],
        "seeds": [123],
    }))
    (task_meta / "package").write_text("pkg\n")
    (task_meta / "task_id").write_text("failed-eval-task\n")
    (scripts / "eval.sh").write_text("printf 'acc=0.9\\n'\nexit 1\n")
    out_dir = tmp_path / "out"

    rc = score_task.cmd_run_evals(argparse.Namespace(
        task_meta=str(task_meta),
        workspace=str(workspace),
        eval_root=str(eval_root),
        out_dir=str(out_dir),
        oracle_cmd_overrides=None,
    ))

    assert rc == 1
    assert json.loads((out_dir / "eval_summary.json").read_text())[0]["logs"][0]["rc"] == 1
    assert "reward forced to 0" in (out_dir / "score_error.txt").read_text()


def test_run_evals_removes_stale_success_proof_before_failure(tmp_path: Path):
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    eval_root = tmp_path / "eval"
    workspace = tmp_path / "workspace"
    package = workspace / "pkg"
    scripts = eval_root / "scripts"
    out_dir = tmp_path / "out"
    task_meta.mkdir()
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    out_dir.mkdir()
    (task_meta / "config.json").write_text(json.dumps({
        "test_cmds": [{
            "cmd": "scripts/eval.sh",
            "label": "eval",
            "package": "pkg",
            "time": "0:01:00",
            "compute": 1.0,
        }],
        "seeds": [123],
    }))
    (task_meta / "package").write_text("pkg\n")
    (task_meta / "task_id").write_text("stale-proof-task\n")
    (scripts / "eval.sh").write_text("exit 7\n")
    (out_dir / "metrics.json").write_text('{"reward": 0.9}\n')
    (out_dir / "verification_result.json").write_text(
        '{"status": "passed", "reward": 0.9}\n'
    )
    (out_dir / "stale.log").write_text("acc=0.9\n")

    rc = score_task.cmd_run_evals(argparse.Namespace(
        task_meta=str(task_meta),
        workspace=str(workspace),
        eval_root=str(eval_root),
        out_dir=str(out_dir),
        oracle_cmd_overrides=None,
    ))

    assert rc == 1
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "verification_result.json").exists()
    assert not (out_dir / "stale.log").exists()
    assert "rc=7" in (out_dir / "score_error.txt").read_text()


def test_validate_eval_summary_rejects_partial_or_failed_matrix(tmp_path: Path):
    score_task = _load_score_task()
    ok_log = tmp_path / "ok.log"
    failed_log = tmp_path / "failed.log"
    ok_log.write_text("acc=0.5\n")
    failed_log.write_text("acc=0.9\n")
    config = {
        "test_cmds": [
            {"cmd": "scripts/visible.sh", "label": "visible"},
            {"cmd": "scripts/hidden.sh", "label": "hidden"},
        ],
        "seeds": [42, 43],
    }
    summary = [
        {"label": "visible", "logs": [
            {"seed": 42, "rc": 0, "log": str(ok_log)},
            {"seed": 43, "rc": 1, "log": str(failed_log)},
        ]},
        {"label": "hidden", "logs": [
            {"seed": 42, "rc": 0, "log": str(ok_log)},
        ]},
    ]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert "visible seed 43: eval exited with rc=1" in error
    assert "hidden seed 43: expected exactly one log, found 0" in error


def test_validate_eval_summary_requires_exact_labels_and_seeds(tmp_path: Path):
    score_task = _load_score_task()
    log = tmp_path / "ok.log"
    log.write_text("acc=0.5\n")
    config = {
        "test_cmds": [{"cmd": "scripts/eval.sh", "label": "expected"}],
        "seeds": [42],
    }
    summary = [
        {
            "label": "expected",
            "logs": [
                {"seed": 42, "rc": 0, "log": str(log)},
                {"seed": 43, "rc": 0, "log": str(log)},
            ],
        },
        {
            "label": "stale",
            "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
        },
    ]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert "unexpected summary labels: stale" in error
    assert "expected: unexpected seeds 43" in error


def test_validate_eval_summary_rejects_empty_success_log(tmp_path: Path):
    score_task = _load_score_task()
    log = tmp_path / "empty.log"
    log.write_text("")
    config = {
        "test_cmds": [{"cmd": "scripts/eval.sh", "label": "eval"}],
        "seeds": [42],
    }
    summary = [{
        "label": "eval",
        "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
    }]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert "eval log is empty" in error


@pytest.mark.parametrize(
    "config",
    [
        {
            "test_cmds": [
                {"cmd": "scripts/a.sh", "label": "duplicate"},
                {"cmd": "scripts/b.sh", "label": "duplicate"},
            ],
            "seeds": [42],
        },
        {
            "test_cmds": [{"cmd": "scripts/a.sh", "label": "eval"}],
            "seeds": [42, 42],
        },
    ],
)
def test_validate_eval_summary_rejects_duplicate_config_matrix(config: dict):
    score_task = _load_score_task()

    error = score_task._validate_eval_summary([], config)

    assert error is not None
    assert "duplicate" in error


def test_validate_eval_summary_rejects_harness_fallback_marker(tmp_path: Path):
    score_task = _load_score_task()
    log = tmp_path / "eval.log"
    log.write_text("AGENT_LOAD_FALLBACK RuntimeError('broken')\nacc=0.9\n")
    config = {
        "test_cmds": [{"cmd": "scripts/eval.sh", "label": "eval"}],
        "seeds": [42],
    }
    summary = [{
        "label": "eval",
        "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
    }]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert "harness failure marker AGENT_LOAD_FALLBACK" in error


def test_validate_eval_summary_rejects_surface_error_marker(tmp_path: Path):
    score_task = _load_score_task()
    log = tmp_path / "eval.log"
    log.write_text("SURFACE_ERROR: bad solution; using random output\nacc=0.9\n")
    config = {
        "test_cmds": [{"cmd": "scripts/eval.sh", "label": "eval"}],
        "seeds": [42],
    }
    summary = [{
        "label": "eval",
        "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
    }]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert "harness failure marker SURFACE_ERROR" in error


@pytest.mark.parametrize(
    "failure_marker",
    [
        "verification failed",
        "training failed",
        "training has failed",
        "evaluation did not complete",
        "Traceback (most recent call last):",
        "[ERROR] RuntimeError: boom",
        "2026-07-11 12:34:56 ERROR ValueError: boom",
    ],
)
def test_validate_eval_summary_rejects_standard_failure_markers(
    tmp_path: Path,
    failure_marker: str,
):
    score_task = _load_score_task()
    log = tmp_path / "eval.log"
    log.write_text(f"{failure_marker}\nacc=0.99\n")
    config = {
        "test_cmds": [{"cmd": "scripts/eval.sh", "label": "eval"}],
        "seeds": [42],
    }
    summary = [{
        "label": "eval",
        "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
    }]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert "harness failure marker" in error


@pytest.mark.parametrize(
    "diagnostic",
    [
        "No training failure occurred",
        "training failure count: 0",
        "MeanSquaredError: 0.012",
        "RuntimeError: handled and recovered",
        "failed_samples=0",
        "failure_rate=0",
    ],
)
def test_failure_marker_does_not_match_diagnostic_identifiers(diagnostic: str):
    score_task = _load_score_task()

    assert score_task._failure_marker(f"{diagnostic}\nacc=0.99\n") is None


@pytest.mark.parametrize(
    ("line", "marker"),
    [
        ("TRAIN_ERROR -> reporting untrained model", "TRAIN_ERROR"),
        ("EVAL_FAILED reason=RuntimeError", "EVAL_FAILED"),
        (
            "PROMPT_CFG build_prompt failed (bad); using object-name fallback",
            "PROMPT_CFG build_prompt failed",
        ),
        ("TOKENSTRAT_CFG set_failed (bad)", "TOKENSTRAT_CFG set_failed"),
        ("LAYER_CFG surgery_failed (bad); running full depth", "LAYER_CFG surgery_failed"),
        ("PROMPT_TEMPLATE_ERROR contrastive-decoding: bad template", "PROMPT_TEMPLATE_ERROR"),
        ("TOKEN_SURFACE_ERROR token_id=7: bad token", "TOKEN_SURFACE_ERROR"),
        ("DETECTOR_ERROR fixed watermark detector: short text", "DETECTOR_ERROR"),
    ],
)
def test_validate_eval_summary_rejects_other_failure_markers(
    tmp_path: Path,
    line: str,
    marker: str,
):
    score_task = _load_score_task()
    log = tmp_path / "eval.log"
    log.write_text(f"{line}\nacc=0.9\n")
    config = {
        "test_cmds": [{"cmd": "scripts/eval.sh", "label": "eval"}],
        "seeds": [42],
    }
    summary = [{
        "label": "eval",
        "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
    }]

    error = score_task._validate_eval_summary(summary, config)

    assert error is not None
    assert f"harness failure marker {marker}" in error


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


def _write_score_fixture(tmp_path: Path, a_log: str, b_log: str) -> tuple[Path, Path]:
    task_meta = tmp_path / "meta"
    out_dir = tmp_path / "out"
    task_meta.mkdir()
    out_dir.mkdir()
    (task_meta / "config.json").write_text(json.dumps({
        "test_cmds": [
            {"cmd": "scripts/a.sh", "label": "a"},
            {"cmd": "scripts/b.sh", "label": "b"},
        ],
        "seeds": [42],
    }))
    (task_meta / "parser.py").write_text(
        "from mlsbench.agent.parsers import OutputParser, ParseResult\n"
        "class Parser(OutputParser):\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        metrics = {}\n"
        "        for token in raw_output.split():\n"
        "            parsed = self.parse_metric_assignment(token)\n"
        "            if parsed is not None:\n"
        "                metrics[parsed[0]] = parsed[1]\n"
        "        return ParseResult(feedback=raw_output, metrics=metrics)\n"
    )
    (task_meta / "score_spec.py").write_text(
        "from mlsbench.scoring.dsl import *\n"
        "term('a', col('a').sigmoid(ref=const(0.5), scale=0.1))\n"
        "term('b', col('b').sigmoid(ref=const(0.5), scale=0.1))\n"
        "setting('a', weighted_mean(('a', 1.0)))\n"
        "setting('b', weighted_mean(('b', 1.0)))\n"
        "task(gmean('a', 'b'))\n"
    )
    (task_meta / "leaderboard.csv").write_text(
        "timestamp,model,is_final,seed,a,b\n"
    )
    logs = []
    for label, content in (("a", a_log), ("b", b_log)):
        path = out_dir / f"{label}.log"
        path.write_text(content)
        logs.append({
            "label": label,
            "logs": [{"seed": 42, "rc": 0, "log": str(path), "elapsed": 1.0}],
        })
    (out_dir / "eval_summary.json").write_text(json.dumps(logs))
    return task_meta, out_dir


def test_score_rejects_pointcloud_metrics_after_standard_failure_trace(
    tmp_path: Path,
):
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    out_dir = tmp_path / "out"
    task_meta.mkdir()
    out_dir.mkdir()
    (task_meta / "config.json").write_text(json.dumps({
        "test_cmds": [{"cmd": "scripts/train_clean.sh", "label": "clean"}],
        "seeds": [42],
    }))
    (task_meta / "parser.py").write_text(
        "import re\n"
        "from mlsbench.agent.parsers import OutputParser, ParseResult\n"
        "class Parser(OutputParser):\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        metrics = {}\n"
        "        match = re.search(r'PN2_METRICS .*test_acc=([0-9.]+)', raw_output)\n"
        "        if match:\n"
        "            metrics[f'test_acc_{cmd_label}'] = float(match.group(1))\n"
        "        return ParseResult(feedback=raw_output, metrics=metrics)\n"
    )
    (task_meta / "score_spec.py").write_text(
        "from mlsbench.scoring.dsl import *\n"
        "term('acc', col('test_acc_clean').sigmoid(ref=const(0.5), scale=0.01))\n"
        "setting('clean', weighted_mean(('acc', 1.0)))\n"
        "task(gmean('clean'))\n"
    )
    (task_meta / "leaderboard.csv").write_text(
        "timestamp,model,is_final,seed,test_acc_clean\n"
    )
    log = out_dir / "clean.log"
    log.write_text(
        "verification failed\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: boom\n"
        "PN2_METRICS mode=group regime=clean test_acc=0.99 "
        "class_acc=0.98 n_train=9843 n_test=2468\n"
    )
    (out_dir / "eval_summary.json").write_text(json.dumps([{
        "label": "clean",
        "logs": [{"seed": 42, "rc": 0, "log": str(log)}],
    }]))
    reward = out_dir / "reward.txt"
    reward.write_text("0.99\n")

    rc = score_task.cmd_score(argparse.Namespace(
        task_meta=str(task_meta),
        out_dir=str(out_dir),
        reward_out=str(reward),
    ))

    assert rc == 0
    assert reward.read_text().strip() == "0"
    assert "harness failure marker" in (out_dir / "score_error.txt").read_text()
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "verification_result.json").exists()


def test_score_rejects_cross_setting_metric_fill(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, out_dir = _write_score_fixture(
        tmp_path,
        "a=0.8 b=0.8\n",
        "junk=1\n",
    )
    reward = out_dir / "reward.txt"
    reward.write_text("0.91\n")
    (out_dir / "metrics.json").write_text('{"reward": 0.91}\n')
    (out_dir / "verification_result.json").write_text(
        '{"status": "passed", "reward": 0.91}\n'
    )

    rc = score_task.cmd_score(argparse.Namespace(
        task_meta=str(task_meta),
        out_dir=str(out_dir),
        reward_out=str(reward),
    ))

    assert rc == 0
    assert reward.read_text().strip() == "0"
    assert "exactly one score setting" in (out_dir / "score_error.txt").read_text()
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "verification_result.json").exists()


def test_score_rejects_agent_atexit_metric_override(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, out_dir = _write_score_fixture(
        tmp_path,
        # The first record represents the trusted harness result. An editable
        # solution can register an atexit callback that prints the second record
        # after main() returns; the fixture parser is intentionally last-wins.
        "a=0.1\na=0.99\n",
        "b=0.8\n",
    )
    reward = out_dir / "reward.txt"

    rc = score_task.cmd_score(argparse.Namespace(
        task_meta=str(task_meta),
        out_dir=str(out_dir),
        reward_out=str(reward),
    ))

    assert rc == 0
    assert reward.read_text().strip() == "0"
    assert "duplicate authoritative metric" in (
        out_dir / "score_error.txt"
    ).read_text()
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "verification_result.json").exists()


def test_score_rejects_invalid_extra_parser_metric_before_filtering(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, out_dir = _write_score_fixture(
        tmp_path,
        "a=0.8\n",
        "b=0.8\n",
    )
    (task_meta / "parser.py").write_text(
        "from mlsbench.agent.parsers import OutputParser, ParseResult\n"
        "class Parser(OutputParser):\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        key = cmd_label\n"
        "        metrics = {key: 0.8}\n"
        "        if cmd_label == 'a':\n"
        "            metrics['ignored_std'] = float('nan')\n"
        "        return ParseResult(feedback=raw_output, metrics=metrics)\n"
    )
    reward = out_dir / "reward.txt"

    rc = score_task.cmd_score(argparse.Namespace(
        task_meta=str(task_meta),
        out_dir=str(out_dir),
        reward_out=str(reward),
    ))

    assert rc == 0
    assert reward.read_text().strip() == "0"
    assert "invalid parser metric" in (out_dir / "score_error.txt").read_text()
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "verification_result.json").exists()


def test_score_validation_error_publishes_zero(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, out_dir = _write_score_fixture(
        tmp_path,
        "a=0.8\n",
        "b=0.8\n",
    )
    (task_meta / "score_spec.py").write_text(
        "from mlsbench.scoring.dsl import *\n"
        "term('a', col('a').sigmoid(ref=const(0.5), scale=0.1))\n"
        "term('b', col('b').sigmoid(ref=const(0.5), scale=0.1))\n"
        "setting('a', weighted_mean(('a', 0.0)))\n"
        "setting('b', weighted_mean(('b', 1.0)))\n"
        "task(gmean('a', 'b'))\n"
    )
    reward = out_dir / "reward.txt"

    rc = score_task.cmd_score(argparse.Namespace(
        task_meta=str(task_meta),
        out_dir=str(out_dir),
        reward_out=str(reward),
    ))

    assert rc == 0
    assert reward.read_text().strip() == "0"
    assert "invalid score specification" in (out_dir / "score_error.txt").read_text()
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "verification_result.json").exists()


def test_score_hash_survives_mangrove_json_transport(tmp_path: Path):
    score_task = _load_score_task()
    task_meta, out_dir = _write_score_fixture(
        tmp_path,
        "a=0.8\n",
        "b=0.8\n",
    )
    reward = out_dir / "reward.txt"

    rc = score_task.cmd_score(argparse.Namespace(
        task_meta=str(task_meta),
        out_dir=str(out_dir),
        reward_out=str(reward),
    ))

    assert rc == 0
    assert float(reward.read_text()) > 0.0
    proof = json.loads((out_dir / "verification_result.json").read_text())
    assert proof["status"] == "passed"
    metrics_text = (out_dir / "metrics.json").read_text()
    assert proof["metrics_sha256"] == hashlib.sha256(metrics_text.encode()).hexdigest()
    mangrove_transport_text = json.dumps(
        json.loads(metrics_text),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    assert mangrove_transport_text == metrics_text
    assert proof["metrics_sha256"] == hashlib.sha256(
        mangrove_transport_text.encode()
    ).hexdigest()


def test_test_sh_wires_sanitized_meta_only_to_run_evals():
    script = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mls_bench"
        / "task-template"
        / "tests"
        / "test.sh"
    ).read_text()
    guard_block = script.split(" guard \\\n", 1)[1].split("guard_rc=$?", 1)[0]
    eval_block = script.split(" run-evals \\\n", 1)[1].split("|| _RUN_EVALS_RC", 1)[0]

    assert "--eval-task-meta" not in guard_block
    assert '--eval-task-meta "${EVAL_META}"' in eval_block


def test_test_sh_preserves_zero_until_success_proof_is_committed():
    script = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mls_bench"
        / "task-template"
        / "tests"
        / "test.sh"
    ).read_text()

    zero_init = script.index("printf '0\\n' > /logs/verifier/reward.txt")
    run_evals = script.index(' run-evals \\\n')
    score = script.index(' score \\\n')
    candidate = script.index('_CANDIDATE_REWARD="/logs/verifier/.reward.candidate"')
    proof = script.index('/logs/verifier/verification_result.json', score)
    publish = script.index(
        'mv -f -- "${_CANDIDATE_REWARD}" /logs/verifier/reward.txt'
    )
    commit = script.index("_VERIFICATION_COMMITTED=1", publish)
    assert zero_init < candidate < run_evals < score < proof < publish < commit
    score_block = script.split(' score \\\n', 1)[1].split("_PROOF_RC=0", 1)[0]
    assert '--reward-out "${_CANDIDATE_REWARD}"' in score_block
    assert "--reward-out /logs/verifier/reward.txt" not in score_block
    assert "canonical_proof_text" in script
    assert '_remove_reward_candidate' in script
    assert 'if [ "${_VERIFICATION_COMMITTED:-0}" -ne 1 ]; then' in script
    assert "trap _abort_verifier HUP INT TERM" in script
    assert "metrics_sha256" in script


def test_eval_preexec_drops_root_privileges(monkeypatch):
    if os.geteuid() != 0:
        pytest.skip("privilege-drop assertion requires a root test process")
    score_task = _load_score_task()
    monkeypatch.setenv("MLSBENCH_EVAL_UID", "65534")
    monkeypatch.setenv("MLSBENCH_EVAL_GID", "65534")

    proc = subprocess.run(
        ["id", "-u"],
        check=True,
        capture_output=True,
        text=True,
        preexec_fn=score_task._eval_preexec_fn(),
    )

    assert proc.stdout.strip() == "65534"


def _stage_verifier_shell_fixture(
    tmp_path: Path,
    *,
    pause_before_proof: bool = False,
) -> tuple[Path, Path, Path, Path, Path | None]:
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = tmp_path / "verifier-smoke"
    tests_root = fixture_root / "tests"
    logs_root = fixture_root / "logs"
    workspace = fixture_root / "workspace"
    fake_root = fixture_root / "root"
    solution_root = fixture_root / "solution"
    task_meta = tests_root / "meta"
    eval_scripts = tests_root / "eval" / "scripts"
    package = workspace / "pkg"
    pristine_package = task_meta / "pristine" / "pkg"
    for directory in (
        logs_root,
        package,
        pristine_package,
        eval_scripts,
        fake_root / ".cache",
        solution_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # pytest's per-run parents are normally mode 0700. The verifier's nobody
    # process must be able to traverse to the staged eval script and workspace.
    for parent in [fixture_root, *fixture_root.parents]:
        if parent == Path("/"):
            break
        parent.chmod(parent.stat().st_mode | 0o055)
        if parent == Path("/tmp"):
            break

    source = "def native_solution():\n    return 1\n"
    (package / "solution.py").write_text(source)
    (pristine_package / "solution.py").write_text(source)
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    config = {
        "use_cuda": False,
        "files": [{
            "filename": "pkg/solution.py",
            "edit": [{"start": -1, "end": -1}],
        }],
        "test_cmds": [{
            "cmd": "scripts/eval.sh",
            "label": "eval",
            "package": "pkg",
            "compute": 0,
            "time": "0:00:10",
        }],
        "seeds": [42],
    }
    (task_meta / "config.json").write_text(json.dumps(config))
    (task_meta / "pristine_manifest.json").write_text(json.dumps({
        "pkg/solution.py": source_sha,
    }))
    (task_meta / "task_id").write_text("smoke-task\n")
    (task_meta / "package").write_text("pkg\n")
    (task_meta / "workdir").write_text(f"{workspace}\n")
    (task_meta / "parser.py").write_text(
        "import re\n"
        "from mlsbench.agent.parsers import OutputParser, ParseResult\n"
        "class Parser(OutputParser):\n"
        "    def parse(self, cmd_label, raw_output):\n"
        "        match = re.search(r'score=([0-9.]+)', raw_output)\n"
        "        metrics = {'score': float(match.group(1))} if match else {}\n"
        "        return ParseResult(feedback=raw_output, metrics=metrics)\n"
    )
    (task_meta / "score_spec.py").write_text(
        "from mlsbench.scoring.dsl import *\n"
        "term('score', col('score').sigmoid(ref=const(0.0), scale=1.0))\n"
        "setting('eval', weighted_mean(('score', 1.0)))\n"
        "task(gmean('eval'))\n"
    )
    (task_meta / "leaderboard.csv").write_text(
        "timestamp,model,is_final,seed,score\n"
    )
    eval_script = eval_scripts / "eval.sh"
    eval_script.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "test \"$(id -u)\" = 65534\n"
        "test -r \"${TASK_DIR}/config.json\"\n"
        "test ! -r \"${TASK_DIR}/parser.py\"\n"
        "test ! -r \"${TASK_DIR}/score_spec.py\"\n"
        "test ! -r \"${TASK_DIR}/leaderboard.csv\"\n"
        "test ! -w solution.py\n"
        "mkdir -p \"${XDG_CACHE_HOME}\"\n"
        "printf 'home-ok\\n' > \"${HOME}/home-artifact.txt\"\n"
        "printf 'cache-ok\\n' > \"${XDG_CACHE_HOME}/cache-artifact.txt\"\n"
        "mkdir -p \"${OUTPUT_DIR}\"\n"
        "printf 'artifact-ok\\n' > \"${OUTPUT_DIR}/artifact.txt\"\n"
        "printf 'artifact_write=ok\\nscore=1.0\\n'\n"
    )
    eval_script.chmod(0o755)

    shutil.copytree(
        repo_root / "src" / "mlsbench",
        tests_root / "mlsbench_src" / "mlsbench",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    score_source = (
        repo_root
        / "harbor_adapter"
        / "src"
        / "mls_bench"
        / "task-template"
        / "tests"
        / "score_task.py"
    ).read_text()
    score_source = score_source.replace("/tests", str(tests_root))
    score_source = score_source.replace("/workspace", str(workspace))
    (tests_root / "score_task.py").write_text(score_source)

    test_source = (
        repo_root
        / "harbor_adapter"
        / "src"
        / "mls_bench"
        / "task-template"
        / "tests"
        / "test.sh"
    ).read_text()
    for original, replacement in (
        ("/tests", str(tests_root)),
        ("/logs", str(logs_root)),
        ("/workspace", str(workspace)),
        ("/solution", str(solution_root)),
        ("/root", str(fake_root)),
    ):
        test_source = test_source.replace(original, replacement)

    pause_marker = None
    if pause_before_proof:
        pause_marker = fixture_root / "score-stage-complete"
        pause_release = fixture_root / "release-proof-check"
        proof_boundary = "\n_PROOF_RC=1\n"
        assert test_source.count(proof_boundary) == 1
        test_source = test_source.replace(
            proof_boundary,
            (
                f'\nprintf "ready\\n" > "{pause_marker}"\n'
                f'while [ ! -e "{pause_release}" ]; do sleep 0.05; done\n'
                "\n_PROOF_RC=1\n"
            ),
            1,
        )
    test_script = tests_root / "test.sh"
    test_script.write_text(test_source)
    test_script.chmod(0o755)

    return (
        fixture_root,
        test_script,
        logs_root / "verifier",
        package / "solution.py",
        pause_marker,
    )


def _verifier_env() -> dict[str, str]:
    return {
        **os.environ,
        "MLSBENCH_VERIFIER_LOG_INTERVAL_SEC": "9999",
    }


def _wait_for_path(path: Path, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        returncode = proc.poll()
        if returncode is not None:
            stdout, stderr = proc.communicate()
            pytest.fail(
                f"verifier exited with rc={returncode} before {path} appeared\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        time.sleep(0.01)
    proc.kill()
    stdout, stderr = proc.communicate()
    pytest.fail(
        f"timed out waiting for {path}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )


def test_verifier_shell_end_to_end_with_unchanged_native_solution(tmp_path: Path):
    if os.geteuid() != 0:
        pytest.skip("full verifier smoke requires root to exercise uid drop")

    fixture_root, test_script, verifier_logs, solution_file, _ = (
        _stage_verifier_shell_fixture(tmp_path)
    )

    result = subprocess.run(
        ["bash", str(test_script)],
        cwd=str(fixture_root),
        env=_verifier_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert float((verifier_logs / "reward.txt").read_text()) > 0.0, (
        result.stdout + result.stderr
    )
    assert solution_file.read_text() == "def native_solution():\n    return 1\n"
    assert "artifact_write=ok" in (verifier_logs / "eval__seed42.log").read_text()
    proof = json.loads((verifier_logs / "verification_result.json").read_text())
    assert proof["status"] == "passed"
    assert proof["strict_fail_closed"] is True
    assert not (verifier_logs / ".reward.candidate").exists()


def test_verifier_sigkill_before_proof_keeps_public_reward_exact_zero(tmp_path: Path):
    if os.geteuid() != 0:
        pytest.skip("full verifier smoke requires root to exercise uid drop")

    fixture_root, test_script, verifier_logs, _solution_file, pause_marker = (
        _stage_verifier_shell_fixture(tmp_path, pause_before_proof=True)
    )
    assert pause_marker is not None
    proc = subprocess.Popen(
        ["bash", str(test_script)],
        cwd=str(fixture_root),
        env=_verifier_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    _wait_for_path(pause_marker, proc)
    candidate = verifier_logs / ".reward.candidate"
    assert float(candidate.read_text()) > 0.0
    assert (verifier_logs / "reward.txt").read_text() == "0\n"

    os.kill(proc.pid, signal.SIGKILL)
    proc.communicate(timeout=5)

    assert proc.returncode == -signal.SIGKILL
    assert (verifier_logs / "reward.txt").read_text() == "0\n"


def test_verifier_caught_signal_removes_reward_candidate(tmp_path: Path):
    if os.geteuid() != 0:
        pytest.skip("full verifier smoke requires root to exercise uid drop")

    fixture_root, test_script, verifier_logs, _solution_file, pause_marker = (
        _stage_verifier_shell_fixture(tmp_path, pause_before_proof=True)
    )
    assert pause_marker is not None
    proc = subprocess.Popen(
        ["bash", str(test_script)],
        cwd=str(fixture_root),
        env=_verifier_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    _wait_for_path(pause_marker, proc)
    candidate = verifier_logs / ".reward.candidate"
    assert float(candidate.read_text()) > 0.0

    os.kill(proc.pid, signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=5)

    assert proc.returncode == 0, stdout + stderr
    assert (verifier_logs / "reward.txt").read_text() == "0\n"
    assert not candidate.exists()


def test_verifier_invalid_proof_removes_candidate_and_keeps_zero(tmp_path: Path):
    if os.geteuid() != 0:
        pytest.skip("full verifier smoke requires root to exercise uid drop")

    fixture_root, test_script, verifier_logs, _solution_file, pause_marker = (
        _stage_verifier_shell_fixture(tmp_path, pause_before_proof=True)
    )
    assert pause_marker is not None
    proc = subprocess.Popen(
        ["bash", str(test_script)],
        cwd=str(fixture_root),
        env=_verifier_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    _wait_for_path(pause_marker, proc)
    candidate = verifier_logs / ".reward.candidate"
    assert float(candidate.read_text()) > 0.0
    proof_path = verifier_logs / "verification_result.json"
    proof_path.write_text(proof_path.read_text() + "\n")
    (fixture_root / "release-proof-check").touch()

    stdout, stderr = proc.communicate(timeout=5)

    assert proc.returncode == 0, stdout + stderr
    assert (verifier_logs / "reward.txt").read_text() == "0\n"
    assert not candidate.exists()
    assert not proof_path.exists()
    assert not (verifier_logs / "metrics.json").exists()
    assert "invalid success proof" in (verifier_logs / "score_error.txt").read_text()
