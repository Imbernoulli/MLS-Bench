from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
import types
from collections import Counter
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
    spec = importlib.util.spec_from_file_location("score_task_scheduler_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakePopen:
    records: list[dict] = []
    next_pid = 1000

    def __init__(
        self,
        cmd,
        *,
        cwd=None,
        env=None,
        stdout=None,
        stderr=None,
        start_new_session=False,
    ):
        assert start_new_session is True
        self.cmd = cmd
        self.cwd = cwd
        self.env = env or {}
        self.stdout = stdout
        self.stderr = stderr
        self.pid = FakePopen.next_pid
        FakePopen.next_pid += 1
        self.returncode = None
        FakePopen.records.append({
            "event": "start",
            "cmd": cmd,
            "label": self.env.get("ENV"),
            "seed": self.env.get("SEED"),
            "cuda": self.env.get("CUDA_VISIBLE_DEVICES"),
        })

    def poll(self):
        FakePopen.records.append({
            "event": "poll",
            "label": self.env.get("ENV"),
            "seed": self.env.get("SEED"),
        })
        self.returncode = 0
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _write_task(tmp_path: Path, config: dict, gpu_count: int):
    task_meta = tmp_path / "meta"
    eval_root = tmp_path / "eval"
    workspace = tmp_path / "workspace"
    task_meta.mkdir()
    eval_root.mkdir()
    workspace.mkdir()
    (task_meta / "config.json").write_text(json.dumps(config))
    default_pkg = config["test_cmds"][0].get("package", "pkg")
    (task_meta / "package").write_text(default_pkg + "\n")
    (task_meta / "task_id").write_text("scheduler-task\n")
    (task_meta / "gpu_count").write_text(f"{gpu_count}\n")

    packages = {default_pkg}
    for tc in config["test_cmds"]:
        script = eval_root / tc["cmd"]
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("printf 'ok\\n'\n")
        packages.add(tc.get("package", default_pkg))
    for package in packages:
        (workspace / package).mkdir(parents=True, exist_ok=True)

    return argparse.Namespace(
        task_meta=str(task_meta),
        workspace=str(workspace),
        eval_root=str(eval_root),
        out_dir=str(tmp_path / "out"),
    )


def _install_fake_popen(monkeypatch, score_task):
    FakePopen.records = []
    FakePopen.next_pid = 1000
    monkeypatch.setattr(score_task.subprocess, "Popen", FakePopen)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")


def _start_records():
    return [record for record in FakePopen.records if record["event"] == "start"]


def _start_batches():
    batches = []
    current = []
    for record in FakePopen.records:
        if record["event"] == "start":
            current.append(record)
        elif record["event"] == "poll" and current:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def _read_summary(args):
    return json.loads((Path(args.out_dir) / "eval_summary.json").read_text())


def test_tdmpc2_seeds_and_entries_bin_pack_in_one_wave(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/walker.sh", "label": "walker", "group": 1, "compute": 0.33, "time": "0:01:00", "package": "tdmpc2"},
                {"cmd": "scripts/cheetah.sh", "label": "cheetah", "group": 1, "compute": 0.33, "time": "0:01:00", "package": "tdmpc2"},
                {"cmd": "scripts/cartpole.sh", "label": "cartpole", "group": 1, "compute": 0.33, "time": "0:01:00", "package": "tdmpc2", "hidden": True},
            ],
            "seeds": [42, 123, 456],
        },
        gpu_count=3,
    )

    assert score_task.cmd_run_evals(args) == 0

    starts = _start_records()
    assert len(starts) == 9
    assert [len(batch) for batch in _start_batches()] == [9]
    assert Counter(record["cuda"] for record in starts) == {"0": 3, "1": 3, "2": 3}
    assert [(record["label"], record["seed"]) for record in starts] == [
        ("walker", "42"),
        ("walker", "123"),
        ("walker", "456"),
        ("cheetah", "42"),
        ("cheetah", "123"),
        ("cheetah", "456"),
        ("cartpole", "42"),
        ("cartpole", "123"),
        ("cartpole", "456"),
    ]
    batches = _start_batches()
    assert Counter(record["cuda"] for record in batches[0]) == {"0": 3, "1": 3, "2": 3}


def test_llm_pretrain_attention_runs_groups_sequentially(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/gpt.sh", "label": "gpt-345m", "group": 1, "compute": 4.0, "time": "0:01:00", "package": "nanoGPT"},
                {"cmd": "scripts/lm_eval.sh", "label": "lm-eval-345m", "group": 2, "compute": 1.0, "time": "0:01:00", "package": "lm-evaluation-harness", "hidden": True},
            ],
            "seeds": [42],
        },
        gpu_count=4,
    )

    assert score_task.cmd_run_evals(args) == 0

    starts = _start_records()
    assert [(record["label"], record["cuda"]) for record in starts] == [
        ("gpt-345m", "0,1,2,3"),
        ("lm-eval-345m", "0"),
    ]
    first_lm_start = next(i for i, record in enumerate(FakePopen.records) if record.get("label") == "lm-eval-345m" and record["event"] == "start")
    assert any(record["event"] == "poll" and record.get("label") == "gpt-345m" for record in FakePopen.records[:first_lm_start])


def test_ai4sci_fractional_group_packs_onto_two_gpus(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/bbbp.sh", "label": "BBBP", "group": 1, "compute": 0.5, "time": "0:01:00", "package": "Uni-Mol"},
                {"cmd": "scripts/bace.sh", "label": "BACE", "group": 1, "compute": 0.5, "time": "0:01:00", "package": "Uni-Mol"},
                {"cmd": "scripts/tox21.sh", "label": "Tox21", "group": 1, "compute": 0.5, "time": "0:01:00", "package": "Uni-Mol", "hidden": True},
            ],
            "seeds": [42],
        },
        gpu_count=2,
    )

    assert score_task.cmd_run_evals(args) == 0

    starts = _start_records()
    assert len(starts) == 3
    assert [len(batch) for batch in _start_batches()] == [3]
    cuda_values = [record["cuda"] for record in starts]
    assert set(cuda_values) == {"0", "1"}


def test_multi_seed_fractional_flat_pack(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/a.sh", "label": "a", "group": 1, "compute": 0.5, "time": "0:01:00", "package": "pkg"},
                {"cmd": "scripts/b.sh", "label": "b", "group": 1, "compute": 0.5, "time": "0:01:00", "package": "pkg"},
            ],
            "seeds": [42, 123],
        },
        gpu_count=2,
    )

    assert score_task.cmd_run_evals(args) == 0

    starts = _start_records()
    assert len(starts) == 4
    assert [len(batch) for batch in _start_batches()] == [4]
    assert Counter(record["cuda"] for record in starts) == {"0": 2, "1": 2}
    assert [(record["label"], record["seed"]) for record in starts] == [
        ("a", "42"),
        ("a", "123"),
        ("b", "42"),
        ("b", "123"),
    ]


def test_cv_3dgs_separate_groups_run_sequentially(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/garden.sh", "label": "garden", "group": 1, "compute": 1.0, "time": "0:01:00", "package": "gsplat"},
                {"cmd": "scripts/bicycle.sh", "label": "bicycle", "group": 2, "compute": 1.0, "time": "0:01:00", "package": "gsplat"},
                {"cmd": "scripts/bonsai.sh", "label": "bonsai", "group": 3, "compute": 1.0, "time": "0:01:00", "package": "gsplat"},
                {"cmd": "scripts/stump.sh", "label": "stump", "group": 4, "compute": 1.0, "time": "0:01:00", "package": "gsplat", "hidden": True},
            ],
            "seeds": [42],
        },
        gpu_count=1,
    )

    assert score_task.cmd_run_evals(args) == 0

    starts = _start_records()
    assert [(record["label"], record["cuda"]) for record in starts] == [
        ("garden", "0"),
        ("bicycle", "0"),
        ("bonsai", "0"),
        ("stump", "0"),
    ]
    assert [len(batch) for batch in _start_batches()] == [1, 1, 1, 1]


def test_budget_check_failure_does_not_abort_group(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/first.sh", "label": "first", "group": 1, "compute": 0.33, "time": "0:01:00", "package": "pkg"},
                {"cmd": "scripts/fail.sh", "label": "budget-fail", "group": 1, "compute": 0.33, "time": "0:01:00", "package": "pkg"},
                {"cmd": "scripts/third.sh", "label": "third", "group": 1, "compute": 0.33, "time": "0:01:00", "package": "pkg"},
            ],
            "seeds": [42],
        },
        gpu_count=1,
    )

    def fake_budget_check(**kwargs):
        if kwargs["label"] == "budget-fail":
            return {"rc": 1, "log": str(tmp_path / "budget-fail.log")}
        return {"rc": 0, "log": str(tmp_path / "budget-ok.log")}

    monkeypatch.setattr(score_task, "_run_budget_check", fake_budget_check)

    assert score_task.cmd_run_evals(args) == 0

    starts = _start_records()
    assert [(record["label"], record["cuda"]) for record in starts] == [
        ("first", "0"),
        ("third", "0"),
    ]
    summary = _read_summary(args)
    assert summary[0]["logs"][0]["rc"] == 0
    assert summary[1]["logs"][0]["rc"] != 0
    assert summary[2]["logs"][0]["rc"] == 0


def test_wave_timeout_kills_stuck_processes_with_sigterm_then_sigkill(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)

    class HangingPopen(FakePopen):
        def poll(self):
            FakePopen.records.append({
                "event": "poll",
                "label": self.env.get("ENV"),
                "seed": self.env.get("SEED"),
            })
            return None

        def wait(self, timeout=None):
            return None

    FakePopen.records = []
    FakePopen.next_pid = 1000
    monkeypatch.setattr(score_task.subprocess, "Popen", HangingPopen)
    kill_calls = []

    def fake_killpg(pid, sig):
        kill_calls.append((pid, sig))

    now = {"value": 0.0}

    def fake_time():
        now["value"] += 1000.0
        return now["value"]

    monkeypatch.setattr(score_task.os, "killpg", fake_killpg)
    monkeypatch.setattr(
        score_task,
        "time",
        types.SimpleNamespace(time=fake_time, sleep=lambda _seconds: None),
    )
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/one.sh", "label": "one", "group": 1, "compute": 1.0, "time": "00:00:01", "package": "pkg"},
                {"cmd": "scripts/two.sh", "label": "two", "group": 1, "compute": 1.0, "time": "00:00:01", "package": "pkg"},
            ],
            "seeds": [42],
        },
        gpu_count=2,
    )

    assert score_task.cmd_run_evals(args) == 0

    starts = _start_records()
    assert [record["label"] for record in starts] == ["one", "two"]
    assert kill_calls == [
        (1000, signal.SIGTERM),
        (1000, signal.SIGKILL),
        (1001, signal.SIGTERM),
        (1001, signal.SIGKILL),
    ]
    summary = _read_summary(args)
    assert [entry["logs"][0]["rc"] for entry in summary] == [124, 124]
    for entry in summary:
        assert "[TIMEOUT]" in Path(entry["logs"][0]["log"]).read_text()


def test_infer_reserved_gpu_count_bin_packs_fractional_jobs(tmp_path: Path):
    """Regression for Codex #6: 9 × 0.4 GPU jobs need 5 bins, not ceil(3.6)=4.

    Each bin can fit at most floor(1/0.4)=2 jobs at 0.4, so 9 such jobs need
    ceil(9/2)=5 bins. The previous `ceil(sum)` heuristic underestimated this,
    causing the rendered task.toml + docker-compose to under-reserve GPUs.
    """
    score_task = _load_score_task()
    cfg = {
        "use_cuda": True,
        "seeds": [42],
        "test_cmds": [
            {"cmd": f"x{i}.sh", "label": f"x{i}", "group": 1,
             "compute": 0.4, "time": "0:01:00", "package": "pkg"}
            for i in range(9)
        ],
    }
    assert score_task._infer_reserved_gpu_count(cfg) == 5


def test_infer_reserved_gpu_count_multiplies_by_seeds(tmp_path: Path):
    """Per-seed fractional jobs accumulate: 1 entry × 0.5 GPU × 4 seeds = 2 bins."""
    score_task = _load_score_task()
    cfg = {
        "use_cuda": True,
        "seeds": [1, 2, 3, 4],
        "test_cmds": [
            {"cmd": "x.sh", "label": "x", "group": 1,
             "compute": 0.5, "time": "0:01:00", "package": "pkg"},
        ],
    }
    assert score_task._infer_reserved_gpu_count(cfg) == 2


def test_oversized_compute_is_rejected_before_launch(monkeypatch, tmp_path: Path):
    score_task = _load_score_task()
    _install_fake_popen(monkeypatch, score_task)
    args = _write_task(
        tmp_path,
        {
            "test_cmds": [
                {"cmd": "scripts/oversized.sh", "label": "oversized", "group": 1, "compute": 8.0, "time": "0:01:00", "package": "pkg"},
            ],
            "seeds": [42],
        },
        gpu_count=4,
    )

    assert score_task.cmd_run_evals(args) == 0

    assert _start_records() == []
    summary = _read_summary(args)
    record = summary[0]["logs"][0]
    assert record["rc"] == 125
    assert "requires 8 GPUs but only 4 reserved/visible" in Path(record["log"]).read_text()
