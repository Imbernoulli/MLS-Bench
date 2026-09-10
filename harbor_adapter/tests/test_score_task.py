from __future__ import annotations

import importlib.util
import hashlib
import json
import re
import sys
import argparse
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


def test_created_files_are_neither_flagged_nor_removed(tmp_path: Path):
    """Creating files is not a violation, and the guard must not delete them.

    Deleting them between guard and eval silently broke agents that split an
    implementation into a helper module: the import died with no visible cause.
    Both a workspace-root file and one inside the guarded package prefix must
    survive.
    """
    score_task = _load_score_task()
    task_meta, pristine, workspace, violation = _write_guard_fixture(
        tmp_path,
        allow_create=False,
    )
    (workspace / "sitecustomize.py").write_text("print('scratch')\n")
    (workspace / "pkg" / "helper.py").write_text("VALUE = 1\n")

    rc = score_task.cmd_guard(argparse.Namespace(
        task_meta=str(task_meta),
        pristine=str(pristine),
        workspace=str(workspace),
        violation_out=str(violation),
    ))

    assert rc == 0, violation.read_text() if violation.exists() else ""
    assert not violation.exists()
    assert (workspace / "sitecustomize.py").exists()
    assert (workspace / "pkg" / "helper.py").exists()


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


def test_edit_guard_allows_single_range_replacement_with_repeated_suffix_line(tmp_path: Path):
    """SequenceMatcher must not steal a repeated fixed suffix line.

    The pristine editable line and fixed suffix are intentionally identical.
    Replacing the editable line leaves the suffix byte-for-byte intact, but a
    global diff aligns the suffix with the editable occurrence and reports the
    actual suffix as deleted.
    """
    score_task = _load_score_task()
    pristine = tmp_path / "pristine.py"
    current = tmp_path / "current.py"

    pristine.write_text("header\nrepeated\nrepeated\n")
    current.write_text("header\nreplacement\nrepeated\n")

    ranges = [score_task.EditRange(2, 2)]
    ok, reason = score_task._check_editable_only(pristine, current, ranges)

    assert ok, reason


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


def test_h200_override_materializes_command_compute_and_env(monkeypatch):
    score_task = _load_score_task()
    monkeypatch.setenv("MLSBENCH_GPU_TYPE", "H200")
    config = {
        "test_cmds": [
            {
                "cmd": "scripts/h100.sh",
                "compute": 4.0,
                "env": {"BASE": "yes"},
                "h200": {
                    "cmd": "scripts/h200.sh",
                    "compute": 2.0,
                    "env": {"BATCH_SIZE": "96"},
                },
            },
            {"cmd": "scripts/eval.sh", "compute": 1.0},
        ]
    }

    materialized = score_task._effective_test_cmds(config)
    assert materialized == [
        {
            "cmd": "scripts/h200.sh",
            "compute": 2.0,
            "env": {"BASE": "yes", "BATCH_SIZE": "96"},
        },
        {"cmd": "scripts/eval.sh", "compute": 1.0},
    ]
    # The source config is never mutated; this matters because budget checks
    # and later eval groups may reuse it.
    assert "h200" in config["test_cmds"][0]


def test_h200_override_is_not_selected_on_h100(monkeypatch):
    score_task = _load_score_task()
    monkeypatch.setenv("MLSBENCH_GPU_TYPE", "H100")
    config = {
        "test_cmds": [
            {
                "cmd": "scripts/h100.sh",
                "compute": 4.0,
                "h200": {"cmd": "scripts/h200.sh", "compute": 2.0},
            }
        ]
    }
    materialized = score_task._effective_test_cmds(config)
    assert materialized == config["test_cmds"]


@pytest.mark.parametrize(
    "task_id, expected_compute, expected_env",
    [
        (
            "llm-pretrain-attention",
            2.0,
            {"BATCH_SIZE": "64", "GRAD_ACCUM": "8"},
        ),
        (
            "llm-rl-advantage",
            1.0,
            {
                "TP_SIZE": "1",
                "MAX_TOKEN_LEN_PER_GPU": "20480",
                "GPU_MEM_UTIL": "0.5",
            },
        ),
    ],
)
def test_h200_materialization_preserves_native_microbatch_settings(
    monkeypatch, task_id, expected_compute, expected_env
):
    """The Daytona verifier consumes the repository's original H200 metadata."""
    score_task = _load_score_task()
    monkeypatch.setenv("MLSBENCH_GPU_TYPE", "H200")
    config = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "tasks"
            / task_id
            / "config.json"
        ).read_text()
    )
    materialized = score_task._effective_test_cmds(config)
    assert len(materialized) == len(config["test_cmds"])
    entry = materialized[0]
    assert entry["compute"] == expected_compute
    assert entry.get("env") == expected_env
    # H200 metadata remains in the source dictionary; materialization is a
    # verifier-local copy and cannot rewrite the native task configuration.
    assert "h200" in config["test_cmds"][0]


def test_all_native_pretrain_h200_profiles_preserve_effective_microbatch():
    """Every published pretraining H200 profile keeps its macro-batch fixed."""
    root = Path(__file__).resolve().parents[2] / "tasks"
    for config_path in sorted((root).glob("llm-pretrain-*/config.json")):
        config = json.loads(config_path.read_text())
        entry = next(
            (tc for tc in config.get("test_cmds", []) if isinstance(tc.get("h200"), dict)),
            None,
        )
        if entry is None:
            continue
        override_env = entry["h200"].get("env", {})
        assert {"BATCH_SIZE", "GRAD_ACCUM"} <= set(override_env)
        script = config_path.parent / "scripts" / "gpt_345m.sh"
        script_text = script.read_text()
        defaults = {}
        for key in ("BATCH_SIZE", "GRAD_ACCUM"):
            match = re.search(rf"{key}=\$\{{{key}:-([^}}]+)\}}", script_text)
            assert match, f"missing H100 {key} default in {script}"
            defaults[key] = int(match.group(1))
        assert int(override_env["BATCH_SIZE"]) * int(override_env["GRAD_ACCUM"]) == (
            defaults["BATCH_SIZE"] * defaults["GRAD_ACCUM"]
        )


def test_all_native_rl_h200_profiles_include_runtime_memory_settings():
    root = Path(__file__).resolve().parents[2] / "tasks"
    for config_path in sorted((root).glob("llm-rl-*/config.json")):
        config = json.loads(config_path.read_text())
        entry = next(tc for tc in config["test_cmds"] if "h200" in tc)
        override_env = entry["h200"].get("env", {})
        assert {"TP_SIZE", "MAX_TOKEN_LEN_PER_GPU", "GPU_MEM_UTIL"} <= set(override_env)


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


def test_budget_scratch_config_reflects_oracle_override(tmp_path: Path):
    """budget_check.py reads its hyperparameters from TASK_DIR/config.json's
    test_cmds. For an oracle run the eval cmd is overridden to the strongest
    baseline's cmd; that override must be written into the budget scratch config
    too, else the check counts the agent model under the ORIGINAL (large)
    eval-script hyperparameters and wrongly rejects the oracle (all-zero TS
    oracle). Regression for that gap."""
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    task_meta.mkdir()
    (task_meta / "config.json").write_text(json.dumps({
        "test_cmds": [{"label": "PSM", "cmd": "scripts/psm.sh"}],
        "baselines": {"timesnet": {"cmd": "scripts/timesnet.sh"}},
        "files": [{"filename": "pkg/x.py", "edit": [{"start": 1, "end": 1}]}],
    }))
    (task_meta / "budget_check.py").write_text("# noop\n")

    override = [{"label": "PSM", "cmd": "scripts/timesnet.sh"}]
    scratch = tmp_path / "scratch"
    task_dir = score_task._copy_task_meta_for_budget(task_meta, scratch, override)
    cfg = json.loads((task_dir / "config.json").read_text())
    assert cfg["test_cmds"] == override, "override must reach the budget config"
    assert "baselines" in cfg and "files" in cfg, "other config fields preserved"

    # No override (agent run) → original test_cmds preserved unchanged.
    scratch2 = tmp_path / "scratch2"
    task_dir2 = score_task._copy_task_meta_for_budget(task_meta, scratch2)
    cfg2 = json.loads((task_dir2 / "config.json").read_text())
    assert cfg2["test_cmds"][0]["cmd"] == "scripts/psm.sh"


def test_budget_scratch_keeps_the_native_layout_so_mid_edit_finds_holdout(tmp_path: Path):
    """optimization-nas's edits/mid_edit.py loads the verifier-only generator
    from ``parents[3] / "holdout" / <task_id>`` (native snapshots
    ``tasks/<task>`` beside ``holdout/<task>``).  A flat budget scratch dir
    resolved that to ``/holdout/optimization-nas/dgp.py``, so the budget check
    crashed and all 15 evals were rc=1 (Daytona oracle sweep, 2026-09-08).
    The scratch copy must keep the layout and ship the bundle's
    ``tests/eval/_inputgen/holdout/<task_id>`` next to it."""
    score_task = _load_score_task()
    task_meta = tmp_path / "meta"
    (task_meta / "edits").mkdir(parents=True)
    (task_meta / "task_id").write_text("optimization-nas\n")
    (task_meta / "config.json").write_text(json.dumps({"test_cmds": []}))
    (task_meta / "budget_check.py").write_text("# noop\n")
    (task_meta / "edits" / "mid_edit.py").write_text(
        "from pathlib import Path\n"
        "_HERE = Path(__file__).resolve()\n"
        "TASK_DIR = _HERE.parents[1]\n"
        "DGP = _HERE.parents[3] / 'holdout' / 'optimization-nas' / 'dgp.py'\n"
    )
    eval_root = tmp_path / "eval"
    holdout = eval_root / "_inputgen" / "holdout" / "optimization-nas"
    holdout.mkdir(parents=True)
    (holdout / "dgp.py").write_text("TABLES = 1\n")
    (holdout / "nb201_tables.json.gz").write_bytes(b"gz")

    scratch = tmp_path / "scratch"
    task_dir = score_task._copy_task_meta_for_budget(
        task_meta, scratch, None, eval_root=eval_root
    )
    assert task_dir == scratch / "tasks" / "optimization-nas"
    assert (task_dir / "budget_check.py").exists()
    ns: dict = {}
    exec((task_dir / "edits" / "mid_edit.py").read_text(),
         {"__file__": str(task_dir / "edits" / "mid_edit.py")}, ns)
    assert ns["TASK_DIR"] == task_dir
    assert ns["DGP"].exists(), "mid_edit's parents[3]/holdout resolution must hit the staged copy"
    assert (scratch / "holdout" / "optimization-nas" / "nb201_tables.json.gz").exists()

    # Bundles without an _inputgen holdout (the other 49 budget_check tasks)
    # stage exactly as before, just one level deeper.
    plain = tmp_path / "plain"
    task_dir_plain = score_task._copy_task_meta_for_budget(task_meta, plain, None, eval_root=tmp_path / "nope")
    assert task_dir_plain == plain / "tasks" / "optimization-nas"
    assert not (plain / "holdout").exists()


def test_infer_reserved_gpu_count_balances_waves_at_the_cap():
    st = _load_score_task()

    # 3 whole-GPU test_cmds x 3 seeds in one group = 9 concurrent single-GPU jobs.
    config = {
        "use_cuda": True,
        "seeds": [42, 123, 456],
        "test_cmds": [
            {"label": "a", "group": 1, "compute": 1.0},
            {"label": "b", "group": 1, "compute": 1.0},
            {"label": "c", "group": 1, "compute": 1.0},
        ],
    }
    # 8 would run waves of 8 and 1; 3 splits the 9 jobs into three even waves.
    assert st._infer_reserved_gpu_count(config) == 3

    # Under the cap: untouched.
    assert st._infer_reserved_gpu_count(
        {"use_cuda": True, "test_cmds": [{"label": "a", "group": 1, "compute": 4.0}]}
    ) == 4

    # A single job larger than the cap cannot be split into waves.
    assert st._infer_reserved_gpu_count(
        {"use_cuda": True, "test_cmds": [{"label": "a", "group": 1, "compute": 16.0}]}
    ) == 16

    # CPU-only stays at zero.
    assert st._infer_reserved_gpu_count({"use_cuda": False, "test_cmds": []}) == 0


def test_partition_group_gpu_batches_serializes_group_overflow():
    st = _load_score_task()

    def task(label, compute):
        return {"entry": {"tc": {"label": label, "compute": compute}}, "seed": 42}

    devices = [str(i) for i in range(st.MAX_PARALLEL_GPUS)]

    # cv-dbm-sampler shape: 3 x 4-GPU jobs in one group on 8 GPUs -> 2 waves.
    batches = st._partition_group_gpu_batches(
        [task("a", 4.0), task("b", 4.0), task("c", 4.0)], devices
    )
    assert [len(tasks) for tasks, _ in batches] == [2, 1]
    assert [assignments for _, assignments in batches] == [
        ["0,1,2,3", "4,5,6,7"],
        ["0,1,2,3"],
    ]

    # rl-intrinsic-exploration shape: 9 single-GPU jobs on 8 GPUs -> 2 waves.
    batches = st._partition_group_gpu_batches(
        [task(f"j{i}", 1.0) for i in range(9)], devices
    )
    assert [len(tasks) for tasks, _ in batches] == [8, 1]
def test_rendered_bundles_match_the_verifier_template():
    """Every rendered task ships its own copy of the verifier.

    `harbor/tasks/*/tests/{score_task.py,test.sh}` are plain copies of the
    task-template files — no Jinja substitution happens on them. A fix landed
    in the template therefore reaches exactly zero agents until the copies are
    re-synced, which is silent and easy to miss (it is how the single-range
    guard fix initially shipped as a no-op). Fail loudly on drift.
    """
    repo_root = Path(__file__).resolve().parents[2]
    tasks_roots = [repo_root / "harbor" / v for v in ("tasks-docker", "tasks-daytona", "tasks-modal")]
    tasks_roots = [r for r in tasks_roots if r.is_dir()]
    if not tasks_roots:
        return  # adapter-only checkout: nothing rendered to compare against

    template_dir = Path(__file__).resolve().parents[1] / "src" / "mls_bench" / "task-template" / "tests"
    for name in ("score_task.py", "test.sh"):
        expected = (template_dir / name).read_bytes()
        drifted = [
            f"{tasks_root.name}/{p.parents[1].name}"
            for tasks_root in tasks_roots
            for p in sorted(tasks_root.glob(f"*/tests/{name}"))
            if p.read_bytes() != expected
        ]
        assert not drifted, (
            f"{len(drifted)} rendered bundle(s) have a {name} that differs from "
            f"task-template/tests/{name}: {', '.join(drifted[:5])}"
            f"{' …' if len(drifted) > 5 else ''}. Re-sync the copies — a template-only "
            "change does not reach any shipped task."
        )


def test_thread_budget_follows_the_cgroup_quota_not_nproc(monkeypatch, tmp_path):
    """A container reports every host core; only the cgroup quota is real.

    Sizing thread pools from ``os.cpu_count()`` inside a 4-CPU cgroup on a
    384-core host starts ~192 BLAS threads and makes a fixed matmul benchmark
    13x slower, which is what pushed CPU-bound evals past their deadline.
    """
    module = _load_score_task()
    monkeypatch.delenv("MLSBENCH_LOCAL_THREADS", raising=False)
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _pid: set(range(384)))

    cgroup_v2 = tmp_path / "cpu.max"
    cgroup_v2.write_text("400000 100000\n")
    real_read = module.Path.read_text

    def fake_read(self, *args, **kwargs):
        if self.as_posix() == "/sys/fs/cgroup/cpu.max":
            return cgroup_v2.read_text()
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(module.Path, "read_text", fake_read)
    assert module._available_cpus() == 4
    # A wave of three concurrent commands splits the quota instead of each
    # claiming all of it.
    assert module._thread_budget() == 4
    assert module._thread_budget(3) == 1
    assert module._thread_budget(0) == 4


def test_thread_budget_is_unlimited_without_a_quota(monkeypatch):
    """Outside a limited cgroup the affinity mask is the honest answer."""
    module = _load_score_task()
    monkeypatch.delenv("MLSBENCH_LOCAL_THREADS", raising=False)
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _pid: set(range(12)))
    monkeypatch.setattr(module, "_cgroup_cpu_quota", lambda: None)
    assert module._available_cpus() == 12
    assert module._thread_budget(4) == 3


def test_mlsbench_local_threads_overrides_the_budget(monkeypatch):
    """Same escape hatch native runs use (`mlsbench/cli.py::local_thread_limit`)."""
    module = _load_score_task()
    monkeypatch.setenv("MLSBENCH_LOCAL_THREADS", "7")
    monkeypatch.setattr(module, "_available_cpus", lambda: 64)
    assert module._thread_budget(8) == 7
    monkeypatch.setenv("MLSBENCH_LOCAL_THREADS", "not-a-number")
    assert module._thread_budget(8) == 8


def test_eval_env_pins_thread_pools_but_lets_a_task_override(tmp_path, monkeypatch):
    """The budget lands in the eval subprocess; task/package env still wins."""
    module = _load_score_task()
    monkeypatch.setenv("OMP_NUM_THREADS", "384")
    task_meta = tmp_path / "meta"
    task_meta.mkdir()
    (task_meta / "package").write_text("demo\n")
    (task_meta / "task_id").write_text("demo-task\n")
    (task_meta / "package_envs.json").write_text(json.dumps({"demo": {}}))

    env = module._eval_env(
        task_meta=task_meta,
        out_dir=tmp_path / "out",
        workspace_root=tmp_path / "ws",
        pkg_dir=tmp_path / "ws" / "demo",
        tc={"label": "a"},
        seed=42,
        threads=5,
    )
    assert env["OMP_NUM_THREADS"] == "5"
    assert env["MKL_NUM_THREADS"] == env["BLIS_NUM_THREADS"] == "5"

    override = module._eval_env(
        task_meta=task_meta,
        out_dir=tmp_path / "out",
        workspace_root=tmp_path / "ws",
        pkg_dir=tmp_path / "ws" / "demo",
        tc={"label": "a", "env": {"OMP_NUM_THREADS": "1"}},
        seed=42,
        threads=5,
    )
    assert override["OMP_NUM_THREADS"] == "1"


def test_eval_failures_name_timeouts_and_their_deadline():
    """A reader must be able to tell a harness timeout from an empty model run."""
    module = _load_score_task()
    config = {"test_cmds": [
        {"label": "Weather", "cmd": "w.sh", "group": 1, "compute": 0.33, "time": "00:59:00"},
        {"label": "ECL", "cmd": "e.sh", "group": 1, "compute": 0.33, "time": "00:59:00"},
        {"label": "ETTh1", "cmd": "t.sh", "group": 1, "compute": 0.33, "time": "00:59:00"},
    ]}
    summary = [
        {"label": "Weather", "logs": [{"seed": 42, "rc": 124, "elapsed": 3870.2}]},
        {"label": "ECL", "logs": [{"seed": 42, "rc": 1, "elapsed": 12.0}]},
        {"label": "ETTh1", "logs": [{"seed": 42, "rc": 0, "elapsed": 49.0}]},
    ]
    lines = module._eval_failures(summary, config)
    assert lines == [
        "Weather seed 42: timed out after 3870s (its own deadline is 3840s incl. 300s grace; a wave shares its slowest member's)",
        "ECL seed 42: exited rc=1 after 12s",
    ]
    assert module._eval_failures([{"label": "ETTh1", "logs": [{"seed": 42, "rc": 0, "elapsed": 1.0}]}], config) == []



def test_eval_task_dir_exposes_the_staged_task_dir_extras(tmp_path: Path):
    """llm-kv-adaptive-quantization's eval resolves the task dir by
    `task_description.md` (native bind-mounts tasks/<t>/ at /workspace/_task);
    the Harbor eval dir carried only scripts/data/third_party and every eval
    died with "Unable to locate task directory" (2026-09-09, both clouds).
    Whatever the adapter stages under tests/meta/evaltask/ must land at the
    eval dir root, while the scoring metadata stays out."""
    score_task = _load_score_task()
    meta = tmp_path / "meta"
    (meta / "scripts").mkdir(parents=True); (meta / "scripts" / "a.sh").write_text("#!/bin/bash\n")
    (meta / "evaltask" / "benchmarks").mkdir(parents=True)
    (meta / "evaltask" / "task_description.md").write_text("# t\n")
    (meta / "evaltask" / "benchmarks" / "spec.json").write_text("{}")
    (meta / "parser.py").write_text("# secret\n")
    d = score_task._build_eval_task_dir(meta)
    assert (d / "task_description.md").read_text() == "# t\n"
    assert (d / "benchmarks" / "spec.json").exists() and (d / "scripts" / "a.sh").exists()
    assert not (d / "parser.py").exists() and not (d / "evaltask").exists()


def test_eval_task_dir_carries_the_marker_the_llm_kv_evals_resolve(tmp_path: Path):
    """`_build_eval_task_dir` must put `task_description.md` at its root.

    `tasks/llm-kv-*/edits/custom_template.py` resolves its task directory by
    trying, in order, its own directory, its parent, `$MLSBENCH_TASK_DIR`,
    `$TASK_DIR`, `cwd/_task` and `cwd/../_task`, and taking the first that
    holds `task_description.md`. `$MLSBENCH_TASK_DIR` points at the real
    task_meta, which deliberately does not hold it, so resolution succeeds
    only through the `_task` symlink onto this directory — the two llm-kv
    tasks scored 0 with "Unable to locate task directory" before the marker
    was staged."""
    import os

    score_task = _load_score_task()
    meta = tmp_path / "meta"
    (meta / "evaltask").mkdir(parents=True)
    (meta / "scripts").mkdir()
    (meta / "evaltask" / "task_description.md").write_text("# task\n")
    (meta / "scripts" / "run.sh").write_text("echo hi\n")
    (meta / "config.json").write_text("{}")

    eval_dir = score_task._build_eval_task_dir(meta)

    assert (eval_dir / "task_description.md").is_file()
    assert (eval_dir / "scripts" / "run.sh").is_file()
    assert not (eval_dir / "config.json").exists()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    os.symlink(eval_dir, workspace / "_task", target_is_directory=True)

    def resolve(here: Path, env: dict) -> Path:
        for candidate in (here, here.parent, Path(env.get("MLSBENCH_TASK_DIR", "")),
                          Path(env.get("TASK_DIR", "")), workspace / "_task",
                          workspace.parent / "_task"):
            if not candidate or str(candidate) == ".":
                continue
            resolved = candidate.resolve()
            if (resolved / "task_description.md").exists():
                return resolved
        raise FileNotFoundError(here)

    assert resolve(workspace / "pkg" / "custom_template.py",
                   {"MLSBENCH_TASK_DIR": str(meta)}) == eval_dir.resolve()
