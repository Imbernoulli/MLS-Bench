from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SRC = ROOT / "harbor_adapter" / "src"
if str(ADAPTER_SRC) not in sys.path:
    sys.path.insert(0, str(ADAPTER_SRC))

from mls_bench.adapter import (  # noqa: E402
    MlsBenchRoot,
    _apply_ops_to_text,
    _baseline_sections,
    _config_with_shifted_edit_ranges,
    _load_ops_file,
    _read_sections,
    _stage_task_scaffold,
    _starting_workspace_text,
    build_task_context,
)


def test_cv_dbm_scheduler_edit_ranges_shift_to_post_setup_lines():
    mb = MlsBenchRoot(ROOT)
    ctx = build_task_context(mb, "cv-dbm-scheduler")

    effective = _config_with_shifted_edit_ranges(mb, ctx)
    entry = next(
        f for f in effective["files"]
        if f["filename"] == "dbim-codebase/ddbm/karras_diffusion.py"
    )

    assert entry["edit"] == [{"start": 310, "end": 320}]


def test_cv_dbm_scheduler_baseline_uses_shifted_post_mid_ranges():
    mb = MlsBenchRoot(ROOT)
    if mb.package_src("dbim-codebase") is None:
        return
    ctx = build_task_context(mb, "cv-dbm-scheduler")
    effective = _config_with_shifted_edit_ranges(mb, ctx)

    sections, warnings = _baseline_sections(mb, ctx, config=effective)

    assert sections
    assert any("Lines 310–" in section["code"] for section in sections)
    assert not any("Lines 301-" in section["code"] for section in sections)


def test_non_rigorous_task_does_not_render_baseline_sections():
    mb = MlsBenchRoot(ROOT)
    ctx = build_task_context(mb, "ts-short-term-forecast")

    sections, warnings = _baseline_sections(mb, ctx)

    assert sections == []
    assert warnings == []


def test_mode1_oracle_cmd_overrides_rendered(tmp_path: Path):
    from mls_bench.adapter import render_task

    mb_root = tmp_path / "mini"
    task_dir = mb_root / "tasks" / "mode1"
    scripts = task_dir / "scripts"
    scripts.mkdir(parents=True)
    (mb_root / "vendor" / "pkg").mkdir(parents=True)
    (mb_root / "vendor" / "pkg" / "main.py").write_text("VALUE = 1\n")
    (mb_root / "vendor" / "pkg_configs" / "pkg").mkdir(parents=True)
    (mb_root / "vendor" / "pkg_configs" / "pkg" / "config.json").write_text(
        json.dumps({"workdir": "/workspace", "use_cuda": False})
    )
    (mb_root / "vendor" / "packages.yaml").write_text("{}\n")
    for name in ("default.sh", "strong.sh"):
        (scripts / name).write_text("#!/bin/bash\nexit 0\n")
    config = {
        "test_cmds": [
            {"cmd": "scripts/default.sh", "label": "A", "compute": 0, "time": "0:01:00", "package": "pkg"}
        ],
        "baselines": {"strong": {"cmd": "scripts/strong.sh"}},
        "files": [{"filename": "pkg/main.py", "edit": [{"start": 1, "end": 1}]}],
    }
    (task_dir / "config.json").write_text(json.dumps(config))
    (task_dir / "task_description.md").write_text("Task\n")

    mb = MlsBenchRoot(mb_root)
    ctx = build_task_context(mb, "mode1")

    assert ctx.baseline_edit_ops == []
    assert ctx.oracle_cmd_overrides == [{"label": "", "cmd": "scripts/strong.sh"}]

    out = render_task(mb, ctx, tmp_path / "out", overwrite=True)
    solve_sh = (out / "solution" / "solve.sh").read_text()
    test_sh = (out / "tests" / "test.sh").read_text()
    overrides = json.loads((out / "solution" / "oracle_cmd_overrides.json").read_text())
    solution_token = (out / "solution" / "oracle_cmd_overrides.token").read_text()
    verifier_token = (out / "tests" / "meta" / "oracle_cmd_overrides.token").read_text()

    assert overrides == [{"label": "", "cmd": "scripts/strong.sh"}]
    assert solution_token == verifier_token
    assert "oracle_cmd_overrides.json" in solve_sh
    assert "--oracle-cmd-overrides" in solve_sh
    assert "oracle_cmd_overrides.token" in test_sh
    assert "--oracle-cmd-overrides" in test_sh


def test_rigorous_read_sections_skip_read_only_files():
    mb = MlsBenchRoot(ROOT)
    ctx = build_task_context(mb, "causal-observational-linear-gaussian")
    effective = _config_with_shifted_edit_ranges(mb, ctx)

    sections, _warnings = _read_sections(mb, ctx, config=effective)
    filenames = {section["filename"] for section in sections}

    assert "causal-learn/bench/custom_algorithm.py" in filenames
    assert "causal-learn/bench/run_eval.py" not in filenames
    assert "causal-learn/bench/data_gen.py" not in filenames


def test_hidden_test_cmds_are_rendered_in_instruction(tmp_path: Path):
    mb = MlsBenchRoot(ROOT)
    ctx = build_task_context(mb, "causal-observational-linear-gaussian")

    from mls_bench.adapter import render_task

    out = render_task(mb, ctx, tmp_path, overwrite=True)
    instruction = (out / "instruction.md").read_text()

    assert "ER20-Noisy" in instruction


def test_ops_loader_supports_next_builtin():
    ops = _load_ops_file(ROOT / "vendor/pkg_configs/lm-evaluation-harness/pre_edit.py")

    assert ops


def test_ops_loader_isolates_custom_template_imports(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory, value in ((first, "FIRST"), (second, "SECOND")):
        (directory / "custom_template.py").write_text(f'_TEMPLATE = "{value}"\n')
        (directory / "mid_edit.py").write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.append(str(Path(__file__).parent))\n"
            "from custom_template import _TEMPLATE\n"
            'OPS = [{"op": "create", "file": "pkg/file.py", "content": _TEMPLATE}]\n'
        )

    assert _load_ops_file(first / "mid_edit.py")[0]["content"] == "FIRST"
    assert _load_ops_file(second / "mid_edit.py")[0]["content"] == "SECOND"


def test_apply_ops_delete_line_form_and_replace_to_eof():
    text = "a\nb\nc\n"
    deleted = _apply_ops_to_text(
        text,
        [{"op": "delete", "file": "pkg/f.py", "line": 2}],
        "pkg/f.py",
    )
    replaced = _apply_ops_to_text(
        text,
        [{"op": "replace", "file": "pkg/f.py", "start_line": 2, "end_line": -1, "content": "z\n"}],
        "pkg/f.py",
    )

    assert deleted == "a\nc\n"
    assert replaced == "a\nz\n"


def test_package_lookup_is_case_and_separator_insensitive():
    mb = MlsBenchRoot(ROOT)

    assert mb.package_src("causal_learn") == ROOT / "vendor" / "causal-learn"


def test_stage_task_scaffold_copies_secondary_package(tmp_path: Path):
    mb_root = tmp_path / "mini"
    task_dir = mb_root / "tasks" / "multi"
    (task_dir / "edits").mkdir(parents=True)
    (mb_root / "vendor" / "primary").mkdir(parents=True)
    (mb_root / "vendor" / "secondary").mkdir(parents=True)
    (mb_root / "vendor" / "secondary" / "lib.py").write_text("VALUE = 1\n")
    (mb_root / "vendor" / "pkg_configs" / "primary").mkdir(parents=True)
    (mb_root / "vendor" / "pkg_configs" / "primary" / "config.json").write_text(
        json.dumps({"workdir": "/workspace"})
    )
    (mb_root / "vendor" / "packages.yaml").write_text("{}\n")
    config = {
        "test_cmds": [
            {"cmd": "scripts/a.sh", "label": "a", "compute": 1, "time": "0:01:00", "package": "primary"},
            {"cmd": "scripts/b.sh", "label": "b", "compute": 1, "time": "0:01:00", "package": "secondary"},
        ],
        "files": [{"filename": "primary/main.py", "edit": [{"start": -1, "end": -1}]}],
    }
    (task_dir / "config.json").write_text(json.dumps(config))
    (task_dir / "task_description.md").write_text("Task\n")

    mb = MlsBenchRoot(mb_root)
    ctx = build_task_context(mb, "multi")
    created = _stage_task_scaffold(mb, ctx, tmp_path / "scaffold")

    assert "secondary" in created
    assert (tmp_path / "scaffold" / "secondary" / "lib.py").read_text() == "VALUE = 1\n"


def test_starting_workspace_text_applies_mid_edit_create(tmp_path: Path):
    mb_root = tmp_path / "mini"
    task_dir = mb_root / "tasks" / "created"
    edits = task_dir / "edits"
    edits.mkdir(parents=True)
    (mb_root / "vendor" / "pkg").mkdir(parents=True)
    (mb_root / "vendor" / "pkg_configs" / "pkg").mkdir(parents=True)
    (mb_root / "vendor" / "pkg_configs" / "pkg" / "config.json").write_text("{}")
    (mb_root / "vendor" / "packages.yaml").write_text("{}\n")
    config = {
        "test_cmds": [
            {"cmd": "scripts/a.sh", "label": "a", "compute": 1, "time": "0:01:00", "package": "pkg"}
        ],
        "files": [{"filename": "pkg/new.py", "edit": [{"start": 1, "end": 1}]}],
    }
    (task_dir / "config.json").write_text(json.dumps(config))
    (task_dir / "task_description.md").write_text("Task\n")
    (edits / "mid_edit.py").write_text(
        'OPS = [{"op": "create", "file": "pkg/new.py", "content": "x = 1"}]\n'
    )

    mb = MlsBenchRoot(mb_root)
    ctx = build_task_context(mb, "created")

    assert _starting_workspace_text(mb, ctx, "pkg/new.py") == "x = 1\n"


def test_baseline_sections_use_post_mid_edit_workspace_text(tmp_path: Path):
    mb_root = tmp_path / "mini"
    task_dir = mb_root / "tasks" / "baseline-mid"
    edits = task_dir / "edits"
    edits.mkdir(parents=True)
    (mb_root / "vendor" / "pkg").mkdir(parents=True)
    (mb_root / "vendor" / "pkg" / "file.py").write_text("a\nOLD\nc\n")
    (mb_root / "vendor" / "pkg_configs" / "pkg").mkdir(parents=True)
    (mb_root / "vendor" / "pkg_configs" / "pkg" / "config.json").write_text("{}")
    (mb_root / "vendor" / "packages.yaml").write_text("{}\n")
    config = {
        "rigorous_codebase": True,
        "test_cmds": [
            {"cmd": "scripts/a.sh", "label": "a", "compute": 1, "time": "0:01:00", "package": "pkg"}
        ],
        "baselines": {"base": {"edit_ops": "edits/base.py"}},
        "files": [
            {
                "filename": "pkg/file.py",
                "read": [{"start": 2, "end": 2}],
                "edit": [{"start": 2, "end": 2}],
            }
        ],
    }
    (task_dir / "config.json").write_text(json.dumps(config))
    (task_dir / "task_description.md").write_text("Task\n")
    (edits / "mid_edit.py").write_text(
        'OPS = [{"op": "replace", "file": "pkg/file.py", "start_line": 2, "end_line": 2, "content": "MID\\n"}]\n'
    )
    (edits / "base.py").write_text(
        'OPS = [{"op": "replace", "file": "pkg/file.py", "start_line": 2, "end_line": 2, "content": "BASE\\n"}]\n'
    )

    mb = MlsBenchRoot(mb_root)
    ctx = build_task_context(mb, "baseline-mid")
    sections, warnings = _baseline_sections(mb, ctx)

    assert warnings == []
    assert len(sections) == 1
    assert "Lines 2–2:" in sections[0]["code"]
    assert "BASE" in sections[0]["code"]
    assert "OLD" not in sections[0]["code"]


def test_stage_task_scaffold_rejects_drifted_pre_edit_line_replace(tmp_path: Path):
    """Regression: if a pre_edit/mid_edit line-range replace cuts a multi-line
    Python statement (because the vendored source drifted off the line numbers
    the op file targets), the scaffold must fail loudly instead of writing
    broken Python that only surfaces at agent runtime.

    Simulates the upstream Time-Series-Library drift where the colleague's
    vendor copy had `--use_dtw` spanning lines 113-114 (instead of 109-110 in
    the pinned commit). pre_edit replaces line 114 only, deleting the
    `help='...')` continuation and leaving an unclosed `(`.
    """
    import pytest

    mb_root = tmp_path / "mini"
    task_dir = mb_root / "tasks" / "drift"
    edits = task_dir / "edits"
    edits.mkdir(parents=True)
    (mb_root / "vendor" / "pkg").mkdir(parents=True)
    (mb_root / "vendor" / "pkg" / "main.py").write_text(
        # Two-line argparse call (the `(` on line 1, the `)` on line 2)
        # mimics --use_dtw in the drifted upstream.
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--use_dtw', action='store_true', default=False,\n"
        "                    help='enable dtw metric')\n"
    )
    (mb_root / "vendor" / "pkg_configs" / "pkg").mkdir(parents=True)
    (mb_root / "vendor" / "pkg_configs" / "pkg" / "config.json").write_text("{}")
    (mb_root / "vendor" / "packages.yaml").write_text("{}\n")
    config = {
        "test_cmds": [
            {"cmd": "x.sh", "label": "x", "compute": 1, "time": "0:01:00", "package": "pkg"}
        ],
        "files": [{"filename": "pkg/main.py", "edit": [{"start": -1, "end": -1}]}],
    }
    (task_dir / "config.json").write_text(json.dumps(config))
    (task_dir / "task_description.md").write_text("Task\n")
    # Replace line 4 (the help= continuation) with a new add_argument —
    # the `(` from line 3 is now unclosed.
    (edits / "mid_edit.py").write_text(
        "OPS = ["
        '{"op": "replace", "file": "pkg/main.py", '
        '"start_line": 4, "end_line": 4, '
        '"content": "parser.add_argument(\'--seed\', type=int, default=42)"}]\n'
    )

    mb = MlsBenchRoot(mb_root)
    ctx = build_task_context(mb, "drift")
    with pytest.raises(RuntimeError, match=r"invalid Python in 'pkg/main\.py'"):
        _stage_task_scaffold(mb, ctx, tmp_path / "scaffold")


def test_stage_task_scaffold_accepts_valid_line_replace(tmp_path: Path):
    """Sanity counterpart: line-range replace that does NOT break syntax must
    not trigger the syntax guard."""
    mb_root = tmp_path / "mini"
    task_dir = mb_root / "tasks" / "ok"
    edits = task_dir / "edits"
    edits.mkdir(parents=True)
    (mb_root / "vendor" / "pkg").mkdir(parents=True)
    (mb_root / "vendor" / "pkg" / "main.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--seed', type=int, default=2)\n"
    )
    (mb_root / "vendor" / "pkg_configs" / "pkg").mkdir(parents=True)
    (mb_root / "vendor" / "pkg_configs" / "pkg" / "config.json").write_text("{}")
    (mb_root / "vendor" / "packages.yaml").write_text("{}\n")
    config = {
        "test_cmds": [
            {"cmd": "x.sh", "label": "x", "compute": 1, "time": "0:01:00", "package": "pkg"}
        ],
        "files": [{"filename": "pkg/main.py", "edit": [{"start": -1, "end": -1}]}],
    }
    (task_dir / "config.json").write_text(json.dumps(config))
    (task_dir / "task_description.md").write_text("Task\n")
    (edits / "mid_edit.py").write_text(
        "OPS = ["
        '{"op": "replace", "file": "pkg/main.py", '
        '"start_line": 3, "end_line": 3, '
        '"content": "parser.add_argument(\'--seed\', type=int, default=42)"}]\n'
    )

    mb = MlsBenchRoot(mb_root)
    ctx = build_task_context(mb, "ok")
    created = _stage_task_scaffold(mb, ctx, tmp_path / "scaffold")
    assert "pkg/main.py" in created
