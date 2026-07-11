from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
VENDOR = ROOT / "vendor" / "inr-signal-fitting"

ACTIVE = {
    "inr-activation": ("activation.py", {"family"}, "relu_mlp"),
    "inr-eikonal-reg": ("jacobian_reg.py", {"weight"}, "jacobian_high"),
    "inr-encoding-dim": ("encoding_dim.py", {"num_freqs"}, "nfreq2"),
    "inr-fourier-frequency": ("frequency.py", {"sigma"}, "sigma_low"),
    "inr-hash-grid": (
        "hash_grid.py",
        {"n_levels", "base_res", "finest_res"},
        "collapsed",
    ),
    "inr-lr-schedule": ("lr_schedule.py", {"lr", "schedule"}, "lr_big_const"),
    "inr-network-depth": ("depth.py", {"n_layers"}, "depth1"),
    "inr-network-width": ("width.py", {"hidden"}, "width8"),
    "inr-per-layer-w0": ("per_layer_w0.py", {"first", "hidden"}, "w0_3"),
    "inr-skip-connections": ("skip.py", {"skip_at"}, "noskip"),
}
DROPPED = {"inr-coord-transform", "inr-init-scheme"}
CALIBRATED = {"inr-fourier-frequency"}
PENDING_ZERO = set(ACTIVE) - CALIBRATED


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common():
    return _load("ship_inr_common", VENDOR / "common.py")


@pytest.fixture(scope="module")
def harness(common):
    previous = sys.modules.get("common")
    sys.modules["common"] = common
    try:
        yield _load("ship_inr_harness", VENDOR / "harness.py")
    finally:
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous


def _task_dirs() -> list[Path]:
    return sorted(path for path in TASKS.glob("inr-*") if path.is_dir())


def test_exact_active_and_dropped_question_sets():
    from mlsbench.scoring.spec import load_score_spec

    task_dirs = _task_dirs()
    assert len(task_dirs) == 12
    active = set()
    dropped = set()
    for task_dir in task_dirs:
        config = json.loads((task_dir / "config.json").read_text())
        (dropped if config.get("_dropped") else active).add(task_dir.name)
    assert active == set(ACTIVE)
    assert dropped == DROPPED
    for name in DROPPED:
        assert (TASKS / name / "DROPPED.md").exists()
        config = json.loads((TASKS / name / "config.json").read_text())
        assert config["test_cmds"] == []
        assert config["files"] == []
        spec = load_score_spec(TASKS / name)
        assert spec is not None
        assert spec.terms == {}
        assert spec.settings == {}
        with (TASKS / name / "leaderboard.csv").open(newline="") as handle:
            assert list(csv.DictReader(handle)) == []


def test_active_questions_have_distinct_surfaces_and_serial_settings():
    filenames = set()
    schemas = set()
    for name, (filename, keys, _native) in ACTIVE.items():
        config = json.loads((TASKS / name / "config.json").read_text())
        assert config["seeds"] == [0]
        assert [entry["label"] for entry in config["test_cmds"]] == [
            "low",
            "medium",
            "high",
        ]
        assert [entry["group"] for entry in config["test_cmds"]] == [1, 2, 3]
        assert all(entry["compute"] == 1.0 for entry in config["test_cmds"])
        assert config["verifier_only_package_files"] == [
            "inr-signal-fitting/common.py",
            "inr-signal-fitting/harness.py",
        ]
        assert config["agent_data_prune"] == ["/data/inr-signal-fitting"]
        assert config["verifier_data_deps"] == [
            {
                "host_path": "{data_root}/inr-signal-fitting",
                "dest": "data/inr-signal-fitting",
                "required": True,
            }
        ]
        editable = [entry for entry in config["files"] if entry.get("edit")]
        assert len(editable) == 1
        assert editable[0]["filename"] == f"inr-signal-fitting/solution/{filename}"
        filenames.add(filename)
        schemas.add(tuple(sorted(keys)))
    assert len(filenames) == len(ACTIVE)
    assert len(schemas) == len(ACTIVE)


def test_active_edit_ranges_match_surface_config_regions():
    for name, (filename, _keys, _native) in ACTIVE.items():
        source = (VENDOR / "solution" / filename).read_text().splitlines()
        start = source.index("def surface_config():") + 1
        end = start + 1
        config = json.loads((TASKS / name / "config.json").read_text())
        declared = config["files"][0]["edit"][0]
        assert (declared["start"], declared["end"]) == (start, end), name
        assert source[start - 2] == "# EDITABLE REGION"
        assert source[end] == "# END EDITABLE REGION"


def test_native_surface_configs_are_static_json_literals_and_match_schemas(common):
    for _name, (filename, keys, _native) in ACTIVE.items():
        plan = common.load_surface_config(str(VENDOR / "solution" / filename))
        assert isinstance(plan, dict)
        assert set(plan) == keys
        json.dumps(plan, allow_nan=False)


@pytest.mark.parametrize(
    ("surface", "plan", "error"),
    [
        ("activation", {"family": "unknown"}, ValueError),
        ("jacobian_reg", {"weight": float("nan")}, ValueError),
        ("encoding_dim", {"num_freqs": True}, TypeError),
        ("frequency", {"sigma": 0.0}, ValueError),
        (
            "hash_grid",
            {"n_levels": 1, "base_res": 4, "finest_res": 8},
            ValueError,
        ),
        ("lr_schedule", {"lr": 0.001, "schedule": "mystery"}, ValueError),
        ("depth", {"n_layers": 0}, ValueError),
        ("width", {"hidden": 0}, ValueError),
        ("per_layer_w0", {"first": 30.0, "hidden": float("inf")}, ValueError),
        ("skip", {"skip_at": 8}, ValueError),
    ],
)
def test_invalid_single_axis_configs_fail_before_training(common, surface, plan, error):
    coords = torch.zeros(4, 2)
    target = torch.zeros(4, 3)
    with pytest.raises(error):
        common.fit_surface(surface, plan, coords, target, torch.device("cpu"))


def test_surface_config_loader_fails_closed(common, tmp_path):
    missing = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError):
        common.load_surface_config(str(missing))

    broken = tmp_path / "broken.py"
    broken.write_text("def surface_config():\n    raise RuntimeError('broken')\n")
    with pytest.raises(ValueError, match="exactly one return"):
        common.load_surface_config(str(broken))

    nonfinite = tmp_path / "nonfinite.py"
    nonfinite.write_text("def surface_config():\n    return {'value': float('nan')}\n")
    with pytest.raises(ValueError, match="finite JSON literal"):
        common.load_surface_config(str(nonfinite))


def test_surface_config_is_never_executed(common, tmp_path, capsys):
    sentinel = tmp_path / "tampered"
    malicious = tmp_path / "malicious.py"
    malicious.write_text(
        "def surface_config():\n"
        f"    open({str(sentinel)!r}, 'w').write('changed')\n"
        "    print('INR_METRICS signal=low psnr=99 res=256 elapsed=0')\n"
        "    return {'hidden': 8}\n"
    )
    with pytest.raises(ValueError, match="exactly one return"):
        common.load_surface_config(str(malicious))
    assert not sentinel.exists()
    assert "INR_METRICS" not in capsys.readouterr().out


def test_training_and_metric_numerics_fail_closed(common):
    coords = torch.zeros(4, 2)
    target = torch.zeros(4, 3)
    common.train_inr(
        torch.nn.Linear(2, 3),
        coords,
        target,
        torch.device("cpu"),
        steps=1,
        log_every=1,
    )

    class NonFiniteModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, value):
            return value.new_full((value.shape[0], 3), float("nan")) * self.weight

    with pytest.raises(ValueError, match="non-finite"):
        common.train_inr(
            NonFiniteModel(),
            coords,
            target,
            torch.device("cpu"),
            steps=1,
            log_every=1,
        )
    with pytest.raises(ValueError, match="finite"):
        common.psnr_db(torch.full_like(target, float("nan")), target)
    with pytest.raises(ValueError, match="share shape"):
        common.psnr_db(torch.zeros(4, 2), target)


def test_harness_does_not_emit_metrics_after_surface_failure(
    harness, monkeypatch, capsys
):
    coords = torch.zeros(8, 2)
    target = torch.zeros(8, 3)
    monkeypatch.setattr(harness.common, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        harness.common, "load_signal", lambda _name: (coords, target, 256)
    )
    monkeypatch.setattr(harness.common, "load_surface_config", lambda _path: {})
    monkeypatch.setattr(
        harness.common,
        "fit_surface",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("surface failed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["harness.py", "--solution", "solution/width.py", "--signal", "low"],
    )
    with pytest.raises(RuntimeError, match="surface failed"):
        harness.main()
    assert "INR_METRICS" not in capsys.readouterr().out


def test_harness_rejects_nonfinite_final_prediction(harness, monkeypatch, capsys):
    coords = torch.zeros(8, 2)
    target = torch.zeros(8, 3)
    monkeypatch.setattr(harness.common, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        harness.common, "load_signal", lambda _name: (coords, target, 256)
    )
    monkeypatch.setattr(harness.common, "load_surface_config", lambda _path: {})
    monkeypatch.setattr(
        harness.common,
        "fit_surface",
        lambda *_args: lambda value: torch.full(
            (value.shape[0], 3), float("nan")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["harness.py", "--solution", "solution/width.py", "--signal", "low"],
    )
    with pytest.raises(ValueError, match="non-finite"):
        harness.main()
    assert "INR_METRICS" not in capsys.readouterr().out


def test_all_baseline_edits_apply_and_compile():
    for task_dir in _task_dirs():
        config = json.loads((task_dir / "config.json").read_text())
        if config.get("_dropped"):
            for baseline in config.get("baselines", {}).values():
                assert (task_dir / baseline["edit_ops"]).is_file()
            continue
        declared = config["files"][0]
        allowed = declared["edit"][0]
        source_path = ROOT / "vendor" / declared["filename"]
        source_lines = source_path.read_text().splitlines()
        for name, baseline in config.get("baselines", {}).items():
            edit_path = task_dir / baseline["edit_ops"]
            scope = {"__file__": str(edit_path)}
            exec(compile(edit_path.read_text(), str(edit_path), "exec"), scope)
            assert len(scope["OPS"]) == 1, edit_path
            op = scope["OPS"][0]
            assert op["op"] == "replace"
            assert op["file"] == declared["filename"]
            start, end = int(op["start_line"]), int(op["end_line"])
            assert allowed["start"] <= start <= end <= allowed["end"], edit_path
            replacement = str(op["content"]).splitlines()
            candidate = "\n".join(
                source_lines[: start - 1] + replacement + source_lines[end:]
            ) + "\n"
            compile(candidate, f"{task_dir.name}:{name}", "exec")
            if task_dir.name in ACTIVE:
                namespace: dict = {}
                exec(candidate, namespace)
                plan = namespace["surface_config"]()
                assert set(plan) == ACTIVE[task_dir.name][1]
                json.dumps(plan, allow_nan=False)


def test_only_frequency_has_a_positive_score_mapping_and_anchor_rows():
    from mlsbench.scoring.spec import AnchorRef, load_score_spec, validate_score_spec

    for task_dir in _task_dirs():
        spec = load_score_spec(task_dir)
        assert spec is not None
        config = json.loads((task_dir / "config.json").read_text())
        with (task_dir / "leaderboard.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        if task_dir.name not in CALIBRATED:
            assert spec.terms == {}
            assert spec.settings == {}
            assert rows == []
            if not config.get("_dropped"):
                assert config["calibration_status"] == (
                    "pending_exact_zero_task_specific_anchors"
                )
            continue

        assert config["calibration_status"] == (
            "measured_task_specific_full_official_anchors"
        )
        assert set(spec.settings) == {"low", "medium", "high"}
        metrics = [term.metric for term in spec.terms.values()]
        assert not validate_score_spec(spec, metrics), task_dir.name
        for term in spec.terms.values():
            assert term.scale is not None and math.isfinite(term.scale) and term.scale > 0
            if isinstance(term.ref, AnchorRef) and term.ref.kind == "const":
                assert math.isfinite(term.ref.value)

        assert rows
        present = {
            row["model"].removeprefix("baseline:")
            for row in rows
            if row.get("model", "").startswith("baseline:")
        }
        assert set(config.get("baselines", {})) <= present
        for row in rows:
            for metric in ("psnr_low", "psnr_medium", "psnr_high"):
                assert math.isfinite(float(row[metric])), (task_dir.name, metric)


def test_frequency_measured_references_use_consistent_calibration():
    from mlsbench.agent.leaderboard import Leaderboard
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import load_expanded_spec, score_record

    task_dir = TASKS / "inr-fourier-frequency"
    rows = Leaderboard(task_dir / "leaderboard.csv").all_records()
    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
    assert spec is not None
    config = json.loads((task_dir / "config.json").read_text())
    scores = {}
    for baseline in config["baselines"]:
        row = next(
            item for item in rows if item.get("model") == f"baseline:{baseline}"
        )
        scores[baseline] = score_record(spec, row, anchors)
    assert scores["sigma_low"] == pytest.approx(0.1, abs=1e-9)
    assert scores["sigma_tuned"] == pytest.approx(0.5, abs=1e-9)


def test_missing_verifier_metrics_score_exactly_zero():
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import load_expanded_spec, score_record_details

    task_dir = TASKS / "inr-fourier-frequency"
    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
    assert spec is not None
    score, _settings, valid = score_record_details(spec, {"seed": 0}, anchors)
    assert score == 0.0
    assert valid is False


@pytest.mark.parametrize("task_dir", _task_dirs(), ids=lambda path: path.name)
def test_parsers_require_complete_unique_consistent_runtime_proof(task_dir):
    parser = _load(
        f"ship_inr_parser_{task_dir.name}".replace("-", "_"),
        task_dir / "parser.py",
    ).Parser()
    assert parser.parse("low", "").metrics == {}
    assert parser.parse("low", "verification failed\nscore=0.9\n").metrics == {}
    valid_lines = [
        "DATA_INFO signal=low res=256 n_coords=65536 dev=cuda",
        "STEP_METRICS label=frequency step=750/2000 loss=0.1 psnr=20.0",
        "STEP_METRICS label=frequency step=2000/2000 loss=0.01 psnr=30.0",
        "INR_METRICS signal=low psnr=30.5 res=256 elapsed=25.0",
        "INR_DONE signal=low n_coords=65536 steps=2000 seed=0",
    ]
    assert parser.parse("low", "\n".join(valid_lines)).metrics == {
        "psnr_low": 30.5
    }

    invalid_outputs = [
        valid_lines[:2] + valid_lines[3:],
        valid_lines[:-1],
        valid_lines + [valid_lines[3]],
        [valid_lines[0].replace("signal=low", "signal=high")] + valid_lines[1:],
        valid_lines[:3]
        + [valid_lines[3].replace("res=256", "res=64")]
        + valid_lines[4:],
        valid_lines[:3]
        + [valid_lines[3].replace("psnr=30.5", "psnr=nan")]
        + valid_lines[4:],
        valid_lines + ["INR_METRICS signal=low psnr=forged res=256 elapsed=0"],
        valid_lines + ["RuntimeError: failure after metric emission"],
        valid_lines + ["late output"],
        valid_lines[:-1] + ["INR_FAILED late", valid_lines[-1]],
        valid_lines[:-1] + ["INR_FAILURE: late", valid_lines[-1]],
        [valid_lines[2], valid_lines[0], valid_lines[1], *valid_lines[3:]],
    ]
    for lines in invalid_outputs:
        assert parser.parse("low", "\n".join(lines)).metrics == {}


def test_agent_visible_text_is_neutral_and_runtime_has_no_fallback():
    forbidden = re.compile(
        r"\b(?:weak|strong|sota|winner|winning|best|worst|baseline|ordering|"
        r"recommended|well-tuned|degenerate)\b|falls? back|degrades? to|"
        r"reference implementation",
        re.IGNORECASE,
    )
    visible = [VENDOR / "common.py", VENDOR / "harness.py"]
    visible += [VENDOR / "solution" / filename for filename, _keys, _native in ACTIVE.values()]
    visible += [TASKS / name / "task_description.md" for name in ACTIVE]
    for path in visible:
        assert not forbidden.search(path.read_text()), path
    for name in ACTIVE:
        description = (TASKS / name / "task_description.md").read_text()
        assert "Schema:" in description, name
        assert "surface_config" in description, name

    harness_source = (VENDOR / "harness.py").read_text()
    assert "_constant_predictor" not in harness_source
    assert "except Exception" not in harness_source
    assert "INR_METRICS" in harness_source


def test_package_reference_assets_are_declared_private():
    package = json.loads(
        (ROOT / "vendor/pkg_configs/inr-signal-fitting/config.json").read_text()
    )
    assert set(package["agent_pruned_files"]) == {
        "apply_real_anchors.py",
        "gen_tasks.py",
        "new_anchors.json",
        "real_anchors.json",
        "sweep_anchors.py",
        "sweep_new_anchors.py",
        "validate_all.py",
        "validate_oracle.py",
    }
    assert package["mangrove_base_image"] == (
        "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
        "mlsbench-harbor-inr-signal-fitting@sha256:"
        "674af0eed3edb5fe7f916a225564abe2c1cbebb6a1fe9c9659bc45ba39d4bbe3"
    )


def test_staged_signal_digests_match_verifier_contract(common):
    data_root = ROOT / "vendor" / "data" / "inr-signal-fitting"
    if not data_root.exists():
        pytest.skip("staged verifier data is intentionally absent from this worktree")
    for name in ("low", "medium", "high"):
        with np.load(data_root / f"{name}.npz", allow_pickle=False) as data:
            coords = data["coords"]
            target = data["target"]
        digest = lambda value: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        assert digest(coords) == common._COORDS_SHA256
        assert digest(target) == common._TARGET_SHA256[name]


def test_measurement_manifest_matches_active_baseline_edits(common):
    previous = sys.modules.get("common")
    sys.modules["common"] = common
    try:
        sweep = _load("ship_inr_sweep", VENDOR / "sweep_new_anchors.py")
    finally:
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    assert set(sweep.CANDIDATES) == set(ACTIVE)
    assert set(sweep.SURFACES) == set(ACTIVE)
    for task, candidates in sweep.CANDIDATES.items():
        config = json.loads((TASKS / task / "config.json").read_text())
        assert set(candidates) == set(config["baselines"])


def test_retired_generator_cannot_overwrite_tasks():
    tracked = [
        path
        for task in _task_dirs()
        for path in task.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    before = {path: hashlib.sha256(path.read_bytes()).digest() for path in tracked}
    process = subprocess.run(
        [sys.executable, str(VENDOR / "gen_tasks.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode != 0
    assert "intentionally disabled" in process.stdout
    after = {path: hashlib.sha256(path.read_bytes()).digest() for path in tracked}
    assert after == before


def test_read_only_analysis_tools_do_not_mutate_solutions():
    solutions = sorted((VENDOR / "solution").glob("*.py"))
    before = {path: hashlib.sha256(path.read_bytes()).digest() for path in solutions}
    for script in ("apply_real_anchors.py", "validate_all.py"):
        process = subprocess.run(
            [sys.executable, str(VENDOR / script)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
    after = {path: hashlib.sha256(path.read_bytes()).digest() for path in solutions}
    assert after == before


def test_scripts_are_strict_and_configs_are_valid_json():
    for task_dir in _task_dirs():
        json.loads((task_dir / "config.json").read_text())
        for script in sorted((task_dir / "scripts").glob("*.sh")):
            source = script.read_text()
            assert "set -euo pipefail" in source, script
            assert '--seed "${SEED:-0}"' in source, script
            assert "CUDA_VISIBLE_DEVICES=0" not in source, script
            if task_dir.name in ACTIVE:
                assert "MLSBENCH_VERIFIER_DATA_ROOT" in source, script
                assert (
                    'export INR_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/inr-signal-fitting"'
                    in source
                ), script


def test_rgb_jacobian_penalty_cannot_cancel_output_channels():
    source = (VENDOR / "common.py").read_text()
    assert "pred.sum()" not in source
    assert "pred[:, channel].sum()" in source
    assert "torch.stack(channel_grads, dim=1)" in source
    assert "torch.mean(jacobian ** 2)" in source
    assert 'if surface == "jacobian_reg"' in source
    assert 'if surface == "eikonal_reg"' not in source


def test_solution_modules_and_specs_compile():
    paths = [VENDOR / "common.py", VENDOR / "harness.py"]
    paths += sorted((VENDOR / "solution").glob("*.py"))
    paths += sorted(TASKS.glob("inr-*/parser.py"))
    paths += sorted(TASKS.glob("inr-*/score_spec.py"))
    paths += sorted(TASKS.glob("inr-*/edits/*.py"))
    for path in paths:
        source = path.read_text()
        ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
