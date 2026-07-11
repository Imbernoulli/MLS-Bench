"""Static and destructive replay checks for all normflows-density siblings."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

from mlsbench.scoring.anchors import BaselineAnchors
from mlsbench.scoring.evaluate import load_expanded_spec, score_record_details


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "flow-arch-family": {
        "surface": "architecture",
        "target": "pinwheel",
        "choice": "maf",
        "batch_size": 512,
        "lr": 5e-4,
        "transforms": 8,
        "permutations": 8,
        "metric": "nll_pinwheel",
        "nll": -0.25,
    },
    "flow-autoregressive-coupling": {
        "surface": "conditioner",
        "target": "8gaussians",
        "choice": "maf",
        "batch_size": 512,
        "lr": 5e-4,
        "transforms": 8,
        "permutations": 8,
        "metric": "nll_8gaussians",
        "nll": 2.4,
    },
    "flow-base-distribution": {
        "surface": "base_distribution",
        "target": "8gaussians",
        "choice": "gaussian",
        "batch_size": 512,
        "lr": 5e-4,
        "transforms": 1,
        "permutations": 1,
        "metric": "nll_8gaussians",
        "nll": 2.9,
    },
    "flow-batch-size": {
        "surface": "batch_size",
        "target": "checkerboard",
        "choice": "8",
        "batch_size": 8,
        "lr": 5e-4,
        "transforms": 8,
        "permutations": 8,
        "metric": "nll_checkerboard",
        "nll": 3.1,
    },
    "flow-conditioner-width": {
        "surface": "conditioner_width",
        "target": "checkerboard",
        "choice": "4",
        "batch_size": 512,
        "lr": 5e-4,
        "transforms": 8,
        "permutations": 8,
        "metric": "nll_checkerboard",
        "nll": 3.3,
    },
    "flow-coupling-transform": {
        "surface": "coupling_transform",
        "target": "checkerboard",
        "choice": "affine",
        "batch_size": 512,
        "lr": 5e-4,
        "transforms": 8,
        "permutations": 8,
        "metric": "nll_checkerboard",
        "nll": 3.125589,
    },
    "flow-depth-permutation": {
        "surface": "depth",
        "target": "moons",
        "choice": "2",
        "batch_size": 512,
        "lr": 5e-4,
        "transforms": 2,
        "permutations": 2,
        "metric": "nll_moons",
        "nll": 1.4,
    },
    "flow-learning-rate": {
        "surface": "learning_rate",
        "target": "moons",
        "choice": "0.05",
        "batch_size": 512,
        "lr": 0.05,
        "transforms": 8,
        "permutations": 8,
        "metric": "nll_moons",
        "nll": 1.6,
    },
    "flow-masking-pattern": {
        "surface": "masking_pattern",
        "target": "moons",
        "choice": "1010101010101010",
        "batch_size": 512,
        "lr": 5e-4,
        "transforms": 8,
        "permutations": 0,
        "metric": "nll_moons",
        "nll": 1.5,
    },
    "flow-spline-bins": {
        "surface": "spline_bins",
        "target": "checkerboard",
        "choice": "2",
        "batch_size": 512,
        "lr": 5e-4,
        "transforms": 8,
        "permutations": 8,
        "metric": "nll_checkerboard",
        "nll": 3.2,
    },
}
DATA_SHA256 = {
    "checkerboard": (
        "61a03c2d24a4a44eebbf61b7acd397b6df5834850889815aeaa4d4a3f1290ad4",
        "32137a42fad38a1363a3d0934ee3e46e6fbec3eca9ceca4d258b3b7e3d5b4f7f",
    ),
    "moons": (
        "48006975d1b5b065a9492f865fcfef808c2370312c6cf822cc7f0a94efa8b87e",
        "1e0bb490483a9992fc6f6c5321cdef6feb74c3e65a1e03a6099deac028024aac",
    ),
    "pinwheel": (
        "aec14ab3ed4d82f2976c369fdf53b51a1b600bbb9d32317702e0fc5670a88c84",
        "adeca11290413579bdf541647debfbfdfb82b725d2e98e0fdb267dab4a1606db",
    ),
    "8gaussians": (
        "54ec6fc49522ccea8526bd390037403034443fad7d484947ba3863538a51f332",
        "78f79e5b8e87cf865e38c46d303cf6ee99d4dddb7608d575e3abb57f514bf4ed",
    ),
}
PENDING = set(TASKS) - {"flow-coupling-transform"}
PROTOCOL = "flow-2d-community-20k-literal-ast-v3"


def _load_parser(task_name: str):
    path = ROOT / "tasks" / task_name / "parser.py"
    spec = importlib.util.spec_from_file_location(
        f"parser_{task_name.replace('-', '_')}", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Parser()


def _proof(task_name: str) -> str:
    case = TASKS[task_name]
    target = case["target"]
    train_sha, test_sha = DATA_SHA256[target]
    params = 35_344
    nll = case["nll"]
    bpd = nll / (2.0 * math.log(2.0))
    lines = [
        (
            f"FLOW_PROTOCOL version={PROTOCOL} surface={case['surface']} "
            f"choice={case['choice']} target={target} device=cuda device_count=1 "
            f"seed=42 steps=20000 batch_size={case['batch_size']} "
            f"lr={case['lr']:.12g} optimizer=Adam objective=exact_nll"
        ),
        (
            f"FLOW_DESIGN target={target} n_transforms={case['transforms']} "
            f"n_permutations={case['permutations']} "
            f"total_layers={case['transforms'] + case['permutations']} params={params}"
        ),
        (
            f"FLOW_DATA target={target} seed=42 n_train=30000 n_test=30000 "
            f"train_sha256={train_sha} test_sha256={test_sha}"
        ),
    ]
    lines.extend(
        f"FLOW_TRAIN step={step} train_nll=1.2345"
        for step in [*range(0, 20_000, 200), 19_999]
    )
    lines.extend(
        [
            f"FLOW_METRICS nll={nll:.6f} bpd={bpd:.6f} params={params} elapsed=250.0",
            (
                f"FLOW_SETTING_COMPLETE version={PROTOCOL} "
                f"surface={case['surface']} choice={case['choice']} target={target} "
                f"seed=42 optimizer_steps=20000 "
                f"samples_seen={20_000 * case['batch_size']} "
                f"n_train=30000 n_test=30000 params={params}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_complete_proof_is_accepted_for_every_sibling(task_name: str) -> None:
    case = TASKS[task_name]
    parsed = _load_parser(task_name).parse(case["target"], _proof(task_name))
    assert parsed.metrics == {case["metric"]: pytest.approx(case["nll"])}


def test_destructive_replay_20_of_20() -> None:
    rejected = 0
    for task_name, case in sorted(TASKS.items()):
        parser = _load_parser(task_name)
        valid = _proof(task_name)

        trailing_failure = valid + "late verifier failure\n"
        assert parser.parse(case["target"], trailing_failure).metrics == {}
        rejected += 1

        lines = valid.splitlines()
        metric_index = next(i for i, line in enumerate(lines) if line.startswith("FLOW_METRICS"))
        final_step_index = next(
            i for i, line in enumerate(lines) if line.startswith("FLOW_TRAIN step=19999 ")
        )
        lines[metric_index], lines[final_step_index] = (
            lines[final_step_index],
            lines[metric_index],
        )
        reordered = "\n".join(lines) + "\n"
        assert parser.parse(case["target"], reordered).metrics == {}
        rejected += 1

    assert rejected == 20


def test_additional_failure_count_and_identity_gates() -> None:
    task_name = "flow-coupling-transform"
    parser = _load_parser(task_name)
    target = TASKS[task_name]["target"]
    valid = _proof(task_name)
    mutations = [
        "[COMMAND FAILED exit=7]\n" + valid,
        valid.replace("n_train=30000", "n_train=29999", 1),
        valid.replace("optimizer_steps=20000", "optimizer_steps=19999", 1),
        valid.replace("samples_seen=10240000", "samples_seen=10239999", 1),
        valid.replace("n_transforms=8", "n_transforms=16", 1),
        valid.replace("params=35344", "params=35345", 1),
        valid.replace("FLOW_TRAIN step=19999 train_nll=1.2345\n", "", 1),
        valid.replace("FLOW_METRICS", "FLOW_METRICS nll=nan #", 1),
        valid.replace("FLOW_DATA", "FLOW_UNKNOWN", 1),
        valid + valid.splitlines()[-1] + "\n",
    ]
    assert all(parser.parse(target, output).metrics == {} for output in mutations)


@pytest.mark.parametrize("task_name", sorted(PENDING))
def test_uncalibrated_siblings_are_header_only_and_exact_zero(task_name: str) -> None:
    task_dir = ROOT / "tasks" / task_name
    assert len(task_dir.joinpath("leaderboard.csv").read_text().splitlines()) == 1
    score_source = task_dir.joinpath("score_spec.py").read_text()
    assert "const(" not in score_source
    assert "bl_best(" in score_source

    spec = load_expanded_spec(task_dir, BaselineAnchors(task_dir))
    assert spec is not None
    case = TASKS[task_name]
    score, settings, valid = score_record_details(
        spec,
        {case["metric"]: case["nll"]},
        BaselineAnchors(task_dir),
    )
    assert score == 0.0
    assert not valid
    assert settings and settings[0].invalid_reason is not None
    assert "missing_floor" in settings[0].invalid_reason


def test_representative_fresh_anchors_remain_calibrated() -> None:
    task_dir = ROOT / "tasks" / "flow-coupling-transform"
    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
    assert spec is not None
    weak = {
        "nll_checkerboard": 3.125589,
        "nll_moons": 1.030755,
        "nll_8gaussians": 2.386391,
    }
    strong = {
        "nll_checkerboard": 2.954646,
        "nll_moons": 1.025927,
        "nll_8gaussians": 2.373422,
    }
    weak_score, _settings, weak_valid = score_record_details(spec, weak, anchors)
    strong_score, _settings, strong_valid = score_record_details(spec, strong, anchors)
    assert weak_valid and strong_valid
    assert weak_score == pytest.approx(0.1, abs=2e-12)
    assert strong_score == pytest.approx(0.5, abs=2e-12)


def test_static_protocol_and_question_quality() -> None:
    headings = []
    parser_source = (ROOT / "tasks" / "flow-coupling-transform" / "parser.py").read_bytes()
    verifier_package_files = {
        "normflows-density/common.py",
        "normflows-density/flow_blocks.py",
        "normflows-density/harness_flow.py",
        "normflows-density/_flow_data/8gaussians_seed42.npz",
        "normflows-density/_flow_data/checkerboard_seed42.npz",
        "normflows-density/_flow_data/moons_seed42.npz",
        "normflows-density/_flow_data/pinwheel_seed42.npz",
    }
    for task_name, case in TASKS.items():
        task_dir = ROOT / "tasks" / task_name
        config = json.loads(task_dir.joinpath("config.json").read_text())
        assert config["calibration_protocol"] == PROTOCOL
        assert config["seeds"] == [42]
        assert config["allow_create"] is False
        assert config["rigorous_codebase"] is True
        assert set(config["verifier_only_package_files"]) == verifier_package_files
        assert task_dir.joinpath("parser.py").read_bytes() == parser_source
        if task_name in PENDING:
            assert config["calibration_status"] == "pending_community_20k_reanchor"
            assert not any(key.startswith("calibration_representative_") for key in config)
        else:
            assert config["calibration_status"] == "fresh_community_20k_anchors_20260711"
            assert config["calibration_representative_steps"] == 20_000
            assert config["calibration_representative_train_count"] == 30_000
            assert config["calibration_representative_test_count"] == 30_000

        labels = {entry["label"] for entry in config["test_cmds"]}
        expected_labels = {"checkerboard", "moons", "8gaussians"} if (
            task_name == "flow-coupling-transform"
        ) else {case["target"]}
        assert labels == expected_labels
        for entry in config["test_cmds"]:
            script = task_dir.joinpath(entry["cmd"]).read_text()
            assert "exec python harness_flow.py" in script
            assert "cd /workspace/normflows-density || exit 111" in script
            assert "--steps 20000" in script
            assert "--n-train 30000" in script
            assert "--n-test 30000" in script
            assert "echo \"FLOW_SETTING_COMPLETE" not in script
            assert not re.search(
                r"pip install|apt-get|conda install|curl\s|wget\s|git clone|tar\s|unzip",
                script,
            )

        description = task_dir.joinpath("task_description.md").read_text()
        headings.append(description.splitlines()[0])
        assert "20,000 Adam optimizer steps" in description
        assert "30,000 training samples" in description
        assert "30,000 verifier-only" in description
        assert re.search(r"\b(public|hidden)\b", description, re.IGNORECASE) is None

    assert len(headings) == len(set(headings)) == 10
    coupling_description = (
        ROOT / "tasks" / "flow-coupling-transform" / "task_description.md"
    ).read_text()
    coupling_description = re.sub(r"\s+", " ", coupling_description)
    assert "eight coupling transforms" in coupling_description
    assert "eight between-coupling permutation" in coupling_description
    assert "16 total" in coupling_description
    assert "16 coupling" not in coupling_description


def test_literal_custom_templates_and_edit_ranges() -> None:
    for task_name in TASKS:
        task_dir = ROOT / "tasks" / task_name
        config = json.loads(task_dir.joinpath("config.json").read_text())
        file_rule = config["files"][0]
        solution = ROOT / "vendor" / file_rule["filename"]
        source_lines = solution.read_text().splitlines()
        edit_range = file_rule["edit"][0]
        assert 1 <= edit_range["start"] <= edit_range["end"] <= len(source_lines)
        assert source_lines[edit_range["start"] - 1].startswith("def select_")

        template = task_dir / "edits" / "custom_template.py"
        tree = ast.parse(template.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        assert len(functions) == 1
        assert functions[0].name.startswith("select_")
        assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)

        for baseline in config["baselines"].values():
            edit_path = task_dir / baseline["edit_ops"]
            edit_tree = ast.parse(edit_path.read_text())
            assignment = next(node for node in edit_tree.body if isinstance(node, ast.Assign))
            ops = ast.literal_eval(assignment.value)
            assert len(ops) == 1
            op = ops[0]
            assert op["file"] == file_rule["filename"]
            assert op["start_line"] == edit_range["start"]
            assert op["end_line"] == edit_range["end"]


def test_frozen_archive_identities() -> None:
    vendor_data = ROOT / "vendor" / "normflows-density" / "_flow_data"
    for target, (train_sha, test_sha) in DATA_SHA256.items():
        train_path = vendor_data / f"{target}_seed42.npz"
        assert hashlib.sha256(train_path.read_bytes()).hexdigest() == train_sha
        matching_tasks = [
            task_name for task_name, case in TASKS.items() if case["target"] == target
        ]
        for task_name in matching_tasks:
            test_path = ROOT / "tasks" / task_name / "data" / f"{target}_seed42_test.npz"
            assert hashlib.sha256(test_path.read_bytes()).hexdigest() == test_sha


def test_representative_leaderboard_has_only_fresh_full_scale_rows() -> None:
    path = ROOT / "tasks" / "flow-coupling-transform" / "leaderboard.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["model"] for row in rows] == ["baseline:affine", "baseline:spline"]
    assert all(row["is_final"] == "true" and row["seed"] == "42" for row in rows)
    assert all(float(row["elapsed_checkerboard"]) > 200.0 for row in rows)


def test_one_pinned_repository_image_and_host_only_dgp() -> None:
    package_config = json.loads(
        (ROOT / "vendor" / "pkg_configs" / "normflows-density" / "config.json").read_text()
    )
    assert package_config["mangrove_base_image"] == (
        "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
        "mlsbench-harbor-normflows-density@sha256:"
        "3b81a711a7a6a00234a5717f7f5199ae124faee0d7cf24b0527e186ebbf40837"
    )
    assert package_config["use_cuda"] is True
    packages = (ROOT / "vendor" / "packages.yaml").read_text()
    assert packages.count("\n  normflows-density:\n") == 1

    assert (ROOT / "holdout" / "normflows-density" / "dgp.py").is_file()
    assert (ROOT / "holdout" / "normflows-density" / "generate_data.py").is_file()
    assert not (ROOT / "vendor" / "normflows-density" / "baselines").exists()
    vendor_files = {
        path.relative_to(ROOT / "vendor" / "normflows-density").as_posix()
        for path in (ROOT / "vendor" / "normflows-density").rglob("*")
        if path.is_file()
    }
    assert "dgp.py" not in vendor_files
    assert "generate_data.py" not in vendor_files

    worker = ROOT.joinpath("scripts/run_normflows_density_static_worker.sh").read_text()
    assert 'date -Iseconds > FINISHED' in worker
    assert "printf '%s\\n' \"${rc}\" > rc" in worker
    assert 'date -Iseconds > SUCCESS' in worker
    assert worker.rstrip().endswith("finish 0")
