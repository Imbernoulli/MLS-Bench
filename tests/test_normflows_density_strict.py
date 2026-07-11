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
        "target": "checkerboard",
        "choice": "maf",
        "batch_size": 512,
        "lr": 5e-4,
        "transforms": 8,
        "permutations": 8,
        "metric": "nll_checkerboard",
        "nll": 3.125589,
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
        "nll": 2.386391,
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
        "nll": 2.386391,
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
        "nll": 3.125589,
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
        "nll": 3.125589,
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
        "nll": 1.030755,
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
        "nll": 1.030755,
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
        "nll": 1.030755,
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
        "nll": 3.125589,
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
SHARED_CALIBRATION = set(TASKS) - {"flow-coupling-transform"}
PROTOCOL = "flow-2d-community-20k-literal-ast-v3"
EVIDENCE_SHA256 = "73429c480ad6dc0e8f3fb147668e6195fb3d0fcc173079814f9868b8c18d41ef"
QUALITY_ANCHORS = {
    "checkerboard": {
        "weak": 3.125589,
        "strong": 2.954646,
        "scale": 0.077799512058635861,
    },
    "moons": {
        "weak": 1.030755,
        "strong": 1.025927,
        "scale": 0.00219731749307721,
    },
    "8gaussians": {
        "weak": 2.386391,
        "strong": 2.373422,
        "scale": 0.0059024462650617299,
    },
}


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


@pytest.mark.parametrize("task_name", sorted(SHARED_CALIBRATION))
def test_shared_dataset_calibration_is_active_without_fabricated_rows(
    task_name: str,
) -> None:
    task_dir = ROOT / "tasks" / task_name
    assert len(task_dir.joinpath("leaderboard.csv").read_text().splitlines()) == 1
    score_source = task_dir.joinpath("score_spec.py").read_text()
    assert "const(" in score_source
    assert "bl_best(" not in score_source
    assert EVIDENCE_SHA256 in score_source

    spec = load_expanded_spec(task_dir, BaselineAnchors(task_dir))
    assert spec is not None
    case = TASKS[task_name]
    score, settings, valid = score_record_details(
        spec,
        {case["metric"]: case["nll"]},
        BaselineAnchors(task_dir),
    )
    assert valid and 0.0 < score < 1.0
    assert settings and settings[0].invalid_reason is None

    for invalid_record in ({}, {case["metric"]: math.nan}, {case["metric"]: math.inf}):
        invalid_score, invalid_settings, invalid = score_record_details(
            spec, invalid_record, BaselineAnchors(task_dir),
        )
        assert invalid_score == 0.0
        assert not invalid
        assert invalid_settings and invalid_settings[0].invalid_reason is not None

    failed_metrics = _load_parser(task_name).parse(
        case["target"], "[COMMAND FAILED exit=7]\n" + _proof(task_name),
    ).metrics
    assert failed_metrics == {}
    failed_score, _failed_settings, failed_valid = score_record_details(
        spec, failed_metrics, BaselineAnchors(task_dir),
    )
    assert failed_score == 0.0 and not failed_valid


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


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_same_dataset_quality_endpoints_map_consistently(task_name: str) -> None:
    task_dir = ROOT / "tasks" / task_name
    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
    assert spec is not None
    if task_name == "flow-coupling-transform":
        weak = {f"nll_{target}": values["weak"] for target, values in QUALITY_ANCHORS.items()}
        strong = {
            f"nll_{target}": values["strong"]
            for target, values in QUALITY_ANCHORS.items()
        }
    else:
        case = TASKS[task_name]
        values = QUALITY_ANCHORS[case["target"]]
        weak = {case["metric"]: values["weak"]}
        strong = {case["metric"]: values["strong"]}
    weak_score, _weak_settings, weak_valid = score_record_details(spec, weak, anchors)
    strong_score, _strong_settings, strong_valid = score_record_details(
        spec, strong, anchors,
    )
    assert weak_valid and strong_valid
    assert weak_score == pytest.approx(0.1, abs=2e-12)
    assert strong_score == pytest.approx(0.5, abs=2e-12)


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_all_siblings_have_finite_success_and_exact_zero_invalid_scores(
    task_name: str,
) -> None:
    task_dir = ROOT / "tasks" / task_name
    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
    assert spec is not None
    if task_name == "flow-coupling-transform":
        valid_record = {
            f"nll_{target}": values["weak"]
            for target, values in QUALITY_ANCHORS.items()
        }
    else:
        case = TASKS[task_name]
        parsed = _load_parser(task_name).parse(case["target"], _proof(task_name))
        valid_record = parsed.metrics
    score, settings, valid = score_record_details(spec, valid_record, anchors)
    assert valid and math.isfinite(score) and 0.0 < score < 1.0
    assert settings and all(setting.invalid_reason is None for setting in settings)

    for invalid_record in ({}, {next(iter(valid_record)): math.nan}):
        invalid_score, invalid_settings, invalid = score_record_details(
            spec, invalid_record, anchors,
        )
        assert invalid_score == 0.0 and not invalid
        assert invalid_settings and any(
            setting.invalid_reason is not None for setting in invalid_settings
        )


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
        assert EVIDENCE_SHA256 in task_dir.joinpath("score_spec.py").read_text()
        assert config["calibration_protocol"] == PROTOCOL
        assert config["seeds"] == [42]
        assert config["allow_create"] is False
        assert config["rigorous_codebase"] is True
        assert set(config["verifier_only_package_files"]) == verifier_package_files
        assert task_dir.joinpath("parser.py").read_bytes() == parser_source
        if task_name == "flow-coupling-transform":
            assert config["calibration_status"] == "fresh_community_20k_anchors_20260711"
            assert config["calibration_representative_steps"] == 20_000
            assert config["calibration_representative_train_count"] == 30_000
            assert config["calibration_representative_test_count"] == 30_000
        else:
            assert config["calibration_status"] == (
                "shared_dataset_quality_anchors_20260711"
            )
            train_sha, test_sha = DATA_SHA256[case["target"]]
            assert config["calibration_dataset"] == {
                "name": case["target"],
                "train_sha256": train_sha,
                "test_sha256": test_sha,
            }
            assert config["calibration_fixed_workload"] == {
                "seed": 42,
                "optimizer": "Adam",
                "optimizer_steps": 20_000,
                "train_count": 30_000,
                "test_count": 30_000,
                "metric": "exact_nll",
            }
            source = config["calibration_source"]
            assert source["task"] == "flow-coupling-transform"
            assert source["evidence_repository_path"] == (
                "tasks/flow-coupling-transform/calibration_evidence.json"
            )
            assert source["evidence_sha256"] == EVIDENCE_SHA256
            assert source["source_commit"] == (
                "cf0202decf342200005fa40dbc00726c23ed45db"
            )
            assert source["task_checksum"] == (
                "9c02337322ec5ce4299a06ee4409265834bac453a08b9f2680ebfd7927059546"
            )
            assert [
                (run["role"], run["task_db_id"], run["container_db_id"], run["agent"])
                for run in source["runs"]
            ] == [
                ("strong_anchor", 96_207, 4_909_807, "oracle"),
                ("weak_anchor", 96_208, 4_909_808, "nop"),
            ]
            evidence_path = ROOT / source["evidence_repository_path"]
            assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == EVIDENCE_SHA256
            assert json.loads(evidence_path.read_text())["task_name"] == (
                "flow-coupling-transform"
            )
            assert "task-specific" in source["disclaimer"]
            assert "not" in source["disclaimer"]
            if task_name == "flow-arch-family":
                assert config["calibration_reuse_scope"] == (
                    "same_dataset_same_protocol_identical_affine_spline_recipes"
                )
            else:
                assert config["calibration_reuse_scope"] == (
                    "same_dataset_same_protocol_absolute_exact_nll"
                )

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


def test_architecture_checkerboard_anchors_share_the_frozen_recipe_builder() -> None:
    source = (ROOT / "vendor" / "normflows-density" / "harness_flow.py").read_text()
    assert "def _append_canonical_recipe_layer(" in source
    assert source.count("_append_canonical_recipe_layer(") == 3
    assert 'family = "affine" if choice == "affine" else "spline"' in source
    arch_config = json.loads(
        (ROOT / "tasks" / "flow-arch-family" / "config.json").read_text()
    )
    assert [entry["label"] for entry in arch_config["test_cmds"]] == ["checkerboard"]
    arch_script = ROOT / "tasks" / "flow-arch-family" / "scripts" / "checkerboard.sh"
    assert "--target checkerboard" in arch_script.read_text()
    assert not (ROOT / "tasks" / "flow-arch-family" / "scripts" / "pinwheel.sh").exists()


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


def test_representative_provenance_is_explicit_and_bound() -> None:
    task_dir = ROOT / "tasks" / "flow-coupling-transform"
    config = json.loads(task_dir.joinpath("config.json").read_text())
    evidence_path = task_dir / config["calibration_representative_evidence_file"]
    evidence_bytes = evidence_path.read_bytes()
    assert hashlib.sha256(evidence_bytes).hexdigest() == (
        config["calibration_representative_evidence_sha256"]
    )
    evidence = json.loads(evidence_bytes)

    assert evidence["schema_version"] == 1
    assert evidence["task_name"] == "flow-coupling-transform"
    assert evidence["fixed_workload"] == {
        "seed": 42,
        "optimizer": "Adam",
        "optimizer_steps": 20_000,
        "train_count": 30_000,
        "test_count": 30_000,
        "settings": ["checkerboard", "moons", "8gaussians"],
        "gpu_count": 1,
        "gpu_type": "H20",
    }
    replay = evidence["current_parser_replay"]
    assert replay["protocol"] == PROTOCOL
    assert replay["accepted"] is False
    assert replay["scope"] == "all nine archived verifier logs listed in this file"
    assert "No archived log was rewritten" in replay["reason"]

    calibration = evidence["leaderboard_calibration"]
    assert calibration["dataset_version_id"] == 18_721
    assert calibration["dataset_item_id"] == 679_350
    assert calibration["source_commit"] == (
        "cf0202decf342200005fa40dbc00726c23ed45db"
    )
    assert calibration["source_commit_scope"] == "rendered Harbor dataset repository"
    assert calibration["harbor_render_revision"] == (
        "cf0202decf342200005fa40dbc00726c23ed45db"
    )
    assert calibration["task_checksum"] == (
        "9c02337322ec5ce4299a06ee4409265834bac453a08b9f2680ebfd7927059546"
    )

    expected_runs = {
        "baseline:spline": {
            "role": "strong_anchor",
            "task": 96_207,
            "container": 4_909_807,
            "runtime_container": "1331951",
            "agent": "oracle",
            "dataset_version": 18_721,
            "source_commit": "cf0202decf342200005fa40dbc00726c23ed45db",
            "task_checksum": "9c02337322ec5ce4299a06ee4409265834bac453a08b9f2680ebfd7927059546",
            "runner_id": "virt-v6h20",
            "archive": "59c084d1424dbda122e4ca5983fb186a1410ffd48cb25500c73d7c25bbe217b1",
            "logs": {
                "checkerboard": "dc79ce0d4621954e684d8dc12ccbf495c42aded05203f08581dcfc9b7cdcf4f0",
                "moons": "61a7628c679fa738a722b2bba268808f681ff850b4b808082ec17e535356b5db",
                "8gaussians": "8df50987943b38e10c4d1426d0cadbab37888f945d04fd30bea627250909648a",
            },
        },
        "baseline:affine": {
            "role": "weak_anchor",
            "task": 96_208,
            "container": 4_909_808,
            "runtime_container": "1331952",
            "agent": "nop",
            "dataset_version": 18_721,
            "source_commit": "cf0202decf342200005fa40dbc00726c23ed45db",
            "task_checksum": "9c02337322ec5ce4299a06ee4409265834bac453a08b9f2680ebfd7927059546",
            "runner_id": "virt-v6h20",
            "archive": "5ca4212163ff5108205d940fe9cf18557c74ee5652735700e484c29e35accabd",
            "logs": {
                "checkerboard": "035ca849a1ec993dfd5e60ba7edcb6c6fca1875be697bb6b937daac1af375378",
                "moons": "90e9b53c549ca18aa7dc0c17515dad7b5d0f1846640b84d1155b9390e040602b",
                "8gaussians": "eb2c68ad87e4888536db6f02a9ef4e38eef12d60259f1ff7318e550a2fc866fa",
            },
        },
    }
    runs = {run["leaderboard_model"]: run for run in calibration["runs"]}
    assert set(runs) == set(expected_runs)

    with task_dir.joinpath("leaderboard.csv").open(newline="") as handle:
        rows = {row["model"]: row for row in csv.DictReader(handle)}
    assert set(rows) == set(expected_runs)
    for model, expected in expected_runs.items():
        run = runs[model]
        row = rows[model]
        assert run["role"] == expected["role"]
        assert run["mangrove_task_db_id"] == expected["task"]
        assert run["mangrove_container_db_id"] == expected["container"]
        assert run["runtime_container_id"] == expected["runtime_container"]
        assert run["agent"] == expected["agent"]
        assert run["runner_id"] == expected["runner_id"]
        assert run["artifact_archive_sha256"] == expected["archive"]
        assert run["has_exception"] is False
        assert run["gpu_count"] == 1 and run["gpu_type"] == "H20"
        assert row["timestamp"] == run["leaderboard_timestamp"]
        assert row["is_final"] == "true" and int(row["seed"]) == 42
        for setting in ("checkerboard", "moons", "8gaussians"):
            setting_evidence = run["settings"][setting]
            assert float(row[f"nll_{setting}"]) == setting_evidence["nll"]
            assert float(row[f"elapsed_{setting}"]) == (
                setting_evidence["reported_harness_elapsed_seconds"]
            )
            assert setting_evidence["trace_record_count"] == 101
            assert setting_evidence["trace_first_step"] == 0
            assert setting_evidence["trace_final_step"] == 19_999
            log_path = f"verifier/{setting}__seed42.log"
            assert run["artifact_file_sha256"][log_path] == expected["logs"][setting]
        for digest in [
            run["sync_snapshot_sha256"],
            run["artifact_archive_sha256"],
            *run["artifact_file_sha256"].values(),
        ]:
            assert re.fullmatch(r"[0-9a-f]{64}", digest)

    config_runs = {
        run["role"]: run for run in config["calibration_representative_mangrove_runs"]
    }
    for expected in expected_runs.values():
        config_run = config_runs[expected["role"]]
        assert config_run == {
            "role": expected["role"],
            "task_db_id": expected["task"],
            "container_db_id": expected["container"],
            "dataset_version_id": expected["dataset_version"],
            "agent": expected["agent"],
            "source_commit": expected["source_commit"],
            "task_checksum": expected["task_checksum"],
            "runner_id": expected["runner_id"],
            "has_exception": False,
            "artifact_archive_sha256": expected["archive"],
        }

    runtime = evidence["final_render_runtime_reproduction"]
    assert runtime["scope"].startswith("Runtime and workload reproduction")
    assert runtime["leaderboard_source"] is False
    assert runtime["same_affine_nll_as_leaderboard"] is True
    assert runtime["dataset_version_id"] == 18_738
    assert runtime["dataset_item_id"] == 679_817
    assert runtime["source_commit"] == (
        "809b5ca1c659f2ceb3ff2632ff67ecaeeb3f8514"
    )
    assert runtime["source_commit_scope"] == "rendered Harbor dataset repository"
    assert runtime["harbor_render_revision"] == (
        "809b5ca1c659f2ceb3ff2632ff67ecaeeb3f8514"
    )
    assert runtime["task_checksum"] == (
        "b37d6416fc3155f43b180934b8037e173eb54c41afea0da8e96ed4f1dd6e0f5b"
    )
    assert runtime["mangrove_task_db_id"] == 96_410
    assert runtime["mangrove_container_db_id"] == 4_930_273
    assert runtime["runtime_container_id"] == "1475852"
    assert runtime["agent"] == "nop" and runtime["baseline"] == "affine"
    assert runtime["has_exception"] is False
    assert runtime["gpu_count"] == 1 and runtime["gpu_type"] == "H20"
    assert runtime["artifact_archive_sha256"] == (
        "8aa5ae30af3eb35b9e8e72760421ae23ba2eef588b68ddb7b86bab2b5177c3f4"
    )
    weak = runs["baseline:affine"]
    for setting in ("checkerboard", "moons", "8gaussians"):
        assert runtime["settings"][setting]["nll"] == weak["settings"][setting]["nll"]
        assert runtime["settings"][setting]["reported_harness_elapsed_seconds"] != (
            weak["settings"][setting]["reported_harness_elapsed_seconds"]
        )
    assert config["calibration_runtime_reproduction"] == {
        "scope": "runtime_only_not_leaderboard",
        "task_db_id": 96_410,
        "container_db_id": 4_930_273,
        "dataset_version_id": 18_738,
        "agent": "nop",
        "source_commit": "809b5ca1c659f2ceb3ff2632ff67ecaeeb3f8514",
        "task_checksum": "b37d6416fc3155f43b180934b8037e173eb54c41afea0da8e96ed4f1dd6e0f5b",
        "runner_id": "virt-v6h20",
        "has_exception": False,
        "artifact_archive_sha256": "8aa5ae30af3eb35b9e8e72760421ae23ba2eef588b68ddb7b86bab2b5177c3f4",
    }


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
    assert "tests/test_scoring_shared_contract.py" in worker
    assert "tests/test_workspace_tools_fail_closed.py" in worker
    assert "tests/test_scoring_fail_closed.py" not in worker
    assert "tests/test_scoring_no_implicit_fallback.py" not in worker
    assert 'run_logged "${RUN}/pytest.log"' in worker
    assert 'run_logged "${RUN}/render.log"' in worker
    assert 'audit_status=("${PIPESTATUS[@]}")' in worker
    assert 'finish "${audit_rc}"' in worker
    assert 'date -Iseconds > "${RUN}/FINISHED"' in worker
    assert "printf '%s\\n' \"${rc}\" > \"${RUN}/rc\"" in worker
    assert 'date -Iseconds > "${RUN}/SUCCESS"' in worker
    assert '| tee "${RUN}/summary"' not in worker
    assert 'run.joinpath("FINISHED").write_text' in worker
    assert 'run.joinpath("SUCCESS").write_text' in worker
    assert "scorable=10 shared_dataset_calibration=pass" in worker
    assert "pending_zero" not in worker
    assert worker.rstrip().endswith("finish 0")
