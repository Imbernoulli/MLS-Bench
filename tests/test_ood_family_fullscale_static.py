from __future__ import annotations

import ast
import importlib.util
import json
import runpy
import subprocess
from pathlib import Path

import pytest

from mlsbench.scoring.anchors import BaselineAnchors
from mlsbench.scoring.evaluate import load_expanded_spec, score_record_details


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "vendor" / "ood-detection-lab"
TASKS = (
    "ood-distance-metric",
    "ood-ensemble",
    "ood-feature-score",
    "ood-gradient",
    "ood-input-preproc",
    "ood-layer-select",
    "ood-near-far",
    "ood-normalization",
    "ood-react",
    "ood-temperature",
)
DATA_SHA = "796799c9a1c073784b02a3f42a00d0fc8b902387b1f3345b5b3d1f2631e0722d"
MODEL_SHA = "8859e0ff484ab029fcd5bd0f85052c5679d82b8e36a7c066e922e2fdff62b7dc"


def _load_parser(task: str):
    path = ROOT / "tasks" / task / "parser.py"
    spec = importlib.util.spec_from_file_location(f"parser_{task.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _setting(task: str, ood: str) -> str:
    if task == "ood-near-far":
        regime = {"svhn": "far", "cifar100": "near", "tin": "medium"}[ood]
        return f"ood_{regime}_{ood}_full"
    return f"ood_{task.removeprefix('ood-').replace('-', '_')}_{ood}_full"


def _valid_log(task: str) -> str:
    input_preproc = task == "ood-input-preproc"
    rows = [
        "OOD_PROTOCOL protocol=openood_cifar10_resnet18_full_v1 "
        f"task={task} model=openood_resnet18_32x32 batch_size=128 seed=42 status=ok"
    ]
    for ood, count, batches in (
        ("svhn", 26_032, 204),
        ("cifar100", 10_000, 79),
        ("tin", 10_000, 79),
    ):
        id_score_batches = 158 if input_preproc else 0
        ood_score_batches = batches * 2 if input_preproc else 0
        rows.append(
            "OOD_METRICS protocol=openood_cifar10_resnet18_full_v1 "
            f"task={task} setting={_setting(task, ood)} ood={ood} "
            "auroc=0.75000000 fpr95=0.50000000 id_acc=0.94000000 "
            f"n_fit=50000 n_id=10000 n_ood={count} base_ood_batches={batches} "
            f"id_score_batches={id_score_batches} ood_score_batches={ood_score_batches} "
            "inference_seconds=1.250000 status=ok"
        )
    task_images, task_batches = (218_096, 1_714) if input_preproc else (106_032, 832)
    rows.append(
        "OOD_COMPLETE protocol=openood_cifar10_resnet18_full_v1 "
        f"task={task} data_sha256={DATA_SHA} checkpoint_sha256={MODEL_SHA} "
        "n_fit=50000 n_id=10000 n_svhn=26032 n_cifar100=10000 n_tin=10000 "
        "base_forward_images=106032 base_forward_batches=832 "
        f"task_forward_images={task_images} task_forward_batches={task_batches} status=ok"
    )
    return "\n".join(rows)


@pytest.mark.parametrize("task", TASKS)
def test_nonlogit_parser_accepts_only_complete_task_bound_full_protocol(task: str) -> None:
    parser = _load_parser(task)
    label = f"ood_{task.removeprefix('ood-').replace('-', '_')}_full_protocol"
    result = parser.Parser().parse(label, _valid_log(task))
    assert len(result.metrics) == 9
    assert set(result.metrics) == {
        metric
        for ood in ("svhn", "cifar100", "tin")
        for metric in (
            f"auroc_{_setting(task, ood)}",
            f"fpr95_{_setting(task, ood)}",
            f"id_acc_{_setting(task, ood)}",
        )
    }
    native_after_agent_failure = parser.Parser().parse(
        label,
        "AGENT_FAILED before editing; evaluating untouched native solution\n"
        + _valid_log(task),
    )
    assert len(native_after_agent_failure.metrics) == 9


@pytest.mark.parametrize(
    "mutate",
    (
        lambda text: text.replace("task=ood-gradient", "task=ood-react", 1),
        lambda text: text.replace(DATA_SHA, "0" * 64),
        lambda text: text.replace(MODEL_SHA, "1" * 64),
        lambda text: text.replace("n_fit=50000", "n_fit=5000", 1),
        lambda text: text.replace("n_ood=26032", "n_ood=5000", 1),
        lambda text: text.replace("auroc=0.75000000", "auroc=nan", 1),
        lambda text: text.replace("OOD_PROTOCOL", "ood_protocol", 1),
        lambda text: text + "\ntrailing output",
        lambda text: "Budget Check Failed\n" + text,
        lambda text: "status: failed\n" + text,
        lambda text: "Process Exited With Code 9\n" + text,
        lambda text: "command failed\n" + text,
        lambda text: "node_fail\n" + text,
    ),
)
def test_nonlogit_parser_rejects_wrong_failed_or_ambiguous_proof(mutate) -> None:
    parser = _load_parser("ood-gradient")
    assert parser.Parser().parse(
        "ood_gradient_full_protocol", mutate(_valid_log("ood-gradient")),
    ).metrics == {}
    assert parser.Parser().parse(
        "wrong_label", _valid_log("ood-gradient"),
    ).metrics == {}


def test_all_nonlogit_configs_use_one_serial_full_command_and_verifier_owned_contract() -> None:
    expected_private = {
        "ood-detection-lab/common.py",
        "ood-detection-lab/openood_resnet18.py",
        "ood-detection-lab/harness.py",
        "ood-detection-lab/parser_contract.py",
    }
    for task in TASKS:
        task_dir = ROOT / "tasks" / task
        config = json.loads((task_dir / "config.json").read_text())
        slug = task.removeprefix("ood-").replace("-", "_")
        assert config["seeds"] == [42]
        assert set(config["verifier_only_package_files"]) == expected_private
        assert config["agent_pruned_package_files"] == [
            "ood-detection-lab/harness_full_logit.py"
        ]
        assert len(config["test_cmds"]) == 1
        command = config["test_cmds"][0]
        assert command["label"] == f"ood_{slug}_full_protocol"
        assert command["compute"] == 1.0
        assert command["group"] == 1
        assert not any(key in command for key in ("public", "hidden", "private"))
        script = task_dir / command["cmd"]
        source = script.read_text()
        subprocess.run(["bash", "-n", str(script)], check=True)
        assert f"--task {task}" in source
        assert "ood_full_eval_uint8.npz" in source
        assert "openood_resnet18_cifar10_seed0.pt" in source
        assert f"OOD_FAILURE task={task}" in source
        assert "CUDA_VISIBLE_DEVICES=" not in source
        assert not any(
            token in source
            for token in ("pip install", "conda install", "apt-get", "curl ", "wget ", "git clone")
        )


@pytest.mark.parametrize("task", TASKS)
def test_baseline_free_full_protocol_auroc_has_a_positive_direct_score(task: str) -> None:
    task_dir = ROOT / "tasks" / task
    assert len((task_dir / "leaderboard.csv").read_text().splitlines()) == 1
    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
    assert spec is not None
    assert all(term.norm_type == "bounded_power" for term in spec.terms.values())
    assert all(term.ref is None for term in spec.terms.values())
    assert all(term.floor is not None for term in spec.terms.values())
    source = (task_dir / "score_spec.py").read_text().lower()
    assert "pending" not in source
    assert ".sigmoid(" not in source
    assert "ref=" not in source
    record = {
        f"auroc_{_setting(task, ood)}": 0.75
        for ood in ("svhn", "cifar100", "tin")
    }
    score, settings, valid = score_record_details(spec, record, anchors)
    assert valid is True
    assert score == pytest.approx(0.5, abs=1e-12)
    assert [item.score for item in settings] == pytest.approx([0.5, 0.5, 0.5])


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("failure", ("missing", "nan", "inf"))
def test_incomplete_or_nonfinite_direct_auroc_scores_exact_zero(
    task: str, failure: str,
) -> None:
    task_dir = ROOT / "tasks" / task
    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
    assert spec is not None
    record = {
        f"auroc_{_setting(task, ood)}": 0.75
        for ood in ("svhn", "cifar100", "tin")
    }
    key = f"auroc_{_setting(task, 'svhn')}"
    if failure == "missing":
        del record[key]
    else:
        record[key] = float(failure)
    score, _settings, valid = score_record_details(spec, record, anchors)
    assert score == 0.0
    assert valid is False


def test_gradient_surface_config_template_and_baselines_are_consistent() -> None:
    task_dir = ROOT / "tasks" / "ood-gradient"
    source = (PACKAGE / "solution" / "gradient_score.py").read_text()
    template = (task_dir / "edits" / "custom_template.py").read_text()
    config = json.loads((task_dir / "config.json").read_text())
    assert config["files"][0]["edit"] == [{"start": 14, "end": 20}]
    assert source.splitlines()[13].startswith("class Scorer:")
    assert template.splitlines()[13].startswith("class Scorer:")
    description = (task_dir / "task_description.md").read_text().lower()
    assert "select_gradient_sign" not in description
    assert "preferred sign" not in description
    for name, descriptor in config["baselines"].items():
        operation = runpy.run_path(str(task_dir / descriptor["edit_ops"]))["OPS"][0]
        assert operation["file"] == "ood-detection-lab/solution/gradient_score.py"
        assert (operation["start_line"], operation["end_line"]) == (14, 20)
        compile(operation["content"], f"gradient-{name}", "exec")


@pytest.mark.parametrize("task", TASKS)
def test_every_baseline_targets_the_declared_surface_and_compiles(task: str) -> None:
    task_dir = ROOT / "tasks" / task
    config = json.loads((task_dir / "config.json").read_text())
    declared_file = config["files"][0]["filename"]
    declared_edit = config["files"][0]["edit"][0]
    source_path = ROOT / "vendor" / declared_file
    source_lines = source_path.read_text().splitlines()
    start, end = declared_edit["start"], declared_edit["end"]
    assert 1 <= start <= end <= len(source_lines)
    for name, descriptor in config["baselines"].items():
        operations = runpy.run_path(str(task_dir / descriptor["edit_ops"]))["OPS"]
        assert len(operations) == 1
        operation = operations[0]
        assert operation["file"] == declared_file
        assert (operation["start_line"], operation["end_line"]) == (start, end)
        candidate = list(source_lines)
        candidate[start - 1:end] = operation["content"].splitlines()
        compile("\n".join(candidate) + "\n", f"{task}-{name}", "exec")


def test_full_harnesses_are_static_valid_and_have_no_training_fallback() -> None:
    for name in (
        "common.py",
        "openood_resnet18.py",
        "harness.py",
        "harness_full_logit.py",
        "parser_contract.py",
    ):
        source = (PACKAGE / name).read_text()
        ast.parse(source, filename=name)
    harness = (PACKAGE / "harness.py").read_text()
    assert "BASE_FORWARD_IMAGES = 106_032" in (PACKAGE / "common.py").read_text()
    assert "expected_task_images = 218_096" in harness
    assert "expected_task_batches = 1_714" in harness
    assert "ODIN_TEMPERATURE = 1000.0" in harness
    assert "neighbors=51" in harness
    assert "SmallCNN" not in harness
    assert "get_classifier" not in harness
    assert "fallback" not in harness.lower().replace("there is no cached-model training or implementation fallback", "")
    assert not (PACKAGE / "harness_logit_precomputed.py").exists()
    assert not (ROOT / "vendor/data_scripts/ood-detection-lab/prepare_data.py").exists()
    assert not (ROOT / "vendor/data_scripts/ood-detection-lab/prepare_logit_eval.py").exists()


def test_gradient_instruction_discloses_the_exact_fixed_representation() -> None:
    description = (
        ROOT / "tasks/ood-gradient/task_description.md"
    ).read_text()
    assert "g_j = |softmax(z)_j - 0.1| * ||h||_1" in description
    assert "[50000, 10]" in description
