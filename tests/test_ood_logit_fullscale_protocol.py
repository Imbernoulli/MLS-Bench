from __future__ import annotations

import importlib.util
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlsbench.agent.tools import WorkspaceTools
from mlsbench.scoring.anchors import BaselineAnchors
from mlsbench.scoring.evaluate import load_expanded_spec, score_record_details


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "ood-logit-score"
PACKAGE = ROOT / "vendor" / "ood-detection-lab"
MODEL_SHA = "1" * 64


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PARSER = _load("ood_logit_fullscale_parser", TASK / "parser.py")


def _metric(
    setting: str,
    ood: str,
    n_ood: int,
    batches: int,
    *,
    auroc: str = "0.75000000",
    fpr95: str = "0.50000000",
    accuracy: str = "0.94000000",
    seconds: str = "1.250000",
) -> str:
    return (
        f"OOD_METRICS protocol={PARSER.PROTOCOL} task=ood-logit-score "
        f"setting={setting} ood={ood} "
        f"auroc={auroc} fpr95={fpr95} id_acc={accuracy} "
        f"n_fit=50000 n_id=10000 n_ood={n_ood} forward_batches={batches} "
        f"inference_seconds={seconds} status=ok"
    )


def _completion(*, data_sha: str | None = None, model_sha: str = MODEL_SHA) -> str:
    return (
        f"OOD_COMPLETE protocol={PARSER.PROTOCOL} task=ood-logit-score "
        f"data_sha256={data_sha or PARSER.EXPECTED_DATA_SHA256} "
        f"checkpoint_sha256={model_sha} n_fit=50000 n_id=10000 "
        "n_svhn=26032 n_cifar100=10000 n_tin=10000 "
        "total_forward_images=106032 total_forward_batches=832 status=ok"
    )


def _valid_log() -> str:
    return "\n".join(
        [
            "OOD_PROTOCOL protocol=openood_cifar10_resnet18_full_v1 "
            "task=ood-logit-score model=openood_resnet18_32x32 "
            "batch_size=128 seed=42 device='NVIDIA H20'",
            _metric("ood_logit_svhn_full", "svhn", 26_032, 204),
            _metric("ood_logit_cifar100_full", "cifar100", 10_000, 79),
            _metric("ood_logit_tin_full", "tin", 10_000, 79),
            _completion(),
        ]
    )


def _swap_first_two_metric_rows(text: str) -> str:
    lines = text.splitlines()
    return "\n".join((lines[0], lines[2], lines[1], *lines[3:]))


@pytest.fixture(autouse=True)
def _pin_test_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PARSER, "EXPECTED_MODEL_SHA256", MODEL_SHA)


def test_parser_accepts_exactly_one_complete_three_setting_proof() -> None:
    result = PARSER.Parser().parse("ood_logit_full_protocol", _valid_log())

    assert set(result.metrics) == {
        "auroc_ood_logit_svhn_full",
        "fpr95_ood_logit_svhn_full",
        "id_acc_ood_logit_svhn_full",
        "auroc_ood_logit_cifar100_full",
        "fpr95_ood_logit_cifar100_full",
        "id_acc_ood_logit_cifar100_full",
        "auroc_ood_logit_tin_full",
        "fpr95_ood_logit_tin_full",
        "id_acc_ood_logit_tin_full",
    }


def test_agent_failure_can_leave_the_untouched_native_solution_to_verify() -> None:
    result = PARSER.Parser().parse(
        "ood_logit_full_protocol",
        "AGENT_FAILED before editing; evaluating untouched native solution\n" + _valid_log(),
    )
    assert len(result.metrics) == 9


@pytest.mark.parametrize(
    "raw_output",
    [
        "",
        "\n".join(_valid_log().splitlines()[:-2] + [_completion()]),
        "\n".join(_valid_log().splitlines()[:-1] + [_valid_log().splitlines()[1], _completion()]),
        _valid_log().replace("n_svhn=26032", "n_svhn=5000"),
        _valid_log().replace(PARSER.EXPECTED_DATA_SHA256, "0" * 64),
        _valid_log().replace(MODEL_SHA, "2" * 64),
        _valid_log().replace("n_ood=26032", "n_ood=5000"),
        _valid_log().replace("ood=svhn", "ood=tin", 1),
        _valid_log().replace("id_acc=0.94000000", "id_acc=0.93000000", 1),
        _valid_log().replace("auroc=0.75000000", "auroc=nan", 1),
        _valid_log().replace("fpr95=0.50000000", "fpr95=inf", 1),
        _valid_log().replace("inference_seconds=1.250000", "inference_seconds=0", 1),
        _valid_log() + "\ntrailing output",
        "Traceback (most recent call last)\n" + _valid_log(),
        "OOD_FAILURE reason=test\n" + _valid_log(),
        "[COMMAND FAILED exit=17]\n" + _valid_log(),
        "Budget Check Failed\n" + _valid_log(),
        "STATUS: FAILED\n" + _valid_log(),
        "Process Exited With Code 9\n" + _valid_log(),
        "command failed\n" + _valid_log(),
        "node_fail\n" + _valid_log(),
        _valid_log().replace("OOD_PROTOCOL", "ood_protocol", 1),
        _swap_first_two_metric_rows(_valid_log()),
    ],
)
def test_parser_rejects_failed_ambiguous_or_incomplete_logs(raw_output: str) -> None:
    assert PARSER.Parser().parse("ood_logit_full_protocol", raw_output).metrics == {}


def test_nonzero_command_cannot_submit_valid_looking_ood_metrics() -> None:
    tools = object.__new__(WorkspaceTools)
    tools.container_runtime = "local"
    tools._current_test_had_failures = False
    tools._run_budget_check = lambda *args, **kwargs: None
    tools._build_local_exec_spec = lambda *args, **kwargs: (["false"], Path("."), {})
    tools._run_local_command = lambda *args, **kwargs: (
        SimpleNamespace(returncode=17),
        _valid_log(),
        False,
    )
    tools.parser = PARSER.Parser()

    _feedback, metrics, _elapsed = tools._run_single_cmd(
        {"label": "ood_logit_full_protocol", "cmd": "verify.sh"}, seed=42,
    )

    assert metrics == {}
    assert tools._current_test_had_failures is True


def test_task_config_has_one_serial_full_protocol_and_no_hidden_semantics() -> None:
    config = json.loads((TASK / "config.json").read_text())

    assert config["seeds"] == [42]
    assert len(config["test_cmds"]) == 1
    command = config["test_cmds"][0]
    assert command["cmd"] == "scripts/ood_logit_full.sh"
    assert command["compute"] == 1.0
    assert "hidden" not in command
    assert "agent_data_prune" not in config
    assert "verifier_data_deps" not in config
    assert set(config["agent_pruned_package_files"]) == {
        "ood-detection-lab/harness.py",
        "ood-detection-lab/parser_contract.py",
    }


def test_package_uses_one_pinned_prebuilt_image_without_install_commands() -> None:
    config = json.loads(
        (ROOT / "vendor/pkg_configs/ood-detection-lab/config.json").read_text()
    )

    expected_image = (
        "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
        "mlsbench-harbor-ood-detection-lab@"
        "sha256:c96492da2073103f2d59b3aad629b7c9560ed07d9ce6b5d315e0d46ec046fe8e"
    )
    assert config["base_image"] == expected_image
    assert config["mangrove_base_image"] == expected_image
    assert all(
        token not in "\n".join(config["install_cmds"])
        for token in ("pip install", "conda install", "apt-get", "curl ", "wget ")
    )


def test_measured_calibration_maps_msp_to_point_one_and_strong_to_point_five() -> None:
    anchors = BaselineAnchors(TASK)
    spec = load_expanded_spec(TASK, anchors)
    assert spec is not None
    weak = {
        "auroc_ood_logit_svhn_full": 0.92846834,
        "auroc_ood_logit_cifar100_full": 0.87088166,
        "auroc_ood_logit_tin_full": 0.87007299,
    }
    strong = {
        "auroc_ood_logit_svhn_full": 0.93220756,
        "auroc_ood_logit_cifar100_full": 0.89527483,
        "auroc_ood_logit_tin_full": 0.89377779,
    }

    weak_score, weak_settings, weak_valid = score_record_details(spec, weak, anchors)
    strong_score, strong_settings, strong_valid = score_record_details(
        spec, strong, anchors,
    )

    assert weak_valid and strong_valid
    assert weak_score == pytest.approx(0.1, abs=1e-12)
    assert strong_score == pytest.approx(0.5, abs=1e-12)
    assert [item.score for item in weak_settings] == pytest.approx([0.1] * 3)
    assert [item.score for item in strong_settings] == pytest.approx([0.5] * 3)


def test_positive_logit_rows_have_immutable_mangrove_artifact_provenance() -> None:
    rows = (TASK / "leaderboard.csv").read_text().splitlines()
    assert len(rows) == 3
    assert rows[1].split(",")[1] == "baseline:msp"
    assert rows[2].split(",")[1] == "baseline:pseudo_cosine"
    provenance = (PACKAGE / "anchors" / "README.md").read_text()
    for required in (
        "dataset version `18763`",
        "ba0f93fec2a6e041bd87902a9a339c5370f9b161",
            "task `96612`",
            "`4950395`",
            "task `96611`",
            "`4950394`",
        "be9e85cbd686012b45010909613e78fbfc78368bf0afdc2b3d1a44b7dc07c0d4",
        "63e7771ec5d74015efa4f90e441800ca387d941c0e8cdd4624583ef99cb573ff",
        "423b6a26841540806f385b6e72fc9872b8b68ebc6504d77215643d6c9ea43779",
        "76fdbede23845652bf551faaa948c3f6d8497996d5615ae999a5180f737e7fe2",
        "command `rc=0`",
    ):
        assert required in provenance


@pytest.mark.parametrize(
    "invalid",
    [
        {
            "auroc_ood_logit_svhn_full": 0.93,
            "auroc_ood_logit_cifar100_full": 0.89,
        },
        {
            "auroc_ood_logit_svhn_full": float("nan"),
            "auroc_ood_logit_cifar100_full": 0.89,
            "auroc_ood_logit_tin_full": 0.89,
        },
        {
            "auroc_ood_logit_svhn_full": 0.93,
            "auroc_ood_logit_cifar100_full": float("inf"),
            "auroc_ood_logit_tin_full": 0.89,
        },
    ],
)
def test_incomplete_or_nonfinite_three_setting_record_scores_exact_zero(
    invalid: dict[str, float],
) -> None:
    anchors = BaselineAnchors(TASK)
    spec = load_expanded_spec(TASK, anchors)
    assert spec is not None

    score, _settings, valid = score_record_details(spec, invalid, anchors)

    assert score == 0.0
    assert valid is False


def test_edit_surface_and_baselines_target_the_same_complete_class_region() -> None:
    source = (PACKAGE / "solution" / "logit_score.py").read_text()
    template = (TASK / "edits" / "custom_template.py").read_text()
    config = json.loads((TASK / "config.json").read_text())
    edit = config["files"][0]["edit"][0]

    assert edit == {"start": 13, "end": 18}
    assert source.splitlines()[12].startswith("class Scorer:")
    assert template.splitlines()[12].startswith("class Scorer:")
    assert "Multiple fixed OOD evaluations" in template
    assert not any(name in template for name in ("SVHN", "CIFAR-100", "Tiny-ImageNet"))
    for name, descriptor in config["baselines"].items():
        operations = runpy.run_path(str(TASK / descriptor["edit_ops"]))["OPS"]
        assert len(operations) == 1
        operation = operations[0]
        assert operation["file"] == "ood-detection-lab/solution/logit_score.py"
        assert (operation["start_line"], operation["end_line"]) == (13, 18)
        compile(operation["content"], f"ood-logit-{name}", "exec")


def test_final_verifier_script_is_offline_and_inherits_one_assigned_gpu() -> None:
    path = TASK / "scripts" / "ood_logit_full.sh"
    source = path.read_text()

    subprocess.run(["bash", "-n", str(path)], check=True)
    assert "ood_full_eval_uint8.npz" in source
    assert "openood_resnet18_cifar10_seed0.pt" in source
    assert "/data/ood-detection-lab" in source
    assert "CUDA_VISIBLE_DEVICES=" not in source
    assert "OOD_FAILURE task=ood-logit-score" in source
    assert not any(
        token in source
        for token in ("pip install", "conda install", "curl ", "wget ", "git clone")
    )


def test_asset_preparation_never_retrains_the_pinned_checkpoint() -> None:
    source = (
        ROOT / "vendor/data_scripts/ood-detection-lab/prepare_full_assets.sh"
    ).read_text()

    subprocess.run(
        ["bash", "-n", str(ROOT / "vendor/data_scripts/ood-detection-lab/prepare_full_assets.sh")],
        check=True,
    )
    assert "train_openood_resnet18.py" not in source
    assert "CHECKPOINT_SHA256=8859e0ff" in source


def test_harness_pins_complete_real_forward_inventory() -> None:
    source = (PACKAGE / "harness_full_logit.py").read_text()

    assert "torch.cuda.device_count() != 1" in source
    assert "total_images != 106_032 or total_batches != 832" in source
    assert source.count("extract_logits(") == 4
    assert "OOD_COMPLETE" in source
    assert "8859e0ff484ab029fcd5bd0f85052c5679d82b8e36a7c066e922e2fdff62b7dc" in source
    assert "TO_BE_PINNED_AFTER_OPENOOD_TRAINING" not in source
