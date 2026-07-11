from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASKS = sorted((ROOT / "tasks").glob("nli-*"))
VERIFIER_RUNTIME = {
    "natural-language-inference/common.py",
    "natural-language-inference/harness_augmentation.py",
    "natural-language-inference/harness_classifier_head.py",
    "natural-language-inference/harness_class_weighting.py",
    "natural-language-inference/harness_finetune.py",
    "natural-language-inference/harness_hypothesis_bias.py",
    "natural-language-inference/harness_interaction_head.py",
    "natural-language-inference/harness_pair_encoding.py",
    "natural-language-inference/harness_pooling.py",
    "natural-language-inference/harness_regularization.py",
    "natural-language-inference/harness_truncation.py",
}
DATA = {
    "snli_train": (549367, "3cdde4e94e0c5ca8e7e3d95b0c7c7b9fc03b101d3b9e79c422150bf5c17f1f73"),
    "snli": (9824, "e30ea21eb677dab4806e1cc4c646dffc23985ffd982fd6bd15ab3617cd601dd8"),
    "mnli_m": (9815, "a612ccdf07b2fbe73e2904b061b9e278f552a39b553999bc626de6df6ec4b66d"),
    "mnli_mm": (9832, "a08757b4ddc34421f8f6eac69eb5dd97b2125693078c541cad2d54689013f68d"),
}
MODEL_ASSETS = {
    "config.json": "69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83",
    "model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
TASK_PROTOCOL = {
    "nli-augmentation": ("augmentation", "none"),
    "nli-class-weighting": ("class_weighting", "class_weighting"),
    "nli-classifier-head": ("classifier_head", "linear"),
    "nli-finetune": ("finetune", "frozen"),
    "nli-hypothesis-bias": ("hypothesis_bias", "premise"),
    "nli-interaction-head": ("interaction_head", "infersent"),
    "nli-pair-encoding": ("pair_encoding", "cross"),
    "nli-pooling": ("pooling", "mean"),
    "nli-regularization": ("regularization", "standard"),
    "nli-truncation": ("truncation", "len128"),
}
REGULARIZATION = {
    "none": (0.0, 0.0),
    "standard": (0.1, 0.01),
    "heavy": (0.7, 0.3),
}
EXPECTED_MODELS = {
    "nli-augmentation": {
        mode: ("cross", 2307) for mode in ("none", "swap", "negation")
    },
    "nli-class-weighting": {"class_weighting": ("cross", 2307)},
    "nli-classifier-head": {
        "linear": ("siamese", 9219),
        "mlp": ("siamese", 1574915),
    },
    "nli-finetune": {
        "frozen": ("cross", 2307),
        "finetune": ("cross", 2307),
    },
    "nli-hypothesis-bias": {
        "premise": ("cross", 2307),
        "hyponly": ("cross", 2307),
    },
    "nli-interaction-head": {
        "concat": ("siamese", 4611),
        "infersent": ("siamese", 9219),
    },
    "nli-pair-encoding": {
        "cross": ("cross", 2307),
        "siamese": ("siamese", 9219),
    },
    "nli-pooling": {
        mode: ("siamese", 9219) for mode in ("cls", "mean", "max", "sum")
    },
    "nli-regularization": {
        mode: ("cross", 2307) for mode in REGULARIZATION
    },
    "nli-truncation": {
        f"len{value}": ("cross", 2307) for value in range(8, 129)
    },
}


def _load_parser(task_name: str):
    path = ROOT / f"tasks/{task_name}/parser.py"
    spec = importlib.util.spec_from_file_location(f"parser_{task_name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Parser()


def _load_prepare_module():
    path = ROOT / "vendor/data_scripts/natural-language-inference/prepare_data.py"
    spec = importlib.util.spec_from_file_location("nli_prepare_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _valid_log(
    task_name: str,
    max_length: int | None = None,
    policy: str | None = None,
) -> str:
    surface, default_policy = TASK_PROTOCOL[task_name]
    mode = policy or default_policy
    if max_length is None:
        if task_name in {
            "nli-classifier-head", "nli-interaction-head", "nli-pooling"
        } or (task_name == "nli-pair-encoding" and mode == "siamese"):
            max_length = 64
        elif task_name == "nli-truncation" and mode.startswith("len"):
            max_length = int(mode.removeprefix("len"))
        else:
            max_length = 128
    architecture, head_params = EXPECTED_MODELS[task_name][mode]
    encoder_params = 66362880
    total_params = encoder_params + head_params
    lines = [
        "NLI_DEVICE type=cuda visible=1",
        "NLI_PROTOCOL version=nli-full-snli-distilbert-v1 "
        f"task={task_name} surface={surface} settings=3 "
        f"train_rows=549367 eval_rows=29471 epochs=3 train_batch=32 "
        f"eval_batch=128 max_length={max_length} seed=42"
    ]
    for split, (rows, digest) in DATA.items():
        lines.append(f"NLI_DATA split={split} rows={rows} sha256={digest}")
    lines.extend([
        f"NLI_POLICY mode={mode}",
    ])
    if task_name == "nli-class-weighting":
        lines.append(
            "NLI_CLASS_WEIGHTS entailment=1 neutral=1 contradiction=1"
        )
    if task_name == "nli-regularization":
        dropout, weight_decay = REGULARIZATION[mode]
        lines.append(
            f"NLI_REGULARIZATION dropout={dropout:.8g} "
            f"weight_decay={weight_decay:.8g}"
        )
    else:
        weight_decay = 0.01
    lines.extend([
        "NLI_MODEL model=distilbert-base-uncased "
        "revision=12040accade4e8a0f71eabdb258fecc2e7e948be "
        f"architecture={architecture} encoder_params={encoder_params} "
        f"head_params={head_params} total_params={total_params} dtype=float32 "
        "config_sha256=69c94b0222d5d1f4b0ad027ca7416cdafb98378cbbb8305d0bf47c9365c60c83 "
        "weights_sha256=5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063 "
        "tokenizer_sha256=ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98 "
        "tokenizer_config_sha256=a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10 "
        "vocab_sha256=07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        f"NLI_TRAIN mode={mode} optimizer=adamw encoder_lr="
        f"{'0' if mode == 'frozen' else '2e-05'} head_lr=0.001 "
        f"weight_decay={weight_decay:.8g} warmup_ratio=0.1 epochs=3 batch=32 "
        f"max_length={max_length} expected_steps=51504",
    ])
    for epoch in (1, 2, 3):
        steps = epoch * 17168
        lines.append(
            f"NLI_EPOCH mode={mode} epoch={epoch} optimizer_steps={steps} "
            f"expected={steps} loss=0.5"
        )
    lines.append("NLI_TRAIN_DONE epochs=3 optimizer_steps=51504 expected_steps=51504")
    for setting in ("snli", "mnli_m", "mnli_mm"):
        rows = DATA[setting][0]
        lines.extend([
            f"NLI_METRICS setting={setting} acc=0.5 n_eval={rows}",
            f"NLI_SETTING_DONE setting={setting} predicted={rows} expected={rows}",
        ])
    lines.extend([
        "NLI_EVAL_DONE settings=3 rows=29471",
        "NLI_DONE settings=3 train_rows=549367 eval_rows=29471 seed=42 elapsed=10.0",
        "NLI_COMMAND_DONE rc=0",
    ])
    return "\n".join(lines)


def test_data_builder_is_pinned_full_scale_and_order_preserving():
    source = (ROOT / "vendor/data_scripts/natural-language-inference/prepare_data.py").read_text()
    assert "PER_CLASS" not in source
    assert "random" not in source
    assert "output_rows\": 549367" in source
    assert "output_rows\": 9824" in source
    assert "output_rows\": 9815" in source
    assert "output_rows\": 9832" in source
    assert "SNLI_REVISION" in source and "MNLI_REVISION" in source
    for _, digest in DATA.values():
        assert digest in source


def test_model_asset_staging_supports_offline_pinned_sources(monkeypatch, tmp_path):
    module = _load_prepare_module()
    source_root = tmp_path / "source"
    destination = tmp_path / "destination"
    source_root.mkdir()
    for filename in MODEL_ASSETS:
        (source_root / filename).write_bytes(filename.encode("ascii"))

    monkeypatch.setattr(
        module,
        "_sha256",
        lambda path: MODEL_ASSETS[path.name],
    )
    staged = module._stage_model_assets(
        destination, tmp_path / "cache", source_root
    )
    assert staged == MODEL_ASSETS
    for filename in MODEL_ASSETS:
        assert (destination / filename).read_bytes() == filename.encode("ascii")


def test_runtime_declares_full_protocol_and_one_training_pass():
    common = (ROOT / "vendor/natural-language-inference/common.py").read_text()
    assert 'TRAIN_BATCH = 32' in common
    assert 'MAX_SEQUENCE_LENGTH = 128' in common
    assert 'MAX_EPOCHS = 3' in common
    assert "TOTAL_OPTIMIZER_STEPS" in common
    assert "authenticate_model_assets()" in common
    assert "use_safetensors=True" in common
    assert "use_fast=True" in common
    for digest in MODEL_ASSETS.values():
        assert digest in common
    for harness in (ROOT / "vendor/natural-language-inference").glob("harness_*.py"):
        text = harness.read_text()
        assert 'add_argument("--domain"' not in text
        assert "load_training_data()" in text
        assert ("evaluate_all_domains(" in text
                or harness.name == "harness_finetune.py")


def test_ten_siblings_share_private_runtime_and_single_serial_command():
    assert len(TASKS) == 10
    score_hashes = set()
    for task in TASKS:
        config = json.loads((task / "config.json").read_text())
        assert config["rigorous_codebase"] is True
        assert config["pristine_manifest_mode"] == "scaffold"
        assert set(config["verifier_only_package_files"]) == VERIFIER_RUNTIME
        assert config["test_cmds"] == [{
            "cmd": "scripts/run.sh",
            "label": "nli",
            "group": 1,
            "compute": 1,
            "time": "6:00:00",
            "mem": 64,
            "package": "natural-language-inference",
        }]
        assert config["calibration_status"] == (
            "fresh_fullscale_h20_anchors_and_static_sibling_audit_20260711"
        )
        assert config["calibration_protocol"] == "nli-full-snli-distilbert-v1"
        assert config["calibration_representative_task"] == "nli-finetune"
        assert config["calibration_static_sibling_count"] == 10
        assert config["calibration_train_rows"] == 549367
        assert config["calibration_eval_rows"] == 29471
        assert config["calibration_epochs"] == 3
        assert config["calibration_optimizer_steps"] == 51504
        assert config["calibration_anchor_image"].endswith(
            "@sha256:054dc61204ab8731eccaddcc079e9a00a3783867e3a2f2a9f156a52204c1a955"
        )
        assert config["calibration_runtime_image"].endswith(
            "@sha256:3413891ea22deecf213026a9c34403d65133702286042175057dcb88f329e7e6"
        )
        assert all(item["filename"] != "natural-language-inference/common.py"
                   for item in config["files"])
        script = (task / "scripts/run.sh").read_text()
        assert "NLI_COMMAND_DONE rc=%s" in script
        assert "CUDA_VISIBLE_DEVICES" not in script
        assert not re.search(r"\b(?:pip|conda|apt-get|git clone|wget|curl)\b", script)
        parser_module = _load_parser(task.name)
        assert parser_module.__class__.__module__.startswith("parser_nli-")
        score_hashes.add(hashlib.sha256((task / "score_spec.py").read_bytes()).hexdigest())
    assert len(score_hashes) == 1
    score_source = (TASKS[0] / "score_spec.py").read_text()
    assert "bounded_power" in score_source
    assert "leaderboard" in score_source and "never reads" in score_source
    assert "bl_best" not in score_source and "bl_worst" not in score_source


def test_fail_closed_parser_rejects_every_incomplete_terminal_state():
    for task_name in TASK_PROTOCOL:
        parser = _load_parser(task_name)
        valid = _valid_log(task_name)
        mode = TASK_PROTOCOL[task_name][1]
        architecture, head_params = EXPECTED_MODELS[task_name][mode]
        total_params = 66362880 + head_params
        protocol_cap_match = re.search(r"NLI_PROTOCOL .* max_length=(\d+) seed=42", valid)
        assert protocol_cap_match is not None
        tokenizer_cap = int(protocol_cap_match.group(1))
        wrong_cap = 64 if tokenizer_cap != 64 else 128
        train_line = next(
            line for line in valid.splitlines() if line.startswith("NLI_TRAIN mode=")
        )
        train_done_line = (
            "NLI_TRAIN_DONE epochs=3 optimizer_steps=51504 expected_steps=51504"
        )
        expected_weight_decay = (
            REGULARIZATION[mode][1]
            if task_name == "nli-regularization"
            else 0.01
        )
        assert set(parser.parse("nli", valid).metrics) == {
            "acc_snli", "acc_mnli_m", "acc_mnli_mm"
        }
        wrong_task = next(name for name in TASK_PROTOCOL if name != task_name)
        corruptions = [
            valid.replace("NLI_DEVICE type=cuda visible=1\n", ""),
            valid.replace(
                "NLI_TRAIN_DONE epochs=3 optimizer_steps=51504 expected_steps=51504\n",
                "",
            ),
            valid.replace("NLI_EVAL_DONE settings=3 rows=29471\n", ""),
            valid.replace("NLI_COMMAND_DONE rc=0", "NLI_COMMAND_DONE rc=7"),
            valid.replace(
                "optimizer_steps=51504 expected_steps=51504",
                "optimizer_steps=51503 expected_steps=51504",
            ),
            valid.replace("NLI_METRICS setting=mnli_m acc=0.5 n_eval=9815\n", ""),
            valid + "\ntrailing output",
            "Traceback (most recent call last)\n" + valid,
            "[COMMAND FAILED rc=1]\n" + valid,
            "[BUDGET CHECK FAILED]\n" + valid,
            valid.replace(f"task={task_name}", f"task={wrong_task}"),
            valid.replace(
                train_line,
                train_line + "\nNLI_TRAIN\t" + train_line.removeprefix("NLI_TRAIN "),
                1,
            ),
            valid.replace(train_line, train_line + "\nNLI_TRAIN_ERROR stale", 1),
            valid.replace(train_line, train_line + "\nNLI_TRAINmode=stale", 1),
            valid.replace(train_done_line, train_done_line + "\n" + train_done_line, 1),
            valid.replace(
                f"weight_decay={expected_weight_decay:.8g} warmup_ratio=0.1",
                "weight_decay=0.02 warmup_ratio=0.1",
                1,
            ),
            valid.replace(
                f"architecture={architecture}",
                f"architecture={'siamese' if architecture == 'cross' else 'cross'}",
                1,
            ),
            valid.replace(f"head_params={head_params}", f"head_params={head_params + 1}", 1),
            valid.replace(
                f"total_params={total_params}", f"total_params={total_params + 1}", 1
            ),
            valid.replace(f"max_length={tokenizer_cap}", f"max_length={wrong_cap}"),
        ]
        if task_name == "nli-class-weighting":
            corruptions.extend([
                valid.replace(
                    "NLI_CLASS_WEIGHTS entailment=1 neutral=1 contradiction=1\n",
                    "",
                ),
                valid.replace(
                    "NLI_CLASS_WEIGHTS entailment=1 neutral=1 contradiction=1",
                    "NLI_CLASS_WEIGHTS entailment=0.25 neutral=0.25 contradiction=2",
                ),
            ])
        if task_name == "nli-regularization":
            corruptions.extend([
                valid.replace(
                    "NLI_REGULARIZATION dropout=0.1 weight_decay=0.01\n", ""
                ),
                valid.replace(
                    "NLI_REGULARIZATION dropout=0.1 weight_decay=0.01",
                    "NLI_REGULARIZATION dropout=0.7 weight_decay=0.01",
                ),
            ])
        corruptions.extend(
            valid.replace(digest, "0" * 64, 1)
            for digest in MODEL_ASSETS.values()
        )
        for raw_output in corruptions:
            assert parser.parse("nli", raw_output).metrics == {}
        assert parser.parse("wrong-label", valid).metrics == {}


def test_regularization_proof_binds_every_policy_value():
    parser = _load_parser("nli-regularization")
    for mode, (dropout, weight_decay) in REGULARIZATION.items():
        valid = _valid_log("nli-regularization", policy=mode)
        assert parser.parse("nli", valid).metrics
        assert parser.parse(
            "nli",
            valid.replace(
                f"NLI_REGULARIZATION dropout={dropout:.8g} "
                f"weight_decay={weight_decay:.8g}",
                "NLI_REGULARIZATION dropout=0.2 weight_decay=0.2",
            ),
        ).metrics == {}


def test_mode_specific_architecture_and_tokenizer_caps_are_accepted():
    for task_name, mode, cap in (
        ("nli-classifier-head", "mlp", 64),
        ("nli-interaction-head", "concat", 64),
        ("nli-pair-encoding", "siamese", 64),
        ("nli-pooling", "max", 64),
        ("nli-truncation", "len64", 64),
    ):
        parser = _load_parser(task_name)
        valid = _valid_log(task_name, policy=mode)
        assert f"max_length={cap} seed=42" in valid
        assert parser.parse("nli", valid).metrics


def test_package_uses_the_measured_immutable_runtime_image():
    package = json.loads(
        (ROOT / "vendor/pkg_configs/natural-language-inference/config.json").read_text()
    )
    assert package["mangrove_base_image"].endswith(
        "@sha256:3413891ea22deecf213026a9c34403d65133702286042175057dcb88f329e7e6"
    )
    assert package["extra_files"] == [{
        "src": "{data_root}/natural-language-inference",
        "dst": "/data/natural-language-inference",
    }]
    ready_files = package["data_deps"][0]["ready_files"]
    for filename in MODEL_ASSETS:
        assert any(path.endswith(f"/models/distilbert-base-uncased/{filename}")
                   for path in ready_files)


def test_agent_templates_have_valid_edit_ranges_and_no_answer_leakage():
    banned = re.compile(
        r"\b(?:weak|weaker|weakest|strong|stronger|best|outperform|"
        r"better|worse|underfits?|preferred value|default here)\b",
        re.IGNORECASE,
    )
    for task in TASKS:
        config = json.loads((task / "config.json").read_text())
        template = (task / "edits/custom_template.py").read_text()
        assert not banned.search(template), (task.name, banned.search(template).group())
        assert not re.search(r"\b(?:public|hidden)\b", (task / "task_description.md").read_text(), re.I)
        lines = template.splitlines()
        edit = config["files"][0]["edit"][0]
        region = "\n".join(lines[edit["start"] - 1:edit["end"]])
        assert "def build_" in region and "return {" in region


def test_every_baseline_edit_materializes_valid_python():
    for task in TASKS:
        native_lines = (task / "edits/custom_template.py").read_text().splitlines(
            keepends=True
        )
        for edit_path in sorted((task / "edits").glob("*.edit.py")):
            spec = importlib.util.spec_from_file_location(
                f"edit_{task.name}_{edit_path.stem}", edit_path
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            assert len(module.OPS) == 1
            operation = module.OPS[0]
            assert operation["op"] == "replace"
            start = operation["start_line"]
            end = operation["end_line"]
            materialized = "".join(
                native_lines[:start - 1]
                + [operation["content"].rstrip("\n") + "\n"]
                + native_lines[end:]
            )
            compile(materialized, str(edit_path), "exec")


def test_baseline_free_score_has_no_positive_missing_metric_fallback():
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import load_expanded_spec, score_record_details

    task_dir = ROOT / "tasks/nli-finetune"
    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
    assert spec is not None
    actual = {
        "acc_snli": 0.57206840,
        "acc_mnli_m": 0.39286806,
        "acc_mnli_mm": 0.38415378,
    }
    score, settings, valid = score_record_details(spec, actual, anchors)
    assert valid is True
    assert score == pytest.approx(0.3668595303)
    assert all(setting.valid for setting in settings)

    for broken in (
        {"acc_snli": actual["acc_snli"], "acc_mnli_m": actual["acc_mnli_m"]},
        {**actual, "acc_mnli_m": float("nan")},
        {**actual, "acc_mnli_mm": float("inf")},
    ):
        broken_score, _, broken_valid = score_record_details(spec, broken, anchors)
        assert broken_valid is False
        assert broken_score == 0.0
