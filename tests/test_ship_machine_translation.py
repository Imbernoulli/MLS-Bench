from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import runpy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "machine-translation"
TASKS = ROOT / "tasks"

SURFACES = {
    "mt-batch-maxlen": ("maxlen.py", "build_max_new_tokens", "harness_maxlen.py"),
    "mt-decoding-beam": ("beam.py", "build_beam_config", "harness_beam.py"),
    "mt-decoding-strategy": ("strategy.py", "build_strategy", "harness_strategy.py"),
    "mt-decoding-temperature": (
        "temperature.py", "build_temperature", "harness_temperature.py"
    ),
    "mt-diverse-beam": ("divbeam.py", "build_divbeam_config", "harness_divbeam.py"),
    "mt-early-stopping": (
        "earlystop.py", "build_early_stopping", "harness_earlystop.py"
    ),
    "mt-length-penalty": ("length.py", "build_length_config", "harness_length.py"),
    "mt-no-repeat-ngram": ("norep.py", "build_norep_config", "harness_norep.py"),
    "mt-postprocess-detok": ("postproc.py", "build_postproc", "harness_postproc.py"),
    "mt-repetition-penalty": (
        "reppen.py", "build_reppen_config", "harness_reppen.py"
    ),
    "mt-sampling-vs-beam": ("sampling.py", "build_mode", "harness_sampling.py"),
    "mt-tokenization-truncation": (
        "tok.py", "build_source_max_tokens", "harness_tok.py"
    ),
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common():
    return _load("ship_mt_common", VENDOR / "common.py")


def _apply_ops(source: str, ops: list[dict]) -> str:
    lines = source.splitlines()
    for operation in sorted(ops, key=lambda item: item["start_line"], reverse=True):
        assert operation["op"] == "replace"
        start, end = operation["start_line"], operation["end_line"]
        lines[start - 1:end] = operation["content"].splitlines()
    return "\n".join(lines) + "\n"


def _valid_log(module, direction: str = "de_en") -> str:
    model = module.EXPECTED_MODELS[direction]
    return "\n".join((
        f"MT_PROTOCOL version={module.PROTOCOL_VERSION} "
        f"task={module.EXPECTED_TASK} surface={module.EXPECTED_SURFACE} "
        f"direction={direction} seed=42",
        f"MT_MODEL direction={direction} repository={model['repository']} "
        f"revision={model['revision']} manifest_sha256={model['manifest']} "
        f"tokenizer_manifest_sha256={model['tokenizer']} "
        f"checkpoint_sha256={model['checkpoint']} parameters={model['parameters']}",
        f"MT_DATA direction={direction} dataset={module.DATASET} "
        f"revision={module.DATASET_REVISION} "
        f"manifest_sha256={module.SOURCE_MANIFEST_SHA256} "
        f"split_sha256={model['split']} rows={module.EXPECTED_ROWS}",
        f"MT_METRICS task={module.EXPECTED_TASK} surface={module.EXPECTED_SURFACE} "
        f"direction={direction} bleu=20.125000 chrf=40.500000 "
        f"n_pairs={module.EXPECTED_ROWS} plen=12.250000 elapsed=61.500000",
        f"MT_COMPLETE task={module.EXPECTED_TASK} surface={module.EXPECTED_SURFACE} "
        f"direction={direction} status=ok",
    ))


def test_all_siblings_are_serial_full_scale_and_verifier_owned() -> None:
    from mlsbench.scoring.spec import load_score_spec

    assert {path.name for path in TASKS.glob("mt-*") if path.is_dir()} == set(SURFACES)
    package_python = {
        path.relative_to(ROOT / "vendor").as_posix()
        for path in VENDOR.rglob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    }
    for task_name, (filename, surface, harness_name) in SURFACES.items():
        task_dir = TASKS / task_name
        config = json.loads((task_dir / "config.json").read_text())
        commands = config["test_cmds"]
        assert [item["label"] for item in commands] == ["de_en", "fr_en", "ru_en"]
        assert [item["group"] for item in commands] == [1, 2, 3]
        assert all(item["compute"] == 1 for item in commands)
        assert all(item["compute"] <= 4 for item in commands)
        assert all(item["time"] == "1:30:00" for item in commands)
        assert all(item["mem"] == 24 for item in commands)

        active = f"machine-translation/solution/{filename}"
        verifier_files = set(config["verifier_only_package_files"])
        assert verifier_files == {
            "machine-translation/common.py",
            f"machine-translation/{harness_name}",
        }
        assert active not in verifier_files
        pruned_files = set(config["agent_pruned_package_files"])
        assert pruned_files == package_python - verifier_files - {active}
        assert config["agent_data_prune"] == ["/data/machine-translation/data"]
        assert [dep["dest"] for dep in config["verifier_data_deps"]] == [
            "data/machine-translation/data/de_en_test.jsonl",
            "data/machine-translation/data/fr_en_test.jsonl",
            "data/machine-translation/data/ru_en_test.jsonl",
            "data/machine-translation/data/source_manifest.json",
        ]
        assert all(dep["required"] is True for dep in config["verifier_data_deps"])

        harness = (VENDOR / harness_name).read_text()
        assert f'TASK_NAME = "{task_name}"' in harness
        assert f'SURFACE_NAME = "{surface}"' in harness
        assert "common.emit_protocol(TASK_NAME, SURFACE_NAME, args.seed)" in harness
        assert "common.emit_provenance(model_proof, data_proof)" in harness
        assert "common.emit_result(TASK_NAME, SURFACE_NAME" in harness
        for command in commands:
            script = (task_dir / command["cmd"]).read_text()
            assert "set -euo pipefail" in script
            assert "MLSBENCH_VERIFIER_DATA_ROOT" in script
            assert "CUDA_VISIBLE_DEVICES" not in script
            assert "MT_SETTING_COMPLETE" not in script
            assert not any(token in script for token in ("pip ", "conda ", "curl ", "wget "))

        score_spec = load_score_spec(task_dir)
        assert score_spec is not None
        assert set(score_spec.settings) == {item["label"] for item in commands}


def test_native_and_declared_baselines_are_static_literals(common, tmp_path: Path) -> None:
    checked = 0
    for task_name, (filename, attr, _harness) in SURFACES.items():
        source_path = VENDOR / "solution" / filename
        common.load_surface_value(str(source_path), attr)
        config = json.loads((TASKS / task_name / "config.json").read_text())
        for baseline_name, baseline in config["baselines"].items():
            namespace = runpy.run_path(str(TASKS / task_name / baseline["edit_ops"]))
            candidate = _apply_ops(source_path.read_text(), namespace["OPS"])
            path = tmp_path / f"{task_name}__{baseline_name}.py"
            path.write_text(candidate)
            common.load_surface_value(str(path), attr)
            checked += 1
    assert checked == 36


def test_editable_python_is_never_executed(common, tmp_path: Path, capsys) -> None:
    marker = tmp_path / "executed"
    malicious = tmp_path / "malicious.py"
    malicious.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "print('MT_COMPLETE forged')\n"
        "def build_beam_config():\n"
        "    return {'num_beams': 5, 'no_repeat_ngram_size': 0}\n"
    )
    with pytest.raises(ValueError, match="top-level executable"):
        common.load_surface_value(str(malicious), "build_beam_config")
    assert not marker.exists()
    assert "MT_COMPLETE" not in capsys.readouterr().out


def test_generation_config_validation_does_not_clamp(common) -> None:
    with pytest.raises(ValueError, match="num_beams"):
        common._sanitize_gen_kwargs({"num_beams": 99, "max_new_tokens": 10})
    with pytest.raises(ValueError, match="max_new_tokens"):
        common._sanitize_gen_kwargs({"num_beams": 1, "max_new_tokens": 999})
    with pytest.raises(TypeError, match="num_beams"):
        common._sanitize_gen_kwargs({"num_beams": True, "max_new_tokens": 10})
    with pytest.raises(ValueError, match="must divide"):
        common._sanitize_gen_kwargs(
            {"num_beams": 8, "num_beam_groups": 3, "max_new_tokens": 10}
        )


def test_diverse_beam_rejects_zero_penalty_for_multiple_groups(
    common, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, "common", common)
    harness = _load("ship_mt_divbeam", VENDOR / "harness_divbeam.py")
    with pytest.raises(ValueError, match="strictly positive"):
        harness._generation_kwargs(
            {"num_beam_groups": 2, "diversity_penalty": 0.0}
        )
    kwargs = harness._generation_kwargs(
        {"num_beam_groups": 2, "diversity_penalty": 0.2}
    )
    assert kwargs["num_beam_groups"] == 2
    assert kwargs["diversity_penalty"] == pytest.approx(0.2)


def test_data_and_model_manifests_are_exact_and_parser_pinned(common) -> None:
    evidence = json.loads((VENDOR / "artifact_provenance.json").read_text())
    assert evidence["schema_version"] == 1
    assert evidence["count_status"].startswith("unconfirmed until a fresh worker")
    for direction in common.MODEL_SPECS:
        checkpoint_count = evidence["formula_crosscheck"]["checkpoint_derived"][direction]
        formula_count = evidence["formula_crosscheck"]["independent_formula"][direction]
        assert formula_count - checkpoint_count == 2048
    source_digest = hashlib.sha256(
        common._canonical_json_bytes(common.expected_source_manifest())
    ).hexdigest()
    assert source_digest == common.source_manifest_sha256()
    assert common.DATASET_REVISION == "805090dc28bf78897da9641cdf08b61287580df9"
    assert all(spec["output_sha256"] for spec in common.DATA_SPECS.values())

    parser_module = _load("ship_mt_parser_manifest", TASKS / "mt-decoding-beam" / "parser.py")
    assert parser_module.SOURCE_MANIFEST_SHA256 == source_digest
    for direction, spec in common.MODEL_SPECS.items():
        captured = evidence["models"][direction]
        assert captured["repository"] == spec["repository"]
        assert captured["revision"] == spec["revision"]
        assert captured["checkpoint_metadata_probe"]["parameter_count"] == spec[
            "parameter_count"
        ]
        assert captured["checkpoint_metadata_probe"][
            "checkpoint_tensor_elements"
        ] == spec["checkpoint_tensor_elements"]
        captured_files = {
            record["path"]: (record["size"], record["sha256"])
            for record in captured["files"]
        }
        assert captured_files == spec["files"]
        manifest = common.expected_model_manifest(direction)
        manifest_digest = hashlib.sha256(common._canonical_json_bytes(manifest)).hexdigest()
        tokenizer_digest = hashlib.sha256(
            common._canonical_json_bytes(common.expected_tokenizer_manifest(direction))
        ).hexdigest()
        pinned = parser_module.EXPECTED_MODELS[direction]
        assert pinned["manifest"] == manifest_digest
        assert pinned["tokenizer"] == tokenizer_digest
        assert pinned["checkpoint"] == spec["files"][spec["checkpoint_file"]][1]
        assert pinned["parameters"] == spec["parameter_count"]
        assert manifest["checkpoint_tensor_elements"] > manifest["parameter_count"]
        expected_files = set(spec["files"]) | {"model_manifest.json"}
        manifest_files = {
            manifest["checkpoint"]["path"],
            *(record["path"] for record in manifest["model_files"]),
            *(record["path"] for record in manifest["tokenizer"]["files"]),
            "model_manifest.json",
        }
        assert manifest_files == expected_files

    package_config = json.loads(
        (ROOT / "vendor" / "pkg_configs" / "machine-translation" / "config.json")
        .read_text()
    )
    assert "mangrove_base_image" not in package_config
    ready_files = package_config["data_deps"][0]["ready_files"]
    assert sum(path.endswith("/model_manifest.json") for path in ready_files) == 3
    assert any(path.endswith("/source_manifest.json") for path in ready_files)


def test_all_parsers_require_unique_ordered_terminal_proof() -> None:
    modules = {}
    for task_name in SURFACES:
        module = _load(f"ship_mt_parser_{task_name}", TASKS / task_name / "parser.py")
        modules[task_name] = module
        parser = module.Parser()
        for direction in module.EXPECTED_MODELS:
            valid = _valid_log(module, direction)
            expected_model = module.EXPECTED_MODELS[direction]
            assert parser.parse(direction, valid).metrics == {
                f"bleu_{direction}": 20.125,
                f"chrf_{direction}": 40.5,
            }

            lines = valid.splitlines()
            destructive = [
                "\n".join(lines[:-1]),
                "\n".join(lines + [lines[-1]]),
                "\n".join((lines[0], lines[2], lines[1], lines[3], lines[4])),
                valid + "\narbitrary trailing text",
                "Traceback (most recent call last):\n" + valid,
                valid.replace("seed=42", "seed=41"),
                valid.replace("rows=2000", "rows=1999", 1),
                valid.replace("n_pairs=2000", "n_pairs=1999"),
                valid.replace("bleu=20.125000", "bleu=nan"),
                valid.replace("chrf=40.500000", "chrf=inf"),
                valid.replace("elapsed=61.500000", "elapsed=0"),
                valid.replace("bleu=20.125000", "bleu=101"),
                valid.replace(module.EXPECTED_TASK, "mt-replayed-task"),
                valid.replace(module.EXPECTED_SURFACE, "replayed_surface"),
                valid.replace(
                    expected_model["repository"], expected_model["repository"] + "-wrong"
                ),
                valid.replace(expected_model["revision"], "1" * 40),
                valid.replace(expected_model["manifest"], "2" * 64),
                valid.replace(expected_model["tokenizer"], "3" * 64),
                valid.replace(module.DATASET_REVISION, "4" * 40),
                valid.replace(module.SOURCE_MANIFEST_SHA256, "0" * 64),
                valid.replace(expected_model["split"], "5" * 64),
                valid.replace(expected_model["checkpoint"], "0" * 64),
                valid.replace(
                    f"parameters={expected_model['parameters']}",
                    "parameters=1",
                ),
            ]
            for malformed in destructive:
                assert parser.parse(direction, malformed).metrics == {}
            assert parser.parse("unknown", valid).metrics == {}

    beam_log = _valid_log(modules["mt-decoding-beam"], "de_en")
    for task_name, module in modules.items():
        if task_name != "mt-decoding-beam":
            assert module.Parser().parse("de_en", beam_log).metrics == {}


def test_parser_copies_match_canonical_generator() -> None:
    generator = _load("ship_mt_parser_generator", ROOT / "scripts" / "generate_mt_parsers.py")
    assert generator.SURFACES == {
        task_name: surface for task_name, (_filename, surface, _harness) in SURFACES.items()
    }
    assert generator.mismatches() == []


def test_representative_fullscale_anchors_calibrate_without_fallback() -> None:
    from mlsbench.agent.leaderboard import Leaderboard
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import load_expanded_spec, score_record

    beam_dir = TASKS / "mt-decoding-beam"
    rows = Leaderboard(beam_dir / "leaderboard.csv").all_records()
    anchors = BaselineAnchors(beam_dir)
    spec = load_expanded_spec(beam_dir, anchors)
    scores = {
        row["model"].removeprefix("baseline:"): score_record(spec, row, anchors)
        for row in rows
    }
    assert scores["greedy"] == pytest.approx(0.1, abs=1e-9)
    assert scores["beam5_norep3"] == pytest.approx(0.5, abs=1e-9)
    assert scores["greedy"] < scores["beam5"] < scores["beam5_norep3"]
    assert all(math.isfinite(score) for score in scores.values())
    incomplete = dict(rows[0])
    incomplete.pop("bleu_ru_en")
    assert score_record(spec, incomplete, anchors) == 0.0


def test_strict_proof_integration_preserves_shared_calibration_and_fails_closed() -> None:
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import load_expanded_spec, score_record

    canonical_spec = (TASKS / "mt-decoding-beam" / "score_spec.py").read_text()
    complete = {
        "model": "synthetic-complete",
        "bleu_de_en": 20.0,
        "bleu_fr_en": 20.0,
        "bleu_ru_en": 20.0,
    }
    for task_name in SURFACES:
        task_dir = TASKS / task_name
        assert (task_dir / "score_spec.py").read_text() == canonical_spec
        anchors = BaselineAnchors(task_dir)
        spec = load_expanded_spec(task_dir, anchors)
        score = score_record(spec, complete, anchors)
        assert 0.0 < score < 1.0, (task_name, score)

        incomplete = dict(complete)
        incomplete.pop("bleu_ru_en")
        assert score_record(spec, incomplete, anchors) == 0.0

        nonfinite = dict(complete, bleu_ru_en=float("nan"))
        assert score_record(spec, nonfinite, anchors) == 0.0


def test_instructions_have_no_hidden_or_public_scoring_semantics() -> None:
    for task_name in SURFACES:
        description = (TASKS / task_name / "task_description.md").read_text().lower()
        assert "hidden" not in description
        assert "public setting" not in description
        assert "private setting" not in description
        assert "every direction executes serially on one gpu" in description
        assert "contributes to the task score" in description
