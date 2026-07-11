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
    package_files = {
        path.relative_to(ROOT / "vendor").as_posix()
        for path in VENDOR.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }
    evidence_files = {
        "machine-translation/artifact_provenance.json",
        "machine-translation/image_provenance.json",
        "machine-translation/representative_logs/de_en.log",
        "machine-translation/representative_logs/fr_en.log",
        "machine-translation/representative_logs/ru_en.log",
        "machine-translation/representative_probe.json",
        "machine-translation/runtime_probe.json",
        "machine-translation/surface_probe_early_stopping.json",
        "machine-translation/surface_probe_maxlen.json",
        "machine-translation/surface_probe_postprocess.json",
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
        public_files = {"machine-translation/__init__.py", active}
        assert pruned_files == package_files - verifier_files - public_files
        assert evidence_files <= pruned_files
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
    image_evidence = json.loads((VENDOR / "image_provenance.json").read_text())
    runtime_probe = json.loads((VENDOR / "runtime_probe.json").read_text())
    image = (
        "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
        "mlsbench-harbor-machine-translation@"
        "sha256:9290a9de78e9b39ec20518b75788d6207880c134c78d05b931870d18eb4fc97e"
    )
    assert evidence["schema_version"] == 1
    assert image_evidence["schema_version"] == 1
    assert image_evidence["final_image"] == image
    assert image_evidence["strict_probe"]["evidence_file"] == "runtime_probe.json"
    assert image_evidence["strict_probe"]["rc"] == 0
    assert image_evidence["layer"]["digest"] == (
        "sha256:7c1c38923723e35ca136dc35faf0a19db277bd9554accb14f0ae20108ee1d114"
    )
    assert image_evidence["manifest"]["recorded_base_digest"] == (
        image_evidence["base_image"].rsplit("@", 1)[1]
    )
    assert "/data/machine-translation/data/flores_de_en_test.jsonl" in (
        image_evidence["layer"]["removals"]
    )
    assert evidence["count_status"].startswith("confirmed by strict actual-load")
    assert evidence["runtime_probe"] == {
        "evidence_file": "runtime_probe.json",
        "image": image,
        "persistent_worker_path": (
            "/home/lvbohan/mt-proof-probe-c58e4a9e7/output-final-strict-v4"
        ),
        "rc": 0,
        "worker": "dev-qghqc-53440-worker-0",
        "worker_gpu": "NVIDIA H20-3e",
        "worker_zone": "m4h20",
    }
    assert runtime_probe["image_ref"] == image
    assert runtime_probe["image_digest"] == image.rsplit("@", 1)[1]
    assert runtime_probe["worker"]["hostname"] == "dev-qghqc-53440-worker-0"
    assert runtime_probe["worker"]["cuda_device_count"] == 1
    assert runtime_probe["worker"]["cuda_device_names"] == ["NVIDIA H20-3e"]
    data_probe = runtime_probe["data"]
    expected_data_inventory = {
        common.DIRECTIONS[direction][1]: {
            "sha256": spec["output_sha256"],
        }
        for direction, spec in common.DATA_SPECS.items()
    }
    source_manifest_bytes = common._canonical_json_bytes(
        common.expected_source_manifest()
    )
    expected_data_inventory["source_manifest.json"] = {
        "sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
    }
    observed_data_inventory = {
        record["path"]: {"sha256": record["sha256"]}
        for record in data_probe["top_level_inventory"]
        if not record["symlink"]
    }
    assert len(observed_data_inventory) == len(data_probe["top_level_inventory"])
    assert observed_data_inventory == expected_data_inventory
    assert data_probe["source_manifest_sha256"] == common.source_manifest_sha256()
    for direction, proof in data_probe["directions"].items():
        assert proof == {
            "dataset": common.DATASET_ID,
            "direction": direction,
            "manifest_sha256": common.source_manifest_sha256(),
            "revision": common.DATASET_REVISION,
            "rows": common.OFFICIAL_TEST_PAIRS,
            "split_sha256": common.DATA_SPECS[direction]["output_sha256"],
        }
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

        actual = runtime_probe["models"][direction]
        assert actual["direction"] == direction
        assert actual["expected_runtime_files_match"] is True
        assert actual["checkpoint_candidates"] == [spec["checkpoint_file"]]
        assert actual["selected_checkpoint"] == spec["checkpoint_file"]
        assert actual["loader_checkpoint_paths"] == [
            f"/data/machine-translation/models/{common.DIRECTIONS[direction][0]}/"
            f"{spec['checkpoint_file']}"
        ]
        assert actual["model_class"].endswith(".MarianMTModel")
        assert actual["tokenizer_class"].endswith(".MarianTokenizer")
        assert actual["config_class"].endswith(".MarianConfig")
        assert actual["vocab_size"] == actual["tokenizer_length"] == spec["vocab_size"]
        assert actual["parameter_tensor_count"] == 255
        assert actual["state_dict_tensor_count"] == 259
        assert actual["parameter_numel"] == spec["parameter_count"]
        assert actual["final_logits_bias"]["numel"] == (
            spec["checkpoint_tensor_elements"] - spec["parameter_count"]
        )
        assert actual["parameter_plus_final_logits_bias_numel"] == spec[
            "checkpoint_tensor_elements"
        ]
        assert all(actual["tied_aliases"].values())
        assert all(not values for values in actual["loading_info"].values())
        expected_inventory = {
            name: {"size": size, "sha256": digest}
            for name, (size, digest) in spec["files"].items()
        }
        manifest_bytes = common._canonical_json_bytes(manifest)
        expected_inventory["model_manifest.json"] = {
            "size": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        observed_inventory = {
            record["path"]: {
                "size": record["size"],
                "sha256": record["sha256"],
            }
            for record in actual["top_level_inventory"]
            if record["kind"] == "file" and not record["symlink"]
        }
        assert len(observed_inventory) == len(actual["top_level_inventory"])
        assert observed_inventory == expected_inventory

    package_config = json.loads(
        (ROOT / "vendor" / "pkg_configs" / "machine-translation" / "config.json")
        .read_text()
    )
    assert package_config["mangrove_base_image"] == image
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


def test_representative_worker_run_covers_all_settings_with_real_parser() -> None:
    evidence = json.loads((VENDOR / "representative_probe.json").read_text())
    package_config = json.loads(
        (ROOT / "vendor" / "pkg_configs" / "machine-translation" / "config.json")
        .read_text()
    )
    assert evidence["schema_version"] == 1
    assert evidence["accepted"] is True
    assert evidence["task"] == "mt-decoding-beam"
    assert evidence["surface"] == "build_beam_config"
    assert evidence["image_ref"] == package_config["mangrove_base_image"]
    execution = evidence["execution"]
    assert execution["overall_rc"] == 0
    assert execution["gpu_count"] == 1
    assert execution["mode"] == "serial"
    assert execution["requested_zone"] == "m4h20"

    runner = ROOT / execution["runner_script"]
    runner_source = runner.read_text()
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == execution[
        "runner_script_sha256"
    ]
    assert 'export MT_DIR="${direction}"' in runner_source
    assert "MT_DIRECTION" not in runner_source
    assert "direction_count" in runner_source

    discarded = evidence["discarded_predecessor"]
    assert discarded["accepted"] is False
    assert discarded["overall_rc"] == 0
    assert "MT_DIRECTION" in discarded["reason"]

    parser_module = _load(
        "ship_mt_representative_parser", TASKS / "mt-decoding-beam" / "parser.py"
    )
    assert set(evidence["settings"]) == {"de_en", "fr_en", "ru_en"}
    observed_hashes = set()
    for direction, record in evidence["settings"].items():
        log_path = VENDOR / record["log_path"]
        raw = log_path.read_text()
        digest = hashlib.sha256(log_path.read_bytes()).hexdigest()
        observed_hashes.add(digest)
        assert digest == record["log_sha256"]
        assert len(raw.splitlines()) == record["protocol_lines"] == 5
        assert record["rc"] == 0
        assert record["wall_seconds"] > 0
        parsed = parser_module.Parser().parse(direction, raw)
        assert parsed.metrics == record["metrics"]
    assert len(observed_hashes) == 3


def test_worker_spotchecks_keep_question_surfaces_distinct(
    common, tmp_path: Path, monkeypatch
) -> None:
    image = json.loads(
        (ROOT / "vendor" / "pkg_configs" / "machine-translation" / "config.json")
        .read_text()
    )["mangrove_base_image"]
    probes = {
        name: json.loads((VENDOR / f"surface_probe_{name}.json").read_text())
        for name in ("postprocess", "early_stopping", "maxlen")
    }
    for name, probe in probes.items():
        assert probe["schema_version"] == 1
        assert probe["surface"] == name
        assert probe["image_ref"] == image
        assert probe["direction"] == "de_en"
        assert probe["rows"] == common.OFFICIAL_TEST_PAIRS
        assert probe["model_proof"]["manifest_sha256"] == common.model_manifest_sha256(
            "de_en"
        )
        assert probe["data_proof"]["split_sha256"] == common.DATA_SPECS["de_en"][
            "output_sha256"
        ]
        for record in probe["records"].values():
            assert record["rows"] == common.OFFICIAL_TEST_PAIRS
            assert math.isfinite(record["bleu"])
            assert math.isfinite(record["chrf"])

    postprocess = probes["postprocess"]
    assert postprocess["pairwise_output_differences"]["identity_vs_normalize"] == 0
    assert postprocess["records"]["identity"]["prediction_sha256"] == (
        postprocess["records"]["normalize"]["prediction_sha256"]
    )
    assert postprocess["pairwise_output_differences"]["identity_vs_lowercase"] > 0
    assert postprocess["pairwise_output_differences"]["identity_vs_strip_punct"] > 0
    monkeypatch.setitem(sys.modules, "common", common)
    postprocess_harness = _load(
        "ship_mt_postprocess", VENDOR / "harness_postproc.py"
    )
    assert postprocess_harness._VALID == {"normalize", "lowercase", "strip_punct"}

    early_stopping = probes["early_stopping"]
    assert all(
        difference > 0
        for difference in early_stopping["pairwise_output_differences"].values()
    )
    assert len({
        record["prediction_sha256"]
        for record in early_stopping["records"].values()
    }) == 3

    maxlen = probes["maxlen"]
    assert maxlen["pairwise_output_differences"]["m10_vs_m32"] > 0
    assert maxlen["pairwise_output_differences"]["m10_vs_m128"] > 0
    assert maxlen["pairwise_output_differences"]["m32_vs_m128"] > 0
    assert maxlen["pairwise_output_differences"][
        "m128_vs_length_norm1_m128"
    ] == 0
    assert maxlen["records"]["m128"]["prediction_sha256"] == maxlen["records"][
        "length_norm1_m128"
    ]["prediction_sha256"]

    length_source = (VENDOR / "solution" / "length.py").read_text()
    assert set(common.load_surface_value(
        str(VENDOR / "solution" / "length.py"), "build_length_config"
    )) == {"length_penalty"}
    config = json.loads((TASKS / "mt-length-penalty" / "config.json").read_text())
    for baseline in config["baselines"].values():
        operations = runpy.run_path(
            str(TASKS / "mt-length-penalty" / baseline["edit_ops"])
        )["OPS"]
        candidate = tmp_path / f"length-{Path(baseline['edit_ops']).stem}.py"
        candidate.write_text(_apply_ops(length_source, operations))
        assert set(common.load_surface_value(
            str(candidate), "build_length_config"
        )) == {"length_penalty"}
    length_harness = (VENDOR / "harness_length.py").read_text()
    assert '{"length_penalty"}' in length_harness
    assert '"max_new_tokens": 128' in length_harness

    forbidden = ("measured order", "scores highest", "sweet spot", "degenerate")
    for task_name in (
        "mt-postprocess-detok",
        "mt-early-stopping",
        "mt-batch-maxlen",
        "mt-length-penalty",
    ):
        template = (TASKS / task_name / "edits" / "custom_template.py").read_text().lower()
        assert not any(token in template for token in forbidden)


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
