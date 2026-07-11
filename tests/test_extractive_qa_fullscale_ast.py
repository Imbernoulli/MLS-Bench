from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "extractive-qa"
TASK_ROOT = ROOT / "tasks"

SURFACES = {
    "qa-casing": ("casing.py", "harness_casing.py", "build_casing"),
    "qa-doc-stride": ("doc_stride.py", "harness_doc_stride.py", "build_doc_stride"),
    "qa-encoding-order": (
        "encoding_order.py",
        "harness_encoding_order.py",
        "build_encoding_order",
    ),
    "qa-max-answer-length": (
        "max_answer_length.py",
        "harness_max_answer_length.py",
        "build_max_answer_length",
    ),
    "qa-max-seq-len": (
        "max_seq_len.py",
        "harness_max_seq_len.py",
        "build_max_seq_len",
    ),
    "qa-n-best": ("n_best.py", "harness_n_best.py", "build_n_best"),
    "qa-null-threshold": (
        "null_threshold.py",
        "harness_null_threshold.py",
        "build_null_threshold",
    ),
    "qa-question-inclusion": (
        "question_inclusion.py",
        "harness_question_inclusion.py",
        "build_question_mode",
    ),
    "qa-span-aggregation": (
        "span_aggregation.py",
        "harness_span_aggregation.py",
        "build_span_aggregation",
    ),
    "qa-span-decoding": (
        "span_decoding.py",
        "harness_span_decoding.py",
        "build_decoder",
    ),
}

ANSWER_LABELS = ["squad", "newsqa", "hotpotqa", "naturalq"]
NULL_LABELS = ["part0", "part1", "part2"]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def common():
    return _load("extractive_qa_fullscale_common", VENDOR / "common.py")


def _apply_ops(source: str, operations: list[dict]) -> str:
    lines = source.splitlines()
    for operation in sorted(operations, key=lambda item: item["start_line"], reverse=True):
        assert operation["op"] == "replace"
        start = int(operation["start_line"])
        end = int(operation["end_line"])
        lines[start - 1 : end] = operation["content"].splitlines()
    return "\n".join(lines) + "\n"


def _proof(parser_module, label: str, *, f1: str = "61.25") -> str:
    expected = parser_module._EXPECTED[label]
    n = expected["n"]
    n_ans = expected["n_ans"]
    n_noans = expected["n_noans"]
    n_features = expected.get("n_features", n + 7)
    feature_sha256 = "1" * 64
    f1_noans = f1 if n_noans else "0.0"
    em_noans = "55.0" if n_noans else "0.0"
    protocol = (
        "QA_PROTOCOL protocol=qa-official-full-v2 "
        f"task={parser_module._EXPECTED_TASK} "
        f"surface={parser_module._EXPECTED_SURFACE} "
        f"dataset={expected['dataset']} dataset_sha256={expected['sha256']} "
        f"n={n} n_ans={n_ans} n_noans={n_noans} "
        f"model_revision={parser_module._MODEL_REVISION} "
        f"model_files={parser_module._MODEL_FILE_COUNT} "
        f"model_manifest_sha256={parser_module._MODEL_MANIFEST_SHA256} "
        f"model_weight_sha256={parser_module._MODEL_SHA256} "
        f"model_params={parser_module._MODEL_PARAMETER_COUNT} seed=42 device=cuda "
        f"feature_config_sha256={feature_sha256}"
    )
    metric = (
        "QA_METRICS protocol=qa-official-full-v2 "
        f"task={parser_module._EXPECTED_TASK} "
        f"surface={parser_module._EXPECTED_SURFACE} "
        f"dataset={expected['dataset']} dataset_sha256={expected['sha256']} "
        f"model_manifest_sha256={parser_module._MODEL_MANIFEST_SHA256} "
        f"feature_config_sha256={feature_sha256} "
        f"f1={f1} em=55.0 f1_ans={f1} em_ans=55.0 "
        f"f1_noans={f1_noans} em_noans={em_noans} "
        f"n={n} n_ans={n_ans} n_noans={n_noans} "
        f"n_features={n_features} "
        "elapsed=123.5"
    )
    completion = (
        "QA_COMPLETE protocol=qa-official-full-v2 "
        f"task={parser_module._EXPECTED_TASK} "
        f"surface={parser_module._EXPECTED_SURFACE} "
        f"dataset={expected['dataset']} dataset_sha256={expected['sha256']} "
        f"model_manifest_sha256={parser_module._MODEL_MANIFEST_SHA256} "
        f"feature_config_sha256={feature_sha256} n={n} n_features={n_features} "
        f"predictions={n} status=ok"
    )
    return "\n".join((protocol, metric, completion))


def test_exactly_ten_active_scientific_siblings() -> None:
    actual = {path.name for path in TASK_ROOT.glob("qa-*") if path.is_dir()}
    assert actual == set(SURFACES)
    assert not (TASK_ROOT / "qa-answer-policy").exists()
    assert not (TASK_ROOT / "qa-null-confidence").exists()
    dropped = (VENDOR / "dropped_tasks" / "README.md").read_text().lower()
    assert "anti-gaming" in dropped
    assert "monotone reparameterization" in dropped


def test_all_settings_are_fullscale_serial_scored_and_verifier_owned() -> None:
    all_solutions = {item[0] for item in SURFACES.values()}
    all_harnesses = {item[1] for item in SURFACES.values()}
    for task_name, (solution, harness, _attribute) in SURFACES.items():
        config = json.loads((TASK_ROOT / task_name / "config.json").read_text())
        labels = NULL_LABELS if task_name == "qa-null-threshold" else ANSWER_LABELS
        assert config["protocol"] == "qa-official-full-v2"
        assert config["integration_requires"] == [
            "d3ad006b",
            "f714d7d5",
            "c6fa7456",
            "830403af",
            "dd0c8df5",
        ]
        expected_status = (
            "squad2_fullscale_measured_static_proof_hardened_20260711"
            if task_name == "qa-null-threshold"
            else "official_f1_baseline_free_representative_runtime_static_validated_20260711"
        )
        assert config["calibration_status"] == expected_status
        provenance = config["calibration_provenance"]
        if task_name == "qa-null-threshold":
            assert provenance["evidence_scope"] == "qa-null-threshold SQuAD2 only"
            assert provenance["run_id"] == "k1h20-roberta9-v7-4gpu"
            assert provenance["allocation_gpus"] == 4
            assert "QA_COMPLETE" in provenance["completion_contract"]
            assert "workload, model, data, and metrics are unchanged" in provenance[
                "completion_contract"
            ]
        else:
            assert provenance["measured_task"] is None
            assert provenance["representative_model_runtime_evidence"]["task"] == "qa-null-threshold"
            assert provenance["representative_model_runtime_evidence"]["mangrove_task_id"] == 96029
            assert "proof bindings are validated statically" in provenance["limitation"]
        assert config["agent_data_prune"] == ["/data/extractive-qa/data"]
        expected_filenames = (
            [f"squad2_validation_{label}.jsonl" for label in labels]
            if task_name == "qa-null-threshold"
            else [
                "mrqa_squad_validation.jsonl",
                "mrqa_newsqa_validation.jsonl",
                "mrqa_hotpotqa_validation.jsonl",
                "mrqa_naturalquestions_validation.jsonl",
            ]
        )
        assert config["verifier_data_deps"] == [
            {
                "name": f"extractive_qa_gold_{label}",
                "host_path": f"{{data_root}}/extractive-qa/data/{filename}",
                "dest": f"data/extractive-qa/data/{filename}",
            }
            for label, filename in zip(labels, expected_filenames, strict=True)
        ]
        assert config["_verifier_serial"] is True
        assert config["seeds"] == [42]
        commands = config["test_cmds"]
        assert [item["label"] for item in commands] == labels
        assert [item["group"] for item in commands] == list(range(1, len(labels) + 1))
        assert all(item["compute"] == 1.0 for item in commands)
        assert all(item["time"] == "2:00:00" for item in commands)
        assert all(item["mem"] == 32 for item in commands)
        assert all("hidden" not in item and "public" not in item for item in commands)
        for command in commands:
            script = (TASK_ROOT / task_name / command["cmd"]).read_text()
            assert (
                'export QA_DATA="${MLSBENCH_VERIFIER_DATA_ROOT:?}/extractive-qa/data"'
                in script
            )
            assert f"python -u {harness} " in script
            assert f"--solution solution/{solution} " in script
            assert "python -u extractive-qa/" not in script
            assert "--solution extractive-qa/solution/" not in script
            assert "pip install" not in script
            assert "curl " not in script
            assert "wget " not in script
            assert "CUDA_VISIBLE_DEVICES=" not in script
        assert config["verifier_only_package_files"] == [
            "extractive-qa/common.py",
            f"extractive-qa/{harness}",
        ]
        pruned = set(config["agent_pruned_package_files"])
        assert f"extractive-qa/solution/{solution}" not in pruned
        assert f"extractive-qa/{harness}" not in pruned
        for other_solution in all_solutions - {solution}:
            assert f"extractive-qa/solution/{other_solution}" in pruned
        for other_harness in all_harnesses - {harness}:
            assert f"extractive-qa/{other_harness}" in pruned


def test_full_official_dataset_manifest(common) -> None:
    expected = {
        "mrqa_squad_validation.jsonl": (
            10_507,
            10_507,
            0,
            "64ab3a4c69574a258c934044a63605b15d98e1608fa9fb5b244868c5d0af89aa",
        ),
        "mrqa_newsqa_validation.jsonl": (
            4_212,
            4_212,
            0,
            "87b31cff3db4cb8276ddc58c94b03ca3ca500a72af95b8b9e2c63c9266ded7ad",
        ),
        "mrqa_hotpotqa_validation.jsonl": (
            5_901,
            5_901,
            0,
            "a335e1778d3c2de3a99b00e8eeaa3fc6e9b611386afadcc54532c2f33d3d95ad",
        ),
        "mrqa_naturalquestions_validation.jsonl": (
            12_836,
            12_836,
            0,
            "705717e225fc972d9a1df01737ab11d59a2c573a6ba9e7018b5ace4c34de6952",
        ),
        "squad2_validation_part0.jsonl": (
            3_958,
            1_988,
            1_970,
            "bdb7f256bf8893edef347623c6698a16320608d5ddf31c774de8e8234598f5b9",
        ),
        "squad2_validation_part1.jsonl": (
            3_958,
            1_956,
            2_002,
            "4159c7c652415873aa565af317a8c0d460164b5f80b185a35b9cbe6dac40f327",
        ),
        "squad2_validation_part2.jsonl": (
            3_957,
            1_984,
            1_973,
            "4b8fff6cb1dd3370416e1cf36cb7d8ba846ef61fd2cb086ccd02ad80a97ce651",
        ),
    }
    observed = {
        filename: (spec["n"], spec["n_ans"], spec["n_noans"], spec["sha256"])
        for filename, spec in common.DATASET_MANIFEST.items()
    }
    assert observed == expected
    assert sum(item[0] for name, item in expected.items() if name.startswith("mrqa_")) == 33_456
    assert sum(item[0] for name, item in expected.items() if name.startswith("squad2_")) == 11_873


def test_preparation_pins_sources_and_has_no_sampling_or_length_filter(common) -> None:
    data_script_root = ROOT / "vendor" / "data_scripts" / "extractive-qa"
    assert {path.name for path in data_script_root.glob("*.py")} == {"prepare_data.py"}
    prepare = _load(
        "extractive_qa_fullscale_prepare",
        data_script_root / "prepare_data.py",
    )
    assert prepare.MRQA_REVISION == "f3178d9888471dfb2b67c93de14f0ddf499a8d9f"
    assert prepare.SQUAD2_REVISION == "3ffb306f725f7d2ce8394bc1873b24868140c412"
    assert prepare.MODEL_REPO == "deepset/roberta-base-squad2"
    assert prepare.MODEL_REVISION == "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
    assert prepare.MODEL_REVISION == common.MODEL_REVISION
    assert prepare.MODEL_FILES == common.MODEL_MANIFEST
    assert common.MODEL_PARAMETER_COUNT == 124_056_578
    expected_manifest_sha = hashlib.sha256(
        json.dumps(
            common.MODEL_MANIFEST, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    assert common.MODEL_MANIFEST_SHA256 == expected_manifest_sha
    source = (ROOT / "vendor" / "data_scripts" / "extractive-qa" / "prepare_data.py").read_text()
    assert "N_DOMAIN" not in source
    assert "MAX_CTX_WORDS" not in source
    assert "permutation(" not in source
    assert "random" not in source.lower()
    assert "index % 3" in source


def test_native_and_reference_surfaces_are_restricted_literals(common, tmp_path: Path) -> None:
    checked = 0
    for task_name, (solution, _harness, attribute) in SURFACES.items():
        source_path = VENDOR / "solution" / solution
        native_value = common.load_surface_value(str(source_path), attribute)
        config = json.loads((TASK_ROOT / task_name / "config.json").read_text())
        assert config["files"] == [
            {
                "filename": f"extractive-qa/solution/{solution}",
                "read": [{"start": -1, "end": -1}],
                "edit": [{"start": 5, "end": 6}],
            }
        ]
        first_value = None
        for name, baseline in config["baselines"].items():
            namespace = runpy.run_path(str(TASK_ROOT / task_name / baseline["edit_ops"]))
            candidate = _apply_ops(source_path.read_text(), namespace["OPS"])
            candidate_path = tmp_path / f"{task_name}__{name}.py"
            candidate_path.write_text(candidate)
            value = common.load_surface_value(str(candidate_path), attribute)
            if first_value is None:
                first_value = value
            checked += 1
        assert first_value == native_value
    assert checked == 25


def test_agent_python_and_metric_forgery_are_never_executed(common, tmp_path: Path, capsys) -> None:
    marker = tmp_path / "executed"
    malicious = tmp_path / "malicious.py"
    malicious.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "print('QA_METRICS protocol=qa-official-full-v2 dataset=x')\n"
        "def build_casing():\n"
        "    return 'preserve'\n"
    )
    with pytest.raises(ValueError, match="QA_SURFACE_ERROR"):
        common.load_surface_value(str(malicious), "build_casing")
    assert not marker.exists()
    assert "QA_METRICS" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("source", "attribute"),
    [
        ("def build_n_best():\n    return 10 + 10\n", "build_n_best"),
        ("def build_n_best():\n    print('x')\n    return 20\n", "build_n_best"),
        ("def build_n_best(x):\n    return 20\n", "build_n_best"),
        ("def build_n_best() -> int:\n    return 20\n", "build_n_best"),
        ("def build_n_best():\n    return True\n", "build_n_best"),
        ("def build_max_seq_len():\n    return 130\n", "build_max_seq_len"),
        ("def build_null_threshold():\n    return float('nan')\n", "build_null_threshold"),
        ("def build_decoder():\n    return {'method': 'argmax'}\n", "build_decoder"),
        ("def build_doc_stride():\n    return 32\n", "build_doc_stride"),
    ],
)
def test_nonliteral_or_invalid_surface_fails_closed(
    common, tmp_path: Path, source: str, attribute: str
) -> None:
    path = tmp_path / f"{attribute}.py"
    path.write_text(source)
    with pytest.raises(ValueError, match="QA_SURFACE_ERROR"):
        common.load_surface_value(str(path), attribute)


def test_dataset_loader_requires_hash_schema_counts_and_unique_ids(
    common, tmp_path: Path, monkeypatch
) -> None:
    rows = [
        {
            "id": "one",
            "question": "Question?",
            "context": "The answer is here.",
            "answers": ["answer"],
            "is_impossible": False,
        },
        {
            "id": "two",
            "question": "Missing?",
            "context": "No answer is provided.",
            "answers": [],
            "is_impossible": True,
        },
    ]
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    path = tmp_path / "fixture.jsonl"
    path.write_bytes(payload)
    monkeypatch.setenv("QA_DATA", str(tmp_path))
    monkeypatch.setitem(
        common.DATASET_MANIFEST,
        "fixture.jsonl",
        {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "n": 2,
            "n_ans": 1,
            "n_noans": 1,
            "kind": "squad2",
        },
    )
    assert common.load_dataset("fixture.jsonl") == rows
    path.write_bytes(payload.replace(b'"id":"two"', b'"id":"one"'))
    with pytest.raises(RuntimeError, match="QA_DATA_ERROR hash mismatch"):
        common.load_dataset("fixture.jsonl")


def test_scoring_requires_exact_prediction_key_set(common) -> None:
    rows = [
        {
            "id": "q1",
            "question": "Q?",
            "context": "An answer.",
            "answers": ["answer"],
            "is_impossible": False,
        }
    ]
    assert common.score_squad({"q1": "answer"}, rows)["f1"] == 100.0
    with pytest.raises(RuntimeError, match="QA_METRIC_ERROR prediction IDs"):
        common.score_squad({}, rows)
    with pytest.raises(RuntimeError, match="QA_METRIC_ERROR prediction IDs"):
        common.score_squad({"q1": "answer", "extra": ""}, rows)
    source = (VENDOR / "common.py").read_text()
    assert "predictions.get(" not in source


def test_independent_argmax_uses_the_shared_answer_length_cap(common) -> None:
    feature = {
        "offset_mapping": [(0, 1), (2, 3), (4, 5)],
        "_context": "a b c",
    }
    start_logits = [10.0, 0.0, 0.0]
    end_logits = [0.0, 0.0, 10.0]
    text, _score = common.decode_argmax_independent(
        feature,
        start_logits,
        end_logits,
        max_answer_length=2,
    )
    assert text == ""
    text, _score = common.decode_argmax_independent(
        feature,
        start_logits,
        end_logits,
        max_answer_length=3,
    )
    assert text == "a b c"


def test_every_parser_requires_ordered_terminal_proof_from_rendered_path(
    tmp_path: Path,
) -> None:
    for task_name in SURFACES:
        source_path = TASK_ROOT / task_name / "parser.py"
        module = _load(
            f"extractive_qa_parser_{task_name.replace('-', '_')}",
            source_path,
        )
        rendered_path = tmp_path / task_name / "tests" / "meta" / "parser.py"
        rendered_path.parent.mkdir(parents=True)
        rendered_path.write_text(source_path.read_text())
        rendered = _load(
            f"rendered_extract_qa_parser_{task_name.replace('-', '_')}",
            rendered_path,
        )
        assert rendered._EXPECTED_TASK == task_name.removeprefix("qa-").replace("-", "_")

        for parser_module in (module, rendered):
            parser = parser_module.Parser()
            for label in parser_module._EXPECTED:
                valid = _proof(parser_module, label)
                result = parser.parse(label, valid)
                assert result.metrics == {
                    f"f1_{label}": 61.25,
                    f"em_{label}": 55.0,
                }
                protocol, metric, complete = valid.splitlines()
                expected = parser_module._EXPECTED[label]
                invalid = (
                    "",
                    "\n".join((protocol, metric)),
                    "\n".join((metric, protocol, complete)),
                    f"{valid}\ntrailing output",
                    f"{protocol}\n{protocol}\n{metric}\n{complete}",
                    f"{protocol}\n{metric}\n{metric}\n{complete}",
                    f"{protocol}\n{metric}\n{complete}\n{complete}",
                    valid.replace(expected["sha256"], "0" * 64, 1),
                    valid.replace(
                        f"task={parser_module._EXPECTED_TASK}", "task=wrong", 1
                    ),
                    valid.replace(
                        f"surface={parser_module._EXPECTED_SURFACE}",
                        "surface=wrong",
                        1,
                    ),
                    valid.replace("f1=61.25", "f1=nan", 1),
                    valid.replace("f1=61.25", "f1=101", 1),
                    valid.replace(
                        f"n_features={expected.get('n_features', expected['n'] + 7)}",
                        "n_features=1",
                        1,
                    ),
                    f"[COMMAND FAILED exit=9]\n{valid}",
                    f"[TIMEOUT]\n{valid}",
                    f"[COMMAND FAILED - slurm FAILED]\n{valid}",
                    f"Segmentation fault\n{valid}",
                    f"Command exited with code 1\n{valid}",
                    f"Traceback (most recent call last): boom\n{valid}",
                    f"QA_INFERENCE_ERROR incomplete logits\n{valid}",
                )
                for raw_output in invalid:
                    assert parser.parse(label, raw_output).metrics == {}
            label = next(iter(parser_module._EXPECTED))
            assert parser.parse("unknown", _proof(parser_module, label)).metrics == {}


def test_runtime_has_no_agent_import_coercion_clipping_or_metric_fallback() -> None:
    source = (VENDOR / "common.py").read_text()
    assert "importlib" not in source
    assert "exec_module" not in source
    assert "predictions.get(" not in source
    assert "answer_confidence" not in source
    assert "min(max(" not in source
    assert "MAX_CTX_WORDS" not in source
    assert "QA_COMPLETE" in source
    for _task_name, (_solution, harness, _attribute) in SURFACES.items():
        harness_source = (VENDOR / harness).read_text()
        assert "common.cli(" in harness_source
        assert "load_surface" not in harness_source
        assert "QA_METRICS" not in harness_source


def test_generator_reproduces_all_ten_tasks_in_a_temporary_root(tmp_path: Path) -> None:
    generator = _load("extractive_qa_generator_temp", VENDOR / "gen_tasks.py")
    generator.TASK_ROOT = tmp_path / "tasks"
    package_files = generator.all_package_files()
    for task_name, task in generator.TASKS.items():
        generator.write_task(task_name, task, package_files)
    generated = {path.name for path in generator.TASK_ROOT.iterdir() if path.is_dir()}
    assert generated == set(SURFACES)
    for task_name in generated:
        for filename in (
            "config.json",
            "parser.py",
            "score_spec.py",
            "leaderboard.csv",
            "task_description.md",
        ):
            assert (generator.TASK_ROOT / task_name / filename).read_text() == (
                TASK_ROOT / task_name / filename
            ).read_text()


def test_score_specs_use_the_same_baseline_free_official_f1_mapping() -> None:
    for task_name in SURFACES:
        source = (TASK_ROOT / task_name / "score_spec.py").read_text()
        config = json.loads((TASK_ROOT / task_name / "config.json").read_text())
        labels = [item["label"] for item in config["test_cmds"]]
        assert source.count(".bounded_power(bound=100.0, floor=const(0.0),") == len(labels)
        assert source.count("ref=const(50.0), ref_score=0.5") == len(labels)
        assert ".sigmoid(" not in source
        assert config["scoring_status"]["integration_requires"] == [
            "f714d7d5",
            "c6fa7456",
        ]
        assert "bl_worst" not in source and "bl_best" not in source
        leaderboard = (TASK_ROOT / task_name / "leaderboard.csv").read_text().splitlines()
        assert len(leaderboard) == (4 if task_name == "qa-null-threshold" else 1)
        for label in labels:
            assert f"f1_{label}" in source
            assert f'f1_{label}' in leaderboard[0]


def test_measured_anchor_provenance_is_generator_source_of_truth() -> None:
    generator = _load("extractive_qa_generator", VENDOR / "gen_tasks.py")
    null_task = TASK_ROOT / "qa-null-threshold"
    leaderboard = null_task.joinpath("leaderboard.csv").read_text().splitlines()
    assert len(leaderboard) == 4
    assert leaderboard[1:] == [
        ",".join(
            (
                generator.CALIBRATION_TIMESTAMP,
                f"baseline:{baseline_name}",
                "true",
                "mean",
                *(str(metrics[label]) for label in ("f1_part0", "f1_part1", "f1_part2")),
            )
        )
        for baseline_name, metrics in generator.CALIBRATION_RESULTS.items()
    ]
    null_config = json.loads((null_task / "config.json").read_text())
    assert null_config["calibration_provenance"] == generator.CALIBRATION_PROVENANCE
    for task_name in set(SURFACES) - {"qa-null-threshold"}:
        config = json.loads((TASK_ROOT / task_name / "config.json").read_text())
        assert config["calibration_provenance"] == generator.MRQA_REPRESENTATIVE_PROVENANCE


def test_direct_official_f1_and_missing_settings_fail_closed() -> None:
    from mlsbench.scoring.evaluate import (
        BaselineAnchors,
        load_expanded_spec,
        score_record_details,
    )

    for task_name in SURFACES:
        task_dir = TASK_ROOT / task_name
        anchors = BaselineAnchors(task_dir)
        spec = load_expanded_spec(task_dir, anchors)
        assert spec is not None
        complete = {term.metric: 50.0 for term in spec.terms.values()}
        score, _settings, valid = score_record_details(spec, complete, anchors)
        assert valid is True
        assert score == pytest.approx(0.5)
        zeroed = dict(complete)
        zeroed[next(iter(zeroed))] = 0.0
        score, _settings, valid = score_record_details(spec, zeroed, anchors)
        assert valid is True
        assert score == 0.0
        missing = dict(complete)
        missing.pop(next(iter(missing)))
        score, _settings, valid = score_record_details(spec, missing, anchors)
        assert valid is False
        assert score == 0.0
        nonfinite = dict(complete)
        nonfinite[next(iter(nonfinite))] = float("nan")
        score, _settings, valid = score_record_details(spec, nonfinite, anchors)
        assert valid is False
        assert score == 0.0


def test_agent_descriptions_do_not_disclose_settings_anchors_or_ordering() -> None:
    forbidden = (
        "public",
        "private",
        "hidden",
        "newsqa",
        "hotpotqa",
        "naturalq",
        "part0",
        "part1",
        "part2",
        ".jsonl",
        "anchor",
        "leaderboard",
        "baseline:",
        "weak",
        "strong",
        "tuned",
        "best option",
        "minutes",
        "hours",
        "h20",
        "h200",
    )
    for task_name in SURFACES:
        text = (TASK_ROOT / task_name / "task_description.md").read_text().lower()
        for token in forbidden:
            assert token not in text, (task_name, token)
        assert "every configured command contributes" in text
        assert "score of exactly zero" in text


def test_package_versions_model_hashes_and_ready_files_are_pinned(common) -> None:
    package = json.loads(
        (ROOT / "vendor" / "pkg_configs" / "extractive-qa" / "config.json").read_text()
    )
    assert package["mangrove_base_image"].endswith(
        "sha256:3bf1e39f4004791522670ee57d4aaa84ce040135bee896ff8dec16124f4a046e"
    )
    install = " ".join(package["install_cmds"])
    for requirement in (
        "transformers==4.49.0",
        "tokenizers==0.21.0",
        "huggingface_hub==0.28.1",
        "safetensors==0.5.2",
        "regex==2024.11.6",
    ):
        assert requirement in install
    assert "--no-deps" in install
    assert "numpy==" not in install
    assert "numpy.__version__ == '2.1.2'" in install
    dependency = package["data_deps"][0]
    assert dependency["name"] == "extractive_qa_full_v2"
    ready = dependency["ready_files"]
    assert len(ready) == len(common.MODEL_MANIFEST) + len(common.DATASET_MANIFEST)
    for filename in common.MODEL_MANIFEST:
        assert any(item.endswith(f"/models/roberta-base-squad2/{filename}") for item in ready)
    for filename in common.DATASET_MANIFEST:
        assert any(item.endswith(f"/data/{filename}") for item in ready)


def test_model_directory_manifest_rejects_unexpected_entries(
    common, tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "pytorch_model.bin").write_bytes(b"unexpected")
    monkeypatch.setenv("QA_MODEL", str(model_dir))
    with pytest.raises(RuntimeError, match="unexpected model directory entries"):
        common.verify_model_files()


def test_width_preserving_case_normalization_keeps_source_offsets(common) -> None:
    samples = ["ASCII Name", "CAFE", "Straße", "İstanbul", "ΜΙΚΤΟ"]
    for text in samples:
        lowered = common._width_preserving_lower(text)
        assert len(lowered) == len(text)
    assert common._width_preserving_lower("ABC") == "abc"
