from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TASKS = sorted((ROOT / "tasks").glob("summ-*"))
TASK_NAMES = tuple(task.name for task in TASKS)
EXPECTED = {
    "xsum": {
        "rows": 11334,
        "data": "5b7854e6ee8b81493e38a5255ba794dec21735b30ba8f77280a394bf7d6dae53",
        "model": "distilbart-xsum-12-6",
        "revision": "5b2e376c845c201ddc34ec0e55fd1ad9890ba5ee",
        "weights": "6e9ebfc94e474225457570ad33c225cf66ca26279c5cd1cbfb67e089a03a791b",
        "params": 305510400,
    },
    "cnndm": {
        "rows": 11490,
        "data": "525e5bf29cb49a1ea2c58939d11435b6d6422f9a59d9c15e43d226c464b33278",
        "model": "distilbart-cnn-12-6",
        "revision": "a4f8f3ea906ed274767e9906dbaede7531d660ff",
        "weights": "3bac65d18c99463302d12ca75c2220ea714f9c81ce235f205fa818efe71df6ea",
        "params": 305510400,
    },
    "samsum": {
        "rows": 819,
        "data": "b9dd8165689b1c252b2db617b089b408ea068ef5bd6819a93c64e8ebde856b86",
        "model": "bart-large-cnn-samsum",
        "revision": "e49b3d60d923f12db22bdd363356f1a4c68532ad",
        "weights": "9f453aa6edef4dba1893723b7313b57b06b60214442d308a8acc3baa9583dd7b",
        "params": 406290432,
    },
}
SURFACES = {
    "summ-beam-repetition": (
        "SUMM_BEAM num_beams=1 no_repeat_ngram_size=0 repetition_penalty=1.0"
    ),
    "summ-beam-width": "SUMM_BEAMWIDTH num_beams=1",
    "summ-decoding-length": (
        "SUMM_LENGTH min_length=1 max_length=20 length_penalty=0.2"
    ),
    "summ-decoding-temperature": "SUMM_TEMPERATURE temperature=2.0",
    "summ-diverse-beam": (
        "SUMM_DIVERSE num_beams=4 num_beam_groups=4 diversity_penalty=1.0"
    ),
    "summ-norepeat-ngram": "SUMM_NOREPEAT no_repeat_ngram_size=0",
    "summ-nucleus-topp": "SUMM_TOPP top_p=1.0",
    "summ-post-truncation": "SUMM_POSTTRUNC keep_sentences=1",
    "summ-sampling-vs-beam": (
        "SUMM_STRATEGY strategy=sample num_beams=None top_p=0.95 "
        "top_k=0 temperature=1.0"
    ),
    "summ-source-policy": "SUMM_SOURCE policy=abstractive",
}
VERIFIER_RUNTIME = {
    "abstractive-summarization/common.py",
    "abstractive-summarization/harness_beam.py",
    "abstractive-summarization/harness_beamwidth.py",
    "abstractive-summarization/harness_diverse.py",
    "abstractive-summarization/harness_length.py",
    "abstractive-summarization/harness_norepeat.py",
    "abstractive-summarization/harness_posttrunc.py",
    "abstractive-summarization/harness_sampling.py",
    "abstractive-summarization/harness_source.py",
    "abstractive-summarization/harness_temperature.py",
    "abstractive-summarization/harness_topp.py",
}


def _parser(task_name: str):
    path = ROOT / "tasks" / task_name / "parser.py"
    spec = importlib.util.spec_from_file_location(f"parser_{task_name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Parser()


def _vendor_module(filename: str):
    vendor = ROOT / "vendor" / "abstractive-summarization"
    common_spec = importlib.util.spec_from_file_location(
        "summ_contract_common", vendor / "common.py"
    )
    common_module = importlib.util.module_from_spec(common_spec)
    assert common_spec.loader is not None
    common_spec.loader.exec_module(common_module)

    prior_common = sys.modules.get("common")
    sys.modules["common"] = common_module
    try:
        spec = importlib.util.spec_from_file_location(
            f"summ_contract_{Path(filename).stem}", vendor / filename
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if prior_common is None:
            del sys.modules["common"]
        else:
            sys.modules["common"] = prior_common
    return module


def _valid_log(task_name: str, *, source_policy: str = "abstractive") -> str:
    surface = SURFACES[task_name]
    include_models = True
    if task_name == "summ-source-policy":
        surface = f"SUMM_SOURCE policy={source_policy}"
        include_models = source_policy == "abstractive"
    lines = [
        surface,
        "SUMM_PROTOCOL version=summ-full-official-test-v1 settings=3 total_docs=23643",
    ]
    for index, (setting, item) in enumerate(EXPECTED.items()):
        lines.append(
            f"SUMM_DATA setting={setting} n_docs={item['rows']} sha256={item['data']}"
        )
        if include_models:
            lines.append(
                f"SUMM_MODEL setting={setting} model={item['model']} "
                f"revision={item['revision']} params={item['params']} dtype=float16 "
                f"weights_sha256={item['weights']}"
            )
        lines.append(
            f"SUMM_METRICS setting={setting} rougeL={0.30 + index * 0.01:.6f} "
            f"rouge1={0.40 + index * 0.01:.6f} "
            f"rouge2={0.20 + index * 0.01:.6f} plen=42.0 n_docs={item['rows']}"
        )
        lines.append(
            f"SUMM_SETTING_DONE setting={setting} generated={item['rows']} "
            f"expected={item['rows']}"
        )
    lines.extend(
        (
            "SUMM_EVAL_DONE settings=3 total_docs=23643",
            "SUMM_DONE settings=3 total_docs=23643 seed=42 elapsed=1234.5",
        )
    )
    return "\n".join(lines)


def _remove_first(log: str, prefix: str) -> str:
    lines = log.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(prefix))
    del lines[index]
    return "\n".join(lines)


def _duplicate_first(log: str, prefix: str) -> str:
    lines = log.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(prefix))
    lines.insert(index + 1, lines[index])
    return "\n".join(lines)


def _swap_first(log: str, left_prefix: str, right_prefix: str) -> str:
    lines = log.splitlines()
    left = next(i for i, line in enumerate(lines) if line.startswith(left_prefix))
    right = next(i for i, line in enumerate(lines) if line.startswith(right_prefix))
    lines[left], lines[right] = lines[right], lines[left]
    return "\n".join(lines)


def _destructive_categories(task_name: str):
    valid = _valid_log(task_name)
    surface = valid.splitlines()[0]
    other_surface = (
        SURFACES["summ-nucleus-topp"]
        if task_name != "summ-nucleus-topp"
        else SURFACES["summ-beam-width"]
    )
    categories = [
        ("wrong_label", [("not-this-task", valid)]),
        ("missing_surface", [(task_name, _remove_first(valid, "SUMM_"))]),
        ("duplicate_surface", [(task_name, surface + "\n" + valid)]),
        ("wrong_sibling_surface", [(task_name, valid.replace(surface, other_surface, 1))]),
        (
            "surface_protocol_order",
            [(task_name, _swap_first(valid, surface, "SUMM_PROTOCOL"))],
        ),
        ("missing_protocol", [(task_name, _remove_first(valid, "SUMM_PROTOCOL"))]),
        ("duplicate_protocol", [(task_name, _duplicate_first(valid, "SUMM_PROTOCOL"))]),
        ("wrong_data_count", [(task_name, valid.replace("n_docs=11334", "n_docs=11333", 1))]),
        ("wrong_data_digest", [(task_name, valid.replace(EXPECTED["xsum"]["data"], "0" * 64, 1))]),
        (
            "inexact_parameter_count",
            [(task_name, valid.replace("params=305510400", "params=305510401", 1))],
        ),
        (
            "wrong_checkpoint_identity",
            [
                (task_name, valid.replace(EXPECTED["xsum"]["revision"], "0" * 40, 1)),
                (task_name, valid.replace(EXPECTED["cnndm"]["weights"], "1" * 64, 1)),
            ],
        ),
        (
            "out_of_order_setting_proof",
            [
                (
                    task_name,
                    _swap_first(valid, "SUMM_DATA setting=xsum", "SUMM_DATA setting=cnndm"),
                )
            ],
        ),
        (
            "missing_setting_metric",
            [(task_name, _remove_first(valid, "SUMM_METRICS setting=xsum"))],
        ),
        (
            "duplicate_metric",
            [(task_name, _duplicate_first(valid, "SUMM_METRICS setting=xsum"))],
        ),
        (
            "invalid_metric_values",
            [
                (task_name, valid.replace("rougeL=0.300000", "rougeL=nan", 1)),
                (task_name, valid.replace("rouge1=0.400000", "rouge1=inf", 1)),
                (task_name, valid.replace("rouge2=0.200000", "rouge2=1.1", 1)),
                (task_name, valid.replace("plen=42.0", "plen=-1", 1)),
            ],
        ),
        (
            "wrong_metric_inventory",
            [
                (
                    task_name,
                    valid.replace(
                        "plen=42.0 n_docs=11334", "plen=42.0 n_docs=11333", 1
                    ),
                )
            ],
        ),
        (
            "invalid_setting_completion",
            [
                (task_name, valid.replace("generated=819", "generated=818", 1)),
                (task_name, _duplicate_first(valid, "SUMM_SETTING_DONE setting=samsum")),
                (task_name, _remove_first(valid, "SUMM_SETTING_DONE setting=samsum")),
            ],
        ),
        (
            "invalid_eval_completion",
            [
                (task_name, _remove_first(valid, "SUMM_EVAL_DONE")),
                (task_name, _duplicate_first(valid, "SUMM_EVAL_DONE")),
                (task_name, valid.replace("SUMM_EVAL_DONE settings=3", "SUMM_EVAL_DONE settings=2", 1)),
            ],
        ),
        (
            "invalid_final_completion",
            [
                (task_name, _remove_first(valid, "SUMM_DONE")),
                (task_name, _duplicate_first(valid, "SUMM_DONE")),
                (task_name, valid.replace("elapsed=1234.5", "elapsed=nan")),
                (task_name, valid.replace("elapsed=1234.5", "elapsed=0")),
            ],
        ),
        (
            "trailing_or_failure_output",
            [
                (task_name, valid + "\ntrailing output"),
                (task_name, valid + "\nTraceback (most recent call last):"),
                (task_name, valid.replace("SUMM_DONE", "SURFACE_ERROR injected\nSUMM_DONE", 1)),
                (task_name, valid.replace("SUMM_DONE", "VERIFICATION_FAILED rc=9\nSUMM_DONE", 1)),
            ],
        ),
    ]
    assert len(categories) == 20
    return categories


def test_all_ten_task_specific_protocols_are_accepted():
    assert len(TASKS) == 10
    for task_name in TASK_NAMES:
        result = _parser(task_name).parse(task_name, _valid_log(task_name))
        assert len(result.metrics) == 9, (task_name, result.feedback)


def test_diverse_harness_and_parser_accept_exactly_the_same_grouping_boundaries():
    harness = _vendor_module("harness_diverse.py")
    parser = _parser("summ-diverse-beam")
    cases = (
        ({"num_beams": 1, "num_beam_groups": 1, "diversity_penalty": 0.0}, True),
        ({"num_beams": 12, "num_beam_groups": 1, "diversity_penalty": 0.0}, True),
        ({"num_beams": 4, "num_beam_groups": 2, "diversity_penalty": 0.0001}, True),
        ({"num_beams": 12, "num_beam_groups": 12, "diversity_penalty": 10.0}, True),
        ({"num_beams": 4, "num_beam_groups": 1, "diversity_penalty": 1.0}, False),
        ({"num_beams": 4, "num_beam_groups": 2, "diversity_penalty": 0.0}, False),
        ({"num_beams": 5, "num_beam_groups": 2, "diversity_penalty": 1.0}, False),
        ({"num_beams": 13, "num_beam_groups": 1, "diversity_penalty": 0.0}, False),
        ({"num_beams": 4, "num_beam_groups": 4, "diversity_penalty": 10.0001}, False),
    )
    for config, expected in cases:
        try:
            harness._validate_diverse_config(config)
            harness_accepted = True
        except (TypeError, ValueError):
            harness_accepted = False
        surface = (
            f"SUMM_DIVERSE num_beams={config['num_beams']} "
            f"num_beam_groups={config['num_beam_groups']} "
            f"diversity_penalty={config['diversity_penalty']}"
        )
        log = _valid_log("summ-diverse-beam").replace(
            SURFACES["summ-diverse-beam"], surface, 1
        )
        parser_accepted = len(
            parser.parse("summ-diverse-beam", log).metrics
        ) == 9
        assert harness_accepted is expected, config
        assert parser_accepted is expected, config


def test_standalone_temperature_harness_and_parser_share_closed_interval():
    harness = _vendor_module("harness_temperature.py")
    parser = _parser("summ-decoding-temperature")
    for value, expected in (
        (0.0, False),
        (0.01, False),
        (0.049999, False),
        (0.05, True),
        (5.0, True),
        (5.0001, False),
    ):
        try:
            harness._validate_temperature(value)
            harness_accepted = True
        except (TypeError, ValueError):
            harness_accepted = False
        surface = f"SUMM_TEMPERATURE temperature={value}"
        log = _valid_log("summ-decoding-temperature").replace(
            SURFACES["summ-decoding-temperature"], surface, 1
        )
        parser_accepted = len(
            parser.parse("summ-decoding-temperature", log).metrics
        ) == 9
        assert harness_accepted is expected, value
        assert parser_accepted is expected, value


def test_non_model_source_policies_are_explicit_and_do_not_claim_model_proofs():
    parser = _parser("summ-source-policy")
    for policy in ("lead3", "copy_document", "first_token", "empty"):
        result = parser.parse(
            "summ-source-policy", _valid_log("summ-source-policy", source_policy=policy)
        )
        assert len(result.metrics) == 9, (policy, result.feedback)
    invalid = _valid_log("summ-source-policy", source_policy="empty")
    xsum = EXPECTED["xsum"]
    injected = (
        f"SUMM_MODEL setting=xsum model={xsum['model']} revision={xsum['revision']} "
        f"params={xsum['params']} dtype=float16 weights_sha256={xsum['weights']}"
    )
    invalid = invalid.replace("SUMM_METRICS setting=xsum", injected + "\nSUMM_METRICS setting=xsum")
    assert parser.parse("summ-source-policy", invalid).metrics == {}


def test_twenty_destructive_categories_reject_for_every_sibling():
    category_names: set[str] = set()
    replay_count = 0
    for task_name in TASK_NAMES:
        parser = _parser(task_name)
        for category, variants in _destructive_categories(task_name):
            category_names.add(category)
            for cmd_label, mutated in variants:
                result = parser.parse(cmd_label, mutated)
                assert result.metrics == {}, (task_name, category, result.feedback)
                replay_count += 1
    assert len(category_names) == 20
    assert replay_count == 340


def test_ten_siblings_share_parser_and_fullscale_config_contract():
    parser_hashes = {
        hashlib.sha256((task / "parser.py").read_bytes()).hexdigest() for task in TASKS
    }
    score_hashes = {
        hashlib.sha256((task / "score_spec.py").read_bytes()).hexdigest() for task in TASKS
    }
    assert len(parser_hashes) == 1
    assert len(score_hashes) == 1
    for task in TASKS:
        config = json.loads((task / "config.json").read_text())
        assert config["rigorous_codebase"] is True
        assert config["seeds"] == [42]
        assert config["test_cmds"] == [
            {
                "cmd": "scripts/run.sh",
                "label": task.name,
                "group": 1,
                "compute": 1,
                "time": "4:00:00",
                "mem": 64,
                "package": "abstractive-summarization",
            }
        ]
        assert config["calibration_protocol"] == "summ-full-official-test-v1"
        assert config["calibration_anchor_counts"] == {
            "xsum": 11334,
            "cnndm": 11490,
            "samsum": 819,
        }
        details = config["calibration_protocol_details"]
        assert details["settings_order"] == ["xsum", "cnndm", "samsum"]
        assert details["total_documents"] == 23643
        assert details["max_input_tokens"] == 512
        assert details["generation_batch_size"] == 16
        assert details["execution"] == "serial_single_gpu"
        assert set(config["verifier_only_package_files"]) == VERIFIER_RUNTIME


def test_package_uses_established_digest_pinned_mangrove_image_key():
    package_config = json.loads(
        (
            ROOT
            / "vendor"
            / "pkg_configs"
            / "abstractive-summarization"
            / "config.json"
        ).read_text()
    )
    assert "pinned_harbor_image" not in package_config
    assert package_config["mangrove_base_image"] == (
        "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
        "mlsbench-harbor-abstractive-summarization@"
        "sha256:06b0678dc84d47be4a304a150f9f171e1e37f73fc0788c1fbb5651c0b406497a"
    )


def test_512_token_protocol_and_exact_checkpoint_counts_are_code_pinned():
    common_path = ROOT / "vendor" / "abstractive-summarization" / "common.py"
    source = common_path.read_text()
    tree = ast.parse(source, filename=str(common_path))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"DEFAULT_MAX_INPUT_TOKENS", "GEN_BATCH_SIZE"}
    }
    assert assignments == {"DEFAULT_MAX_INPUT_TOKENS": 512, "GEN_BATCH_SIZE": 16}
    assert "max_length=int(max_input_tokens)" in source
    assert "max_input_tokens != DEFAULT_MAX_INPUT_TOKENS" in source
    for item in EXPECTED.values():
        assert f'"parameter_count": {item["params"]}' in source
    assert 'parameter_count != expected["parameter_count"]' in source
    assert "torch.cuda.device_count() != 1" in source
    assert "parameter_dtypes != {torch.float16}" in source
    assert 'parameter_device_types != {"cuda"}' in source


def test_scripts_are_offline_strict_and_preserve_runner_gpu_visibility():
    forbidden = ("pip install", "conda install", "apt-get", "curl ", "wget ", "git clone")
    for task in TASKS:
        script = (task / "scripts" / "run.sh").read_text()
        assert "set -euo pipefail" in script
        assert f'TASK_ID="{task.name}"' in script
        assert "VERIFICATION_FAILED" in script
        assert "cd /workspace/abstractive-summarization || exit 111" in script
        assert "CUDA_VISIBLE_DEVICES" not in script
        assert not any(token in script for token in forbidden)


def test_pending_and_measured_leaderboards_have_no_fabricated_rows():
    for task in TASKS:
        rows = (task / "leaderboard.csv").read_text().splitlines()
        if task.name == "summ-beam-width":
            assert len(rows) == 2
            assert rows[1].split(",")[1] == "baseline:greedy"
            config = json.loads((task / "config.json").read_text())
            evidence = config["calibration_provenance"]
            assert evidence["task_id"] == 96438
            assert evidence["container_id"] == 4932334
            assert evidence["return_code"] == 0
            assert evidence["raw_log_sha256"] == (
                "c7d95e993d7e649d1f49f186f21ba97b8ca4407c3021771d76715b3e56dea23f"
            )
        else:
            assert len(rows) == 1


def test_all_score_specs_keep_missing_nonfinite_and_zero_setting_at_exact_zero():
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import load_expanded_spec, score_record_details

    for task in TASKS:
        anchors = BaselineAnchors(task)
        spec = load_expanded_spec(task, anchors)
        assert spec is not None
        valid_record = {f"rougeL_{setting}": 0.30 for setting in EXPECTED}
        score, _settings, valid = score_record_details(spec, valid_record, anchors)
        assert valid and math.isclose(score, 0.5, rel_tol=0.0, abs_tol=1e-12)

        destructive = [
            {},
            {**valid_record, "rougeL_xsum": math.nan},
            {**valid_record, "rougeL_cnndm": math.inf},
            {**valid_record, "rougeL_samsum": -math.inf},
            {**valid_record, "rougeL_xsum": 0.0},
        ]
        for record in destructive:
            failed_score, _settings, _valid = score_record_details(spec, record, anchors)
            assert failed_score == 0.0


def test_no_legacy_slice_assets_public_hidden_labels_or_answer_leaks_remain():
    assert not (ROOT / "vendor" / "abstractive-summarization" / "_summ_data").exists()
    disallowed = (
        "300-doc",
        "300-document",
        "head-slice",
        "public setting",
        "hidden setting",
        "weak baseline",
        "strong baseline",
        "standard strong",
        "reliably loses",
        "matches-or-beats",
    )
    for task in TASKS:
        searchable = "\n".join(
            path.read_text(errors="replace").lower()
            for path in task.rglob("*")
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".sh"}
        )
        for phrase in disallowed:
            assert phrase not in searchable, f"{task.name} contains {phrase!r}"


def test_agent_visible_contracts_publish_numeric_domains_and_metric_definition():
    expected_fragments = {
        "summ-beam-repetition": ("[1, 12]", "[0, 20]", "(0, 10]"),
        "summ-beam-width": ("[1, 12]",),
        "summ-decoding-length": ("[0, 200]", "[1, 200]", "(0, 10]"),
        "summ-decoding-temperature": ("[0.05, 5.0]",),
        "summ-diverse-beam": (
            "[1, 12]",
            "[1, num_beams]",
            "exactly 0 when groups == 1",
            "(0, 10]",
        ),
        "summ-norepeat-ngram": ("[0, 20]",),
        "summ-nucleus-topp": ("[0.05, 1.0]",),
        "summ-post-truncation": ("[0, 10000]",),
        "summ-sampling-vs-beam": ("[1, 12]", "(0, 1]", "[0, 1000]", "(0, 5]"),
    }
    for task in TASKS:
        visible = (
            (task / "task_description.md").read_text()
            + "\n"
            + (task / "edits" / "custom_template.py").read_text()
        )
        for fragment in expected_fragments.get(task.name, ()):
            assert fragment in visible, (task.name, fragment)
        assert "mean per-example ROUGE-L F1" in visible
        assert "corpus ROUGE-L F1" not in visible


def test_every_baseline_edit_materializes_valid_python_on_native_scaffold():
    editable_files: set[str] = set()
    for task in TASKS:
        config = json.loads((task / "config.json").read_text())
        editable = config["files"][0]["edit"][0]
        editable_files.add(config["files"][0]["filename"])
        native_path = task / "edits" / "custom_template.py"
        native_lines = native_path.read_text().splitlines(keepends=True)
        if editable["start"] != -1:
            segment = "".join(native_lines[editable["start"] - 1 : editable["end"]])
            assert "def " in segment and "return " in segment
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
            materialized = "".join(
                native_lines[: operation["start_line"] - 1]
                + [operation["content"].rstrip("\n") + "\n"]
                + native_lines[operation["end_line"] :]
            )
            compile(materialized, str(edit_path), "exec")
    assert len(editable_files) == 10
