from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = sorted((ROOT / "tasks").glob("summ-*"))
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


def _parser():
    path = ROOT / "tasks" / "summ-beam-width" / "parser.py"
    spec = importlib.util.spec_from_file_location("summ_full_parser", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Parser()


def _valid_log(*, include_models: bool = True, source_policy: str | None = None) -> str:
    lines = [
        "SUMM_PROTOCOL version=summ-full-official-test-v1 settings=3 total_docs=23643"
    ]
    if source_policy is not None:
        lines.append(f"SUMM_SOURCE policy={source_policy}")
    for index, (setting, item) in enumerate(EXPECTED.items()):
        lines.append(
            f"SUMM_DATA setting={setting} n_docs={item['rows']} "
            f"sha256={item['data']}"
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
    lines.extend([
        "SUMM_EVAL_DONE settings=3 total_docs=23643",
        "SUMM_DONE settings=3 total_docs=23643 seed=42 elapsed=1234.5",
    ])
    return "\n".join(lines)


def test_complete_model_protocol_is_accepted():
    result = _parser().parse("summ", _valid_log())
    assert set(result.metrics) == {
        f"rouge{metric}_{setting}"
        for setting in EXPECTED
        for metric in ("L", "1", "2")
    }


def test_complete_non_model_source_policy_is_accepted():
    result = _parser().parse(
        "summ", _valid_log(include_models=False, source_policy="lead3")
    )
    assert len(result.metrics) == 9


def test_every_incomplete_or_invalid_proof_rejects_all_metrics():
    valid = _valid_log()
    mutations = [
        valid.replace("SUMM_EVAL_DONE settings=3 total_docs=23643", ""),
        valid.replace("SUMM_DONE settings=3 total_docs=23643 seed=42", "SUMM_DONE settings=2 total_docs=23643 seed=42"),
        valid.replace("n_docs=11334", "n_docs=11333", 1),
        valid.replace(EXPECTED["xsum"]["data"], "0" * 64, 1),
        valid.replace("dtype=float16", "dtype=float32", 1),
        valid.replace(EXPECTED["cnndm"]["weights"], "1" * 64, 1),
        valid.replace("rougeL=0.310000", "rougeL=nan", 1),
        valid.replace("generated=819", "generated=818", 1),
        valid + "\n" + next(
            line for line in valid.splitlines()
            if line.startswith("SUMM_METRICS setting=xsum")
        ),
        valid.replace("elapsed=1234.5", "elapsed=nan"),
    ]
    for mutated in mutations:
        result = _parser().parse("summ", mutated)
        assert result.metrics == {}, result.feedback


def test_model_task_cannot_omit_model_completion_proofs():
    result = _parser().parse("summ", _valid_log(include_models=False))
    assert result.metrics == {}


def test_ten_siblings_share_fail_closed_parser_and_fullscale_config():
    assert len(TASKS) == 10
    parser_hashes = {
        hashlib.sha256((task / "parser.py").read_bytes()).hexdigest()
        for task in TASKS
    }
    assert len(parser_hashes) == 1
    for task in TASKS:
        config = json.loads((task / "config.json").read_text())
        assert config["rigorous_codebase"] is True
        assert "agent_data_prune" not in config
        assert "verifier_data_deps" not in config
        assert config["seeds"] == [42]
        assert config["calibration_protocol"] == "summ-full-official-test-v1"
        assert config["calibration_anchor_counts"] == {
            "xsum": 11334,
            "cnndm": 11490,
            "samsum": 819,
        }
        assert len(config["test_cmds"]) == 1
        assert config["test_cmds"][0]["compute"] == 1
        assert config["test_cmds"][0]["time"] == "4:00:00"


def test_no_legacy_head_slice_assets_or_claims_remain():
    assert not (ROOT / "vendor" / "abstractive-summarization" / "_summ_data").exists()
    for task in TASKS:
        assert not (task / "data").exists()
        searchable = "\n".join(
            path.read_text(errors="replace")
            for path in task.rglob("*")
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".sh"}
        ).lower()
        assert "300-doc" not in searchable
        assert "300-document" not in searchable
        assert "head-slice" not in searchable


def test_agent_scaffolds_do_not_publish_baseline_answers():
    disallowed = (
        "weak baseline",
        "strong baseline",
        "standard strong",
        "reliably loses",
        "matches-or-beats",
        "default here is weak",
    )
    for task in TASKS:
        scaffold = (task / "edits" / "custom_template.py").read_text().lower()
        for phrase in disallowed:
            assert phrase not in scaffold, f"{task.name} leaks {phrase!r}"


def test_every_baseline_edit_materializes_valid_python_on_the_native_scaffold():
    for task in TASKS:
        config = json.loads((task / "config.json").read_text())
        editable = config["files"][0]["edit"][0]
        native_path = task / "edits" / "custom_template.py"
        native_lines = native_path.read_text().splitlines(keepends=True)
        if editable["start"] != -1:
            segment = "".join(
                native_lines[editable["start"] - 1:editable["end"]]
            )
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
            start = operation["start_line"]
            end = operation["end_line"]
            content = operation["content"]
            materialized = "".join(
                native_lines[:start - 1]
                + [content.rstrip("\n") + "\n"]
                + native_lines[end:]
            )
            compile(materialized, str(edit_path), "exec")
