from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "machine-translation"
TASKS = ROOT / "tasks"

SURFACES = {
    "mt-batch-maxlen": ("maxlen.py", "build_max_new_tokens"),
    "mt-decoding-beam": ("beam.py", "build_beam_config"),
    "mt-decoding-strategy": ("strategy.py", "build_strategy"),
    "mt-decoding-temperature": ("temperature.py", "build_temperature"),
    "mt-diverse-beam": ("divbeam.py", "build_divbeam_config"),
    "mt-early-stopping": ("earlystop.py", "build_early_stopping"),
    "mt-length-penalty": ("length.py", "build_length_config"),
    "mt-no-repeat-ngram": ("norep.py", "build_norep_config"),
    "mt-postprocess-detok": ("postproc.py", "build_postproc"),
    "mt-repetition-penalty": ("reppen.py", "build_reppen_config"),
    "mt-sampling-vs-beam": ("sampling.py", "build_mode"),
    "mt-tokenization-truncation": ("tok.py", "build_source_max_tokens"),
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


def test_all_siblings_are_serial_full_scale_and_verifier_owned() -> None:
    from mlsbench.scoring.spec import load_score_spec

    assert {path.name for path in TASKS.glob("mt-*") if path.is_dir()} == set(SURFACES)
    package_python = {
        path.relative_to(ROOT / "vendor").as_posix()
        for path in VENDOR.rglob("*.py")
        if path.name != "__init__.py" and "__pycache__" not in path.parts
    }
    for task_name, (filename, _attr) in SURFACES.items():
        config = json.loads((TASKS / task_name / "config.json").read_text())
        commands = config["test_cmds"]
        assert [item["label"] for item in commands] == ["de_en", "fr_en", "ru_en"]
        assert [item["group"] for item in commands] == [1, 2, 3]
        assert all(item["compute"] == 1.0 for item in commands)
        assert all(item["time"] == "1:30:00" for item in commands)
        active = f"machine-translation/solution/{filename}"
        verifier_files = set(config["verifier_only_package_files"])
        assert "machine-translation/common.py" in verifier_files
        assert len(verifier_files) == 2
        assert active not in verifier_files
        pruned_files = set(config["agent_pruned_package_files"])
        assert pruned_files == package_python - verifier_files - {active}
        assert any("/baselines/" in path for path in pruned_files)
        assert any("/solution/" in path for path in pruned_files)
        assert config["agent_data_prune"] == ["/data/machine-translation/data"]
        assert [dep["dest"] for dep in config["verifier_data_deps"]] == [
            "data/machine-translation/data/de_en_test.jsonl",
            "data/machine-translation/data/fr_en_test.jsonl",
            "data/machine-translation/data/ru_en_test.jsonl",
            "data/machine-translation/data/source_manifest.json",
        ]
        assert all(dep["required"] is True for dep in config["verifier_data_deps"])
        for command in commands:
            script = (TASKS / task_name / command["cmd"]).read_text()
            assert "set -euo pipefail" in script
            assert "MLSBENCH_VERIFIER_DATA_ROOT" in script
            assert (
                'export MT_DATA="${MLSBENCH_VERIFIER_DATA_ROOT}/'
                'machine-translation/data"'
            ) in script
            assert "CUDA_VISIBLE_DEVICES" not in script
            assert "MT_SETTING_COMPLETE" in script
        score_spec = load_score_spec(TASKS / task_name)
        assert score_spec is not None
        assert set(score_spec.settings) == {"de_en", "fr_en", "ru_en"}


def test_native_and_declared_baselines_are_static_literals(common, tmp_path: Path) -> None:
    checked = 0
    for task_name, (filename, attr) in SURFACES.items():
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
        "print('MT_METRICS bleu=100 chrf=100 n_pairs=2000 plen=1 elapsed=0')\n"
        "def build_beam_config():\n"
        "    return {'num_beams': 5, 'no_repeat_ngram_size': 0}\n"
    )
    with pytest.raises(ValueError, match="top-level executable"):
        common.load_surface_value(str(malicious), "build_beam_config")
    assert not marker.exists()
    assert "MT_METRICS" not in capsys.readouterr().out


def test_surface_requires_one_literal_return(common, tmp_path: Path) -> None:
    invalid_sources = (
        "def build_beam_config():\n    return dict(num_beams=5)\n",
        "def build_beam_config():\n    print('forged')\n    return {'num_beams': 5}\n",
        "def build_beam_config(x):\n    return {'num_beams': 5}\n",
    )
    for source in invalid_sources:
        path = tmp_path / "invalid.py"
        path.write_text(source)
        with pytest.raises(ValueError, match="unsafe machine-translation configuration"):
            common.load_surface_value(str(path), "build_beam_config")


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


def test_complete_official_split_is_required(common, tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    split = data_root / "de_en_test.jsonl"
    rows = [
        {"src": f"Quelle {index}", "ref": f"reference {index}"}
        for index in range(2000)
    ]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    split.write_text(payload)
    monkeypatch.setenv("MT_DATA", str(data_root))
    monkeypatch.setitem(
        common.EXPECTED_TEST_SHA256,
        "de_en",
        hashlib.sha256(payload.encode()).hexdigest(),
    )
    sources, references = common.load_dataset("de_en")
    assert len(sources) == len(references) == 2000

    truncated = "".join(json.dumps(row) + "\n" for row in rows[:-1])
    split.write_text(truncated)
    monkeypatch.setitem(
        common.EXPECTED_TEST_SHA256,
        "de_en",
        hashlib.sha256(truncated.encode()).hexdigest(),
    )
    with pytest.raises(ValueError, match="expected 2000 rows, got 1999"):
        common.load_dataset("de_en")


def test_parser_requires_one_complete_metric_record() -> None:
    module = _load("ship_mt_parser", TASKS / "mt-decoding-beam" / "parser.py")
    parser = module.Parser()
    metric = "MT_METRICS bleu=20.0 chrf=40.0 n_pairs=2000 plen=12.5 elapsed=1"
    valid = f"{metric}\nMT_SETTING_COMPLETE direction=de_en"
    assert parser.parse("de_en", valid).metrics == {
        "bleu_de_en": 20.0,
        "chrf_de_en": 40.0,
    }
    assert parser.parse("de_en", valid.replace("2000", "400")).metrics == {}
    assert parser.parse("unknown", valid).metrics == {}
    assert parser.parse("de_en", f"{valid}\n{valid}").metrics == {}
    assert parser.parse("de_en", metric).metrics == {}
    assert parser.parse("de_en", valid.replace("de_en", "fr_en")).metrics == {}
    assert parser.parse("de_en", valid.replace("elapsed=1", "elapsed=nan")).metrics == {}


def test_data_preparation_is_pinned_to_complete_official_splits() -> None:
    module = _load(
        "ship_mt_prepare",
        ROOT / "vendor" / "data_scripts" / "machine-translation" / "prepare_data.py",
    )
    assert module.EXPECTED_PAIRS == 2000
    assert len(module.DATASET_REVISION) == 40
    assert all(len(revision) == 40 for revision in module.MODEL_REVISIONS.values())
    assert module.EXPECTED_SHA256 == {
        "de_en_test.jsonl": "2e7a80586d269952371ff5e71f8840e26926416c399051e2371a3b14a1b0b6dc",
        "fr_en_test.jsonl": "09477f8a19e67d3f7c09c320d076f5a32168ab4cde55d8f1d88ffb66c02f68a1",
        "ru_en_test.jsonl": "c072e931f99d1ed04829ec4f63b18c34ebc9ead4b1b19b25fceec60720650eb0",
    }

    package_config = json.loads(
        (ROOT / "vendor" / "pkg_configs" / "machine-translation" / "config.json")
        .read_text()
    )
    assert package_config["mangrove_base_image"] == (
        "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
        "mlsbench-harbor-machine-translation@"
        "sha256:8dfc00ac296d6c5404e482af44ad862fb8a24c60b54029bd340c13a49076efba"
    )
    ready_files = package_config["data_deps"][0]["ready_files"]
    assert any(path.endswith("/source_manifest.json") for path in ready_files)
    assert not any(path.endswith("/official_test_manifest.json") for path in ready_files)


def test_descriptions_and_anchors_do_not_reuse_old_slice_contracts() -> None:
    forbidden = (
        "`de_en`",
        "`fr_en`",
        "`ru_en`",
        "setting",
        "anchored",
        "baseline",
        "400-pair",
        "head-slice",
        "correct answer",
        "measured order",
        "sweet spot",
    )
    for task_name in SURFACES:
        task_dir = TASKS / task_name
        description = (task_dir / "task_description.md").read_text().lower()
        assert not any(token in description for token in forbidden), task_name
        expected_lines = 4 if task_name == "mt-decoding-beam" else 1
        assert len((task_dir / "leaderboard.csv").read_text().splitlines()) == expected_lines


def test_representative_fullscale_anchors_calibrate_without_fallback() -> None:
    from mlsbench.agent.leaderboard import Leaderboard
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import load_expanded_spec, score_record

    task_dir = TASKS / "mt-decoding-beam"
    rows = Leaderboard(task_dir / "leaderboard.csv").all_records()
    anchors = BaselineAnchors(task_dir)
    spec = load_expanded_spec(task_dir, anchors)
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
