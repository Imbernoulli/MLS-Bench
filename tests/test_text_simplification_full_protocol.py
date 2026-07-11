from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import runpy

import pytest

from mlsbench.scoring.anchors import BaselineAnchors
from mlsbench.scoring.evaluate import load_expanded_spec, score_record_details


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/text-simplification"
TASKS = ROOT / "tasks"
PROTOCOL = "gem-full-test-v2"
COUNTS = {"asset": 359, "turk": 359, "wiki": 720}
PINNED_IMAGE = (
    "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
    "mlsbench-harbor-text-simplification@"
    "sha256:68bb3e5f9af29c5b260011ea3974a00c42e156173614d5b2157b4ffa66adb338"
)

SURFACES = {
    "simp-beam-width": ("beamwidth.py", "build_num_beams", "beamwidth"),
    "simp-decoding-beam": ("beam.py", "build_beam_config", "beam"),
    "simp-decoding-strategy": ("strategy.py", "build_strategy", "strategy"),
    "simp-decoding-temperature": (
        "temperature.py", "build_temperature", "temperature"
    ),
    "simp-input-truncation": (
        "truncation.py", "build_max_input_tokens", "truncation"
    ),
    "simp-length-control": ("length.py", "build_length_config", "length"),
    "simp-minlen-floor": ("minlen.py", "build_min_length", "minlen"),
    "simp-model-capacity": ("capacity.py", "build_model_choice", "capacity"),
    "simp-nucleus-sampling": ("nucleus.py", "build_top_p", "nucleus"),
    "simp-source-policy": ("policy.py", "build_policy", "policy"),
}

HARNESS = {
    "simp-beam-width": "harness_beamwidth.py",
    "simp-decoding-beam": "harness_beam.py",
    "simp-decoding-strategy": "harness_strategy.py",
    "simp-decoding-temperature": "harness_temperature.py",
    "simp-input-truncation": "harness_truncation.py",
    "simp-length-control": "harness_length.py",
    "simp-minlen-floor": "harness_minlen.py",
    "simp-model-capacity": "harness_capacity.py",
    "simp-nucleus-sampling": "harness_nucleus.py",
    "simp-source-policy": "harness_policy.py",
}

EXPECTED_METRICS = {
    "sari_asset", "bleu_asset", "sari_turk",
    "bleu_turk", "sari_wiki", "bleu_wiki",
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strict_log(task_name: str) -> tuple[object, list[str]]:
    common = _load(VENDOR / "common.py", f"test_simp_common_{task_name}")
    parser_module = _load(
        TASKS / task_name / "parser.py", f"test_simp_parser_{task_name}"
    )
    surface = SURFACES[task_name][2]
    metric_lines = [
        (
            f"SIMP_METRICS protocol={PROTOCOL} task={task_name} "
            f"surface={surface} setting=asset sari=40.0 bleu=20.0 "
            "n_sents=359 plen=10.0 lenratio=0.8"
        ),
        (
            f"SIMP_METRICS protocol={PROTOCOL} task={task_name} "
            f"surface={surface} setting=turk sari=41.0 bleu=21.0 "
            "n_sents=359 plen=11.0 lenratio=0.9"
        ),
        (
            f"SIMP_METRICS protocol={PROTOCOL} task={task_name} "
            f"surface={surface} setting=wiki sari=42.0 bleu=22.0 "
            "n_sents=720 plen=12.0 lenratio=1.0"
        ),
    ]
    metrics_sha = hashlib.sha256(
        ("\n".join(metric_lines) + "\n").encode("utf-8")
    ).hexdigest()
    done = (
        f"SIMP_DONE protocol={PROTOCOL} task={task_name} surface={surface} "
        "settings=asset:359,turk:359,wiki:720 seed=42 "
        f"inventory_sha256={common.DATA_INVENTORY_SHA256} model=base_turk "
        f"model_sha256={common.MODEL_SPEC_SHA256['base_turk']} "
        f"metrics_sha256={metrics_sha} elapsed=1.25 status=ok"
    )
    return parser_module, [*metric_lines, done]


def _assert_rejected(parser: object, rows: list[str], label: str = "simplify") -> None:
    result = parser.parse(label, "\n".join(rows))
    assert result.metrics == {}, result.feedback


def test_complete_official_test_inventory_and_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = _load(VENDOR / "common.py", "test_simp_full_common")
    monkeypatch.setenv("TASK_DIR", str(TASKS / "simp-source-policy"))

    assert common.PROTOCOL == PROTOCOL
    assert common.SETTING_COUNTS == COUNTS
    assert common.TASK_SURFACES == {
        task: surface for task, (_, _, surface) in SURFACES.items()
    }
    for setting, expected in COUNTS.items():
        sources, references = common.load_dataset(setting)
        assert len(sources) == expected
        assert len(references) == expected
        assert all(source.strip() for source in sources)
        assert all(refs and all(ref.strip() for ref in refs) for refs in references)
        with pytest.raises(ValueError, match="requires all"):
            common.load_dataset(setting, 300)


def test_all_three_checkpoint_manifests_are_revision_and_digest_pinned() -> None:
    common = _load(VENDOR / "common.py", "test_simp_model_manifests")
    assert set(common.MODEL_SPECS) == {"small_turk", "small_wikiauto", "base_turk"}
    assert set(common.MODEL_SPEC_SHA256) == set(common.MODEL_SPECS)
    for choice, model in common.MODEL_SPECS.items():
        assert len(model["revision"]) == 40
        assert model["architecture"]["model_type"] == "t5"
        assert model["files"]
        assert len(common.MODEL_SPEC_SHA256[choice]) == 64
        for descriptor in model["files"].values():
            assert descriptor["bytes"] > 0
            assert len(descriptor["sha256"]) == 64

    package = json.loads(
        (ROOT / "vendor/pkg_configs/text-simplification/config.json").read_text()
    )
    assert package["mangrove_base_image"] == PINNED_IMAGE
    ready = package["data_deps"][0]["ready_files"]
    assert len(ready) == 9
    assert all("models/" in path for path in ready)
    assert package["agent_pruned_files"] == ["anchors", "baselines"]


def test_agent_python_is_never_executed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = _load(VENDOR / "common.py", "test_simp_safe_surface")
    marker = tmp_path / "must_not_exist"
    solution = tmp_path / "solution.py"
    solution.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "def build_policy():\n"
        "    print('SIMP_DONE elapsed=1')\n"
        "    return 'beam'\n"
    )

    with pytest.raises(ValueError):
        common.load_surface(str(solution), "build_policy")
    assert not marker.exists()
    assert "SIMP_DONE" not in capsys.readouterr().out


def test_all_native_and_declared_baseline_surfaces_are_finite_literals(
    tmp_path: Path,
) -> None:
    common = _load(VENDOR / "common.py", "test_simp_all_surfaces")
    checked = 0
    for task_name, (filename, symbol, _) in SURFACES.items():
        source = VENDOR / "solution" / filename
        common.load_surface(str(source), symbol)()
        task = TASKS / task_name
        config = json.loads((task / "config.json").read_text())
        source_rel = config["files"][0]["filename"]
        assert Path(source_rel).name == filename
        for baseline_name, baseline in config["baselines"].items():
            namespace = runpy.run_path(str(task / baseline["edit_ops"]))
            lines = source.read_text().splitlines()
            for operation in sorted(
                namespace["OPS"], key=lambda item: item["start_line"], reverse=True
            ):
                assert operation["op"] == "replace"
                assert operation["file"] == source_rel
                start = int(operation["start_line"])
                end = int(operation["end_line"])
                lines[start - 1:end] = str(operation["content"]).splitlines()
            candidate = tmp_path / f"{task_name}__{baseline_name}.py"
            candidate.write_text("\n".join(lines) + "\n")
            common.load_surface(str(candidate), symbol)()
            checked += 1
    assert checked == 30


@pytest.mark.parametrize("task_name", tuple(SURFACES))
def test_every_parser_accepts_only_the_exact_v2_terminal_proof(task_name: str) -> None:
    parser_module, rows = _strict_log(task_name)
    parser = parser_module.Parser()
    valid = parser.parse("simplify", "\n".join(rows))
    assert set(valid.metrics) == EXPECTED_METRICS

    other_task = "simp-source-policy" if task_name != "simp-source-policy" else "simp-beam-width"
    surface = SURFACES[task_name][2]
    mutations = [
        [line.replace(f"task={task_name}", f"task={other_task}") for line in rows],
        [line.replace(f"surface={surface}", "surface=wrong") for line in rows],
        [line.replace(f"protocol={PROTOCOL}", "protocol=gem-full-test-v1") for line in rows],
        rows[:-1] + [rows[-1].replace("seed=42", "seed=43")],
        [rows[0].replace("n_sents=359", "n_sents=300"), *rows[1:]],
        [rows[1], rows[0], rows[2], rows[3]],
        rows[:-1] + [rows[-1].replace("inventory_sha256=", "inventory_sha256=" + "0")],
        rows[:-1] + [rows[-1].replace("model=base_turk", "model=unknown")],
        rows[:-1] + [rows[-1].replace("model_sha256=", "model_sha256=" + "0")],
        rows[:-1] + [rows[-1].replace("metrics_sha256=", "metrics_sha256=" + "0")],
        [rows[0], rows[2], rows[3]],
        [rows[0], *rows],
        [*rows, rows[-1]],
        rows[:-1] + [rows[-1].replace("status=ok", "status=failed")],
        [*rows, "ordinary trailing output"],
        rows[:-1] + ["SIMP_METRICS malformed", rows[-1]],
        rows[:-1] + ["SIMP_DONE malformed", rows[-1]],
        [rows[0].replace("sari=40.0", "sari=nan"), *rows[1:]],
        [rows[0].replace("bleu=20.0", "bleu=inf"), *rows[1:]],
        rows[:-1] + [rows[-1].replace("elapsed=1.25", "elapsed=0")],
        rows[:-1] + [rows[-1].replace("elapsed=1.25", "elapsed=-1")],
        rows[:-1] + [rows[-1].replace("elapsed=1.25", "elapsed=inf")],
    ]
    for mutation in mutations:
        _assert_rejected(parser, mutation)
    _assert_rejected(parser, rows, label="wrong-label")


@pytest.mark.parametrize("task_name", tuple(SURFACES))
def test_every_parser_rejects_every_declared_failure_marker(task_name: str) -> None:
    parser_module, rows = _strict_log(task_name)
    parser = parser_module.Parser()
    for marker in parser_module.FAILURE_MARKERS:
        _assert_rejected(parser, rows[:-1] + [marker, rows[-1]])


def test_all_siblings_are_pending_header_only_and_exact_zero() -> None:
    for task_name in SURFACES:
        task_dir = TASKS / task_name
        config = json.loads((task_dir / "config.json").read_text())
        assert config["rigorous_codebase"] is True
        assert config["calibration_status"] == (
            "pending_fresh_strict_full_official_split_anchors"
        )
        assert config["calibration_protocol"] == PROTOCOL
        assert config["calibration_anchor_seed"] == 42
        assert config["calibration_anchor_counts"] == COUNTS
        assert config["calibration_pending_marker"] == "PENDING_FULL_OFFICIAL_ANCHORS"
        assert "calibration_anchor_evidence_sha256" not in config
        assert "calibration_anchor_evidence_path" not in config
        assert (task_dir / config["calibration_pending_marker"]).is_file()
        assert {entry["time"] for entry in config["test_cmds"]} == {"1:00:00"}
        assert {entry["compute"] for entry in config["test_cmds"]} == {1}
        assert {entry["label"] for entry in config["test_cmds"]} == {"simplify"}
        assert "hidden" not in json.dumps(config).lower()
        assert "public" not in json.dumps(config).lower()

        leaderboard = (task_dir / "leaderboard.csv").read_text().splitlines()
        assert leaderboard == [
            "timestamp,model,is_final,seed,sari_asset,sari_turk,sari_wiki"
        ]
        anchors = BaselineAnchors(task_dir)
        assert anchors.baseline_names() == []
        spec = load_expanded_spec(task_dir, anchors)
        assert spec is not None
        finite = {"sari_asset": 50.0, "sari_turk": 50.0, "sari_wiki": 50.0}
        score, settings, valid = score_record_details(spec, finite, anchors)
        assert valid
        assert score == 0.0
        assert all(setting.score == 0.0 for setting in settings)
        for invalid in (
            {},
            {"sari_asset": math.nan, "sari_turk": 50.0, "sari_wiki": 50.0},
            {"sari_asset": math.inf, "sari_turk": math.inf, "sari_wiki": math.inf},
        ):
            assert score_record_details(spec, invalid, anchors)[0] == 0.0


def test_runtime_scripts_and_harnesses_are_task_bound_and_fail_closed() -> None:
    for task_name, (_, _, surface) in SURFACES.items():
        task_dir = TASKS / task_name
        script = (task_dir / "scripts/simplify.sh").read_text()
        assert "set -euo pipefail" in script
        assert "VERIFICATION_FAILED text-simplification" in script
        assert "CUDA_VISIBLE_DEVICES" not in script
        assert not any(
            token in script
            for token in ("pip install", "conda install", "apt-get", "curl ", "wget ")
        )

        harness = (VENDOR / HARNESS[task_name]).read_text()
        assert f'TASK_ID = "{task_name}"' in harness
        assert f'SURFACE = "{surface}"' in harness
        assert "time.perf_counter()" in harness
        assert "common.emit_metrics(" in harness
        assert "common.emit_done(" in harness


def test_source_visibility_contract_is_consistent_for_all_siblings() -> None:
    for task_name in SURFACES:
        task_dir = TASKS / task_name
        description = (task_dir / "task_description.md").read_text()
        assert "complete source-only ASSET (359)" in description
        assert "Human reference simplifications are never present" in description
        assert "mounted only for verifier scoring" in description
        assert "inference targets, not scoring labels" in description

        config = json.loads((task_dir / "config.json").read_text())
        verifier_only = set(config["verifier_only_package_files"])
        assert "text-simplification/common.py" not in verifier_only
        assert "text-simplification/sari.py" not in verifier_only
        assert f"text-simplification/{HARNESS[task_name]}" not in verifier_only
        assert all("_simp_data" not in path for path in verifier_only)
