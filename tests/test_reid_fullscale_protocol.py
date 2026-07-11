from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PARSER = _load("reid_fullscale_parser", "tasks/reid-spatial-pooling/parser.py")


def _valid_log(parser=PARSER) -> str:
    lines = [parser._EXPECTED_PROTOCOL]
    total = 0
    epoch_steps = (184,) * 23 + (183,) * 37
    assert sum(epoch_steps) == 11_003
    for epoch, steps in enumerate(epoch_steps):
        total += steps
        lines.append(
            f"REID_EPOCH epoch={epoch} steps={steps} total_steps={total} "
            "loss=1.25 lr=0.0003"
        )
    lines.append(
        f"REID_TRAIN_COMPLETE epochs=60 total_steps={total} "
        f"train_samples={total * 64}"
    )
    for setting, n_query in (("easy", 1122), ("medium", 1123), ("hard", 1123)):
        lines.append(
            f"REID_METRICS setting={setting} map=0.51 rank1=0.72 rank5=0.88 "
            f"num_query={n_query} num_gallery=19732 elapsed=1234.5"
        )
    lines.append(parser._EXPECTED_EVAL_DONE)
    return "\n".join(lines)


def test_valid_fullscale_protocol_parses_all_settings():
    result = PARSER.Parser().parse("market", _valid_log())
    assert set(result.metrics) == {
        "map_easy",
        "rank1_easy",
        "rank5_easy",
        "map_medium",
        "rank1_medium",
        "rank5_medium",
        "map_hard",
        "rank1_hard",
        "rank5_hard",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda log: log.replace("train_images=12936", "train_images=494", 1),
        lambda log: log.replace(
            "REID_EPOCH epoch=59 steps=183 total_steps=11003 loss=1.25 lr=0.0003\n",
            "",
        ),
        lambda log: log.replace(
            "total_steps=11003 train_samples=704192",
            "total_steps=11002 train_samples=704192",
            1,
        ),
        lambda log: log.replace("map=0.51", "map=nan", 1),
        lambda log: log.replace("num_query=1123", "num_query=12", 1),
        lambda log: log.replace("num_gallery=19732", "num_gallery=1428", 1),
        lambda log: log.replace(PARSER._EXPECTED_EVAL_DONE, ""),
        lambda log: log + "\n" + next(
            line for line in log.splitlines() if line.startswith("REID_METRICS setting=hard")
        ),
        lambda log: log + "\nREID_SURFACE_FALLBACK name=pooling reason=bad_shape",
        lambda log: log + "\nTraceback (most recent call last):",
        lambda log: log + "\nverification failed after metrics",
        lambda log: "[BUDGET CHECK FAILED]\n" + log,
        lambda log: "[STATUS: FAILED rc=9]\n" + log,
        lambda log: "[exit code 9]\n" + log,
        lambda log: log + "\narbitrary trailing output",
        lambda log: log.replace(
            "task=reid-spatial-pooling", "task=reid-optimizer", 1
        ),
        lambda log: log.replace(
            PARSER._EXPECTED_EVAL_DONE,
            PARSER._EXPECTED_EVAL_DONE + "\npost-completion output",
        ),
        lambda log: log.replace(
            "REID_METRICS setting=easy", "REID_METRICS malformed\nREID_METRICS setting=easy", 1
        ),
    ],
)
def test_incomplete_or_invalid_protocol_returns_no_metrics(mutate):
    result = PARSER.Parser().parse("market", mutate(_valid_log()))
    assert result.metrics == {}


@pytest.mark.parametrize("steps_per_epoch", [150, 160])
def test_legacy_9000_and_9600_step_logs_fail(steps_per_epoch):
    lines = [PARSER._EXPECTED_PROTOCOL]
    total = 0
    for epoch in range(60):
        total += steps_per_epoch
        lines.append(
            f"REID_EPOCH epoch={epoch} steps={steps_per_epoch} total_steps={total} "
            "loss=1.25 lr=0.0003"
        )
    lines.append(
        f"REID_TRAIN_COMPLETE epochs=60 total_steps={total} "
        f"train_samples={total * 64}"
    )
    for setting, n_query in (("easy", 1122), ("medium", 1123), ("hard", 1123)):
        lines.append(
            f"REID_METRICS setting={setting} map=0.51 rank1=0.72 rank5=0.88 "
            f"num_query={n_query} num_gallery=19732 elapsed=1234.5"
        )
    lines.append(PARSER._EXPECTED_EVAL_DONE)
    assert PARSER.Parser().parse("market", "\n".join(lines)).metrics == {}


def test_fullscale_source_has_fixed_budget_and_inventory():
    common = (ROOT / "vendor/torchreid-reid/common.py").read_text()
    harness = (ROOT / "vendor/torchreid-reid/harness_pool.py").read_text()
    script = (ROOT / "tasks/reid-spatial-pooling/scripts/train.sh").read_text()
    assert "EXPECTED_TRAIN_IMAGES = 12_936" in common
    assert "EXPECTED_QUERY_IMAGES = 3_368" in common
    assert "EXPECTED_GALLERY_IMAGES = 19_732" in common
    assert "EXPECTED_TOTAL_STEPS = 11_003" in common
    assert "EXPECTED_TRAIN_SAMPLES = 704_192" in common
    assert "EXPECTED_EPOCH_STEPS = (184,) * 23 + (183,) * 37" in common
    assert 'name="resnet50"' in common
    assert "--steps" not in harness
    assert 'milestones=[40, 50]' in harness
    assert "--epochs 60" in script
    assert "market1501_subset" not in script

    package = json.loads(
        (ROOT / "vendor/pkg_configs/torchreid-reid/config.json").read_text()
    )
    assert package["mangrove_base_image"] == (
        "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
        "mlsbench-harbor-torchreid-reid@"
        "sha256:fbaaa5d4dcd03ea4e2bf1084b1b8cc78c5ae09723033b5a05d4ec96bd2b8264f"
    )


def test_all_ten_siblings_use_bound_fullscale_budget_and_fail_closed_parsers():
    task_dirs = sorted((ROOT / "tasks").glob("reid-*"))
    assert len(task_dirs) == 10
    for task_dir in task_dirs:
        config = json.loads((task_dir / "config.json").read_text())
        parser = _load(f"parser_{task_dir.name.replace('-', '_')}", str(
            task_dir.relative_to(ROOT) / "parser.py"
        ))
        assert parser._TASK_ID == task_dir.name
        assert f"task={task_dir.name}" in parser._EXPECTED_PROTOCOL
        assert f"task={task_dir.name}" in parser._EXPECTED_EVAL_DONE
        assert parser.Parser().parse("market", _valid_log(parser)).metrics
        wrong_task_log = _valid_log(parser).replace(
            f"task={task_dir.name}", "task=reid-wrong-task"
        )
        assert parser.Parser().parse("market", wrong_task_log).metrics == {}
        assert config["test_cmds"][0]["time"] == "6:00:00"
        assert config["test_cmds"][0]["mem"] == 64
        assert set(config["agent_data_prune"]) >= {
            "/data/torchreid/market1501_full/query",
            "/data/torchreid/market1501_full/gallery",
        }
        assert {dep["dest"] for dep in config["verifier_data_deps"]} == {
            "data/market1501_full/query",
            "data/market1501_full/gallery",
        }
        scripts = "\n".join(path.read_text() for path in (task_dir / "scripts").glob("*.sh"))
        assert "--epochs 60" in scripts
        assert f"--task-id {task_dir.name}" in scripts
        assert "--steps" not in scripts
        assert "market1501_subset" not in scripts
        verifier_files = set(config["verifier_only_package_files"])
        assert "torchreid-reid/common.py" in verifier_files
        assert any(path.startswith("torchreid-reid/harness_") for path in verifier_files)
        if task_dir.name != "reid-spatial-pooling":
            assert (task_dir / "PENDING_FULL_MARKET1501_ANCHORS").is_file()
            assert "pending_full_market1501_anchors" in (
                task_dir / "score_spec.py"
            ).read_text()

    harnesses = sorted((ROOT / "vendor/torchreid-reid").glob("harness_*.py"))
    assert len(harnesses) == 9
    for harness in harnesses:
        source = harness.read_text()
        assert 'add_argument("--epochs", type=int, default=60)' in source
        assert 'add_argument("--task-id", required=True)' in source
        assert 'add_argument("--steps"' not in source
        assert "REID_EVAL_COMPLETE" in source or "finish_fullscale_evaluation" in source


def test_overlapping_research_questions_have_distinct_surfaces():
    dim = json.loads((ROOT / "tasks/reid-embedding-dim/config.json").read_text())
    head = json.loads((ROOT / "tasks/reid-embedding-head/config.json").read_text())
    schedule = json.loads((ROOT / "tasks/reid-lr-schedule/config.json").read_text())
    optimizer = json.loads((ROOT / "tasks/reid-optimizer/config.json").read_text())
    assert dim["files"][0]["filename"].endswith("solution/dimension.py")
    assert head["files"][0]["filename"].endswith("solution/head.py")
    assert schedule["files"][0]["filename"].endswith("solution/schedule.py")
    assert optimizer["files"][0]["filename"].endswith("solution/optimizer.py")
    assert "harness_dim.py" in (
        ROOT / "tasks/reid-embedding-dim/scripts/train.sh"
    ).read_text()
    assert "build_lr_schedule" in (
        ROOT / "vendor/torchreid-reid/harness_optim.py"
    ).read_text()


def test_fullscale_reranking_baselines_do_not_build_dense_gallery_graphs():
    config = json.loads((ROOT / "tasks/reid-reranking/config.json").read_text())
    assert set(config["baselines"]) == {"none", "aqe", "alpha_qe"}

    source = (
        ROOT / "vendor/torchreid-reid/baselines/rerank_alpha_qe.py"
    ).read_text()
    assert "chunk_size=64" in source
    assert "gf @ gf.T" not in source
    assert "g_g_dist" not in source
    assert "zeros_like(original_dist)" not in source

    baseline = _load(
        "reid_fullscale_alpha_qe",
        "vendor/torchreid-reid/baselines/rerank_alpha_qe.py",
    )
    rng = np.random.default_rng(42)
    qf = rng.normal(size=(7, 16)).astype(np.float32)
    gf = rng.normal(size=(23, 16)).astype(np.float32)
    qf /= np.linalg.norm(qf, axis=1, keepdims=True)
    gf /= np.linalg.norm(gf, axis=1, keepdims=True)
    distmat = 1.0 - qf @ gf.T
    result = baseline.build_rerank()(distmat, qf, gf)
    assert result.shape == distmat.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_uncalibrated_fullscale_siblings_score_exact_zero():
    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import score_record_details
    from mlsbench.scoring.spec import load_score_spec

    metrics = {
        f"{metric}_{setting}": 0.75
        for setting in ("easy", "medium", "hard")
        for metric in ("map", "rank1", "rank5")
    }
    for task_dir in sorted((ROOT / "tasks").glob("reid-*")):
        if task_dir.name == "reid-spatial-pooling":
            continue
        spec = load_score_spec(task_dir)
        assert spec is not None
        score, settings, valid = score_record_details(
            spec, metrics, BaselineAnchors(task_dir)
        )
        assert valid is True
        assert score == 0.0
        assert len(settings) == 3
        assert all(setting.score == 0.0 for setting in settings)


def test_spatial_pooling_uses_baseline_free_official_map_scale():
    task_dir = ROOT / "tasks/reid-spatial-pooling"
    leaderboard_lines = (task_dir / "leaderboard.csv").read_text().splitlines()
    assert len(leaderboard_lines) == 1
    spec = (task_dir / "score_spec.py").read_text()
    assert 'for _setting in ("easy", "medium", "hard")' in spec
    assert "bounded_power(" in spec
    assert "bound=const(1.0)" in spec
    assert "floor=const(0.0)" in spec
    assert "gamma=1.0" in spec
    assert "sigmoid(" not in spec
    assert "task(gmean(\"easy\", \"medium\", \"hard\"))" in spec
    provenance = (task_dir / "CALIBRATION_PROVENANCE.md").read_text()
    assert "96623" in provenance and "4950705" in provenance


def test_query_difficulty_partition_covers_each_query_once():
    common = _load("reid_fullscale_common", "vendor/torchreid-reid/common.py")
    gallery = []
    queries = []
    for index in range(3368):
        pid = index
        queries.append((f"q{index:04d}.jpg", pid, 1))
        gallery.append((f"g{index:04d}_a.jpg", pid, 2))
        gallery.extend(
            (f"g{index:04d}_{extra}.jpg", pid, 3)
            for extra in range(index % 5)
        )
    groups = common._split_queries_by_difficulty(queries, gallery)
    assert {name: len(rows) for name, rows in groups.items()} == common.EXPECTED_QUERY_COUNTS
    paths = [path for rows in groups.values() for path, _pid, _cam in rows]
    assert len(paths) == len(set(paths)) == 3368
