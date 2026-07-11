#!/usr/bin/env python3
"""Audit every rendered Caption sibling before dataset publication."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import tomllib
from pathlib import Path


TASKS = (
    "caption-decoding-strategy",
    "caption-visual-mapping",
    "caption-training-objective",
    "caption-feature-prep",
    "caption-mapping-init",
    "caption-train-sampling",
    "caption-optimizer",
    "caption-prompt-format",
    "caption-feature-augment",
    "caption-token-weighting",
)
EXPECTED_IMAGE = (
    "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
    "mlsbench-harbor-image-captioning-full@sha256:"
    "2bc773cf6e838e9defe1b06e20efde3d93a3690b1b09e7cd2ea29217c120e7c7"
)
EXPECTED_SCORER_SHA = "4ea76cd05d62fb999b891ad122b1cced4382955c77e2b88a7c481cdcd31b0b20"
PRIVATE_DATA = (
    "source_manifest.json",
    "train_clip.pt",
    "train_refs.json",
    "eval_clip.pt",
    "eval_refs.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proof(mode: str, *, steps: int = 7500, cider: str = "0.586622") -> str:
    return (
        f"CAPTION_RESULT protocol=flickr8k_official_v1 mode={mode} "
        "train_images=6000 train_pairs=30000 eval_images=1000 "
        f"epochs=10 batch_size=40 steps={steps} seed=42 "
        f"split_sha256={'a' * 64} manifest_sha256={'b' * 64} "
        f"predictions_sha256={'c' * 64} cider={cider} "
        "bleu4=0.218874 status=ok"
    )


def _audit_task(output: Path, source: Path, task_id: str) -> dict[str, object]:
    task = output / task_id
    meta = task / "tests/meta"
    scaffold = task / "environment/_scaffold/image-captioning"
    assert task.is_dir(), task

    dockerfile = (task / "environment/Dockerfile").read_text()
    assert f"FROM {EXPECTED_IMAGE}" in dockerfile
    run_lines = [line.strip() for line in dockerfile.splitlines() if line.startswith("RUN ")]
    assert run_lines == [
        "RUN rm -rf /opt/mlsbench-caption",
        "RUN rm -rf " + " ".join(f"/data/image-captioning/{name}" for name in PRIVATE_DATA),
        "RUN rm -rf /workspace/image-captioning",
    ], run_lines
    assert "COPY _scaffold/ /workspace/" in dockerfile
    assert not re.search(
        r"\b(?:pip|conda|apt(?:-get)?|curl|wget|git\s+clone|unzip|compile|gcc|make)\b",
        "\n".join(line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")),
        re.IGNORECASE,
    )

    task_toml = tomllib.loads((task / "task.toml").read_text())
    assert task_toml["task"]["name"] == task_id
    assert task_toml["agent"]["timeout_sec"] == 1800
    assert task_toml["verifier"]["timeout_sec"] >= 4 * 3600
    environment = task_toml["environment"]
    assert environment["allow_internet"] is False
    assert environment["gpus"] == 1
    assert environment["gpu_types"] == ["H20"]
    assert environment["cpus"] == 8
    assert environment["memory_mb"] == 131072

    instruction = (task / "instruction.md").read_text()
    assert not re.search(r"\b(?:public|hidden)\b", instruction, re.IGNORECASE)
    assert scaffold.is_dir()
    scaffold_files = [path for path in scaffold.rglob("*") if path.is_file()]
    assert scaffold_files
    assert not (scaffold / "harness.py").exists()
    assert not (scaffold / "baselines").exists()
    forbidden_names = {"leaderboard.csv", "score_spec.py", "parser.py"}
    assert not any(path.name in forbidden_names for path in scaffold_files)
    assert not any(path.name in PRIVATE_DATA for path in scaffold_files)

    config = json.loads((meta / "config.json").read_text())
    assert config["agent_image_prune"] == ["/opt/mlsbench-caption"]
    assert config["agent_data_prune"] == [
        f"/data/image-captioning/{name}" for name in PRIVATE_DATA
    ]
    assert config["seeds"] == [42]
    assert len(config["test_cmds"]) == 1
    assert config["test_cmds"][0]["label"] == "flickr"
    assert config["test_cmds"][0]["package"] == "image-captioning"
    assert config["verifier_only_package_files"] == ["image-captioning/harness.py"]
    for name in PRIVATE_DATA:
        private_path = meta / "data/image-captioning" / name
        assert private_path.is_file() and private_path.stat().st_size > 0
    assert (meta / "verifier_package_files/image-captioning/harness.py").is_file()

    source_task = source / "tasks" / task_id
    for name in ("parser.py", "score_spec.py", "leaderboard.csv"):
        assert (meta / name).read_bytes() == (source_task / name).read_bytes()
    scorer = task / "tests/score_task.py"
    assert _sha256(scorer) == EXPECTED_SCORER_SHA

    parser = _load_module(f"caption_parser_{task_id.replace('-', '_')}", meta / "parser.py")
    valid = _proof(parser.EXPECTED_MODE)
    metrics = parser.Parser().parse("flickr", valid).metrics
    assert metrics == {"cider_flickr": 0.586622, "bleu4_flickr": 0.218874}
    assert parser.Parser().parse("wrong-label", valid).metrics == {}
    invalid_logs = (
        "Traceback (most recent call last)\n" + valid,
        valid + "\n" + valid,
        valid + "\ntrailing output",
        _proof("wrong"),
        _proof(parser.EXPECTED_MODE, steps=7499),
        _proof(parser.EXPECTED_MODE, cider="nan"),
    )
    assert all(parser.Parser().parse("flickr", raw).metrics == {} for raw in invalid_logs)
    assert all(
        parser.Parser().parse("flickr", marker + "\n" + valid).metrics == {}
        for marker in parser.FAILURE_MARKERS
    )

    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import score_record_details
    from mlsbench.scoring.spec import load_score_spec

    score_spec = load_score_spec(meta)
    assert score_spec is not None and set(score_spec.settings) == {"flickr"}
    anchors = BaselineAnchors(meta)
    valid_scores = []
    for record in (
        {"cider_flickr": 0.245972, "bleu4_flickr": 0.076101},
        {"cider_flickr": 0.586622, "bleu4_flickr": 0.218874},
    ):
        score, settings, is_valid = score_record_details(score_spec, record, anchors)
        assert is_valid and len(settings) == 1 and math.isfinite(score)
        valid_scores.append(score)
    assert valid_scores[0] == 0.0
    assert math.isclose(valid_scores[1], 0.5, rel_tol=0.0, abs_tol=1e-12)
    for record in (
        {},
        {"cider_flickr": 0.586622},
        {"cider_flickr": float("nan"), "bleu4_flickr": 0.218874},
        {"cider_flickr": 0.586622, "bleu4_flickr": float("inf")},
    ):
        score, _, is_valid = score_record_details(score_spec, record, anchors)
        assert not is_valid and score == 0.0

    return {
        "task": task_id,
        "fixed": True,
        "audited": True,
        "scaffold_files": len(scaffold_files),
        "verifier_timeout_sec": task_toml["verifier"]["timeout_sec"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = [_audit_task(args.output, args.source, task_id) for task_id in TASKS]
    report = {"total": len(TASKS), "fixed": len(rows), "audited": len(rows), "tasks": rows}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
