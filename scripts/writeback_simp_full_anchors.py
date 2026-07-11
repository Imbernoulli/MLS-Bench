#!/usr/bin/env python3
"""Validate fresh v2 text-simplification anchors before positive writeback.

This command is intentionally fail closed. It accepts only one-GPU worker runs
whose raw logs pass the shipped task parser and whose task, protocol, inventory,
model manifest, source tree, image, worker metadata, return codes, and elapsed
time are all bound. Positive leaderboards are never reconstructed from legacy
logs. Run this command on an mlaunch worker, not on a login node.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
PROTOCOL = "gem-full-test-v2"
PINNED_IMAGE = (
    "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
    "mlsbench-harbor-text-simplification@"
    "sha256:68bb3e5f9af29c5b260011ea3974a00c42e156173614d5b2157b4ffa66adb338"
)
EXPECTED_COUNTS = {"asset": 359, "turk": 359, "wiki": 720}
EXPECTED_METRICS = {
    "sari_asset", "bleu_asset", "sari_turk",
    "bleu_turk", "sari_wiki", "bleu_wiki",
}
SAFE_ID = re.compile(r"[A-Za-z0-9._-]+")
SHA_LINE = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_./-]+)")
FORBIDDEN_RUNTIME_TEXT = (
    "pip install",
    "conda install",
    "apt-get",
    "apt install",
    "git clone",
    "snapshot_download",
    "huggingface.co/",
    "curl ",
    "wget ",
)
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


@dataclass(frozen=True)
class TaskCalibration:
    weak: str
    strong: str


CALIBRATIONS = {
    "simp-beam-width": TaskCalibration("narrow", "wide"),
    "simp-decoding-beam": TaskCalibration("greedy", "beam_norep"),
    "simp-decoding-strategy": TaskCalibration("sample", "beam"),
    "simp-decoding-temperature": TaskCalibration("hot", "cold"),
    "simp-input-truncation": TaskCalibration("short", "mid"),
    "simp-length-control": TaskCalibration("long", "tuned"),
    "simp-minlen-floor": TaskCalibration("floor60", "floor0"),
    "simp-model-capacity": TaskCalibration("small_wikiauto", "base_turk"),
    "simp-nucleus-sampling": TaskCalibration("wide", "tight"),
    "simp-source-policy": TaskCalibration("empty", "beam"),
}


@dataclass(frozen=True)
class TaskAnchor:
    task: str
    task_dir: Path
    anchor_dir: Path
    config: dict[str, Any]
    baselines: tuple[str, ...]
    metrics: dict[str, dict[str, dict[str, float]]]
    evidence: dict[str, Any]
    evidence_files: dict[str, bytes]
    timestamp: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_text(path: Path, expected: str | None = None) -> str:
    if not path.is_file():
        raise ValueError(f"missing required evidence file: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"empty required evidence file: {path}")
    if expected is not None and value != expected:
        raise ValueError(f"{path}: expected {expected!r}, got {value!r}")
    return value


def _require_zero(path: Path) -> None:
    _require_text(path, "0")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing required JSON evidence: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _parse_time(path: Path) -> tuple[str, datetime]:
    raw = _require_text(path)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{path}: invalid ISO timestamp {raw!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{path}: timestamp must include a timezone")
    return raw, parsed


def _protocol_paths(task: str, config: dict[str, Any]) -> set[str]:
    files = config.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError(f"{task}: expected exactly one editable source")
    source_rel = files[0].get("filename")
    if not isinstance(source_rel, str) or not source_rel.startswith(
        "text-simplification/solution/"
    ):
        raise ValueError(f"{task}: malformed editable source path")

    baselines = config.get("baselines")
    if not isinstance(baselines, dict) or not baselines:
        raise ValueError(f"{task}: no configured baselines")
    paths = {
        "vendor/text-simplification/common.py",
        "vendor/text-simplification/sari.py",
        f"vendor/text-simplification/{HARNESS[task]}",
        f"vendor/{source_rel}",
        "scripts/materialize_simp_anchor_surface.py",
        f"tasks/{task}/parser.py",
        f"tasks/{task}/config.json",
    }
    for setting in EXPECTED_COUNTS:
        paths.add(f"tasks/{task}/data/simp_{setting}_refs.jsonl")
        paths.add(
            f"vendor/text-simplification/_simp_data/simp_{setting}_src.jsonl"
        )
    for baseline, descriptor in baselines.items():
        if not isinstance(descriptor, dict) or set(descriptor) != {"edit_ops"}:
            raise ValueError(f"{task}:{baseline}: malformed baseline descriptor")
        edit_ops = descriptor["edit_ops"]
        if (
            not isinstance(edit_ops, str)
            or edit_ops.startswith("/")
            or ".." in Path(edit_ops).parts
        ):
            raise ValueError(f"{task}:{baseline}: unsafe edit path")
        paths.add(f"tasks/{task}/{edit_ops}")
    return paths


def _read_protocol_hashes(
    path: Path,
    expected_paths: set[str],
    *,
    config_status: str,
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SHA_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"{path}: malformed sha256sum record {line!r}")
        digest, relative = match.groups()
        if relative in observed:
            raise ValueError(f"{path}: duplicate protocol path {relative}")
        observed[relative] = digest
    if set(observed) != expected_paths:
        raise ValueError(
            f"{path}: protocol path set mismatch; "
            f"missing={sorted(expected_paths - set(observed))} "
            f"extra={sorted(set(observed) - expected_paths)}"
        )

    measured = config_status == "measured_fresh_strict_full_official_split_anchors"
    for relative, expected_sha in observed.items():
        source = ROOT / relative
        if not source.is_file():
            raise ValueError(f"protocol source is missing: {source}")
        # The anchor-time config is intentionally replaced by the measured,
        # evidence-bound config after first writeback. Every other source remains
        # byte-identical and is rechecked in both write and check modes.
        if measured and relative.endswith("/config.json"):
            continue
        actual_sha = _sha256_file(source)
        if actual_sha != expected_sha:
            raise ValueError(
                f"protocol source digest mismatch for {relative}: "
                f"{actual_sha} != {expected_sha}"
            )
    return observed


def _parse_surface_log(
    parser_module: ModuleType,
    path: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    lowered = raw.lower()
    forbidden = [token for token in FORBIDDEN_RUNTIME_TEXT if token in lowered]
    if forbidden:
        raise ValueError(f"{path}: runtime install/download text found: {forbidden}")

    parsed = parser_module.Parser().parse("simplify", raw)
    if set(parsed.metrics) != EXPECTED_METRICS:
        raise ValueError(f"{path}: strict parser rejected log: {parsed.feedback}")

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    metric_lines = [line for line in lines if line.startswith("SIMP_METRICS")]
    done_line = lines[-1]
    done_match = parser_module.DONE.fullmatch(done_line)
    if done_match is None:
        raise ValueError(f"{path}: missing strict final completion")
    seed_raw, inventory_sha, model_name, model_sha, metrics_sha, elapsed_raw = (
        done_match.groups()
    )
    elapsed = float(elapsed_raw)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError(f"{path}: invalid completion elapsed {elapsed_raw!r}")

    records: dict[str, dict[str, float]] = {}
    for line in metric_lines:
        match = parser_module.METRIC.fullmatch(line)
        if match is None:
            raise ValueError(f"{path}: malformed metric after parser acceptance")
        setting, sari_raw, bleu_raw, count_raw, plen_raw, ratio_raw = match.groups()
        records[setting] = {
            "sari": float(sari_raw),
            "bleu": float(bleu_raw),
            "n_sents": int(count_raw),
            "plen": float(plen_raw),
            "lenratio": float(ratio_raw),
        }
    if tuple(records) != tuple(EXPECTED_COUNTS):
        raise ValueError(f"{path}: noncanonical metric inventory")

    completion = {
        "seed": int(seed_raw),
        "inventory_sha256": inventory_sha,
        "model": model_name,
        "model_sha256": model_sha,
        "metrics_sha256": metrics_sha,
        "elapsed_seconds": elapsed,
    }
    return records, completion


def _evidence_bytes(anchor_dir: Path, baselines: tuple[str, ...]) -> dict[str, bytes]:
    names = {
        "STARTED",
        "SUCCESS",
        "status",
        "task.rc",
        "worker.name",
        "launch-request.json",
        "runtime.json",
        "protocol.sha256",
        "baselines.txt",
    }
    for baseline in baselines:
        names.update(
            {
                f"{baseline}.worker.log",
                f"{baseline}.rc",
                f"{baseline}.parser.rc",
                f"{baseline}.worker.log.sha256",
            }
        )
    return {name: (anchor_dir / name).read_bytes() for name in sorted(names)}


def _load_task_anchor(anchor_root: Path, task: str, common: ModuleType) -> TaskAnchor:
    task_dir = TASKS / task
    anchor_dir = anchor_root / task
    if not anchor_dir.is_dir():
        raise ValueError(f"missing task anchor directory: {anchor_dir}")
    config = json.loads((task_dir / "config.json").read_text(encoding="utf-8"))
    baselines = tuple(config["baselines"])
    if not baselines:
        raise ValueError(f"{task}: empty baseline set")

    _require_text(anchor_dir / "status", "success")
    _require_zero(anchor_dir / "task.rc")
    started_raw, started = _parse_time(anchor_dir / "STARTED")
    success_raw, success = _parse_time(anchor_dir / "SUCCESS")
    if success <= started:
        raise ValueError(f"{task}: SUCCESS must be later than STARTED")

    worker = _require_text(anchor_dir / "worker.name")
    if SAFE_ID.fullmatch(worker) is None:
        raise ValueError(f"{task}: malformed worker name {worker!r}")
    launch = _load_json(anchor_dir / "launch-request.json")
    required_launch = {
        "run_id", "task", "zone", "image", "source", "mount_root", "priority",
        "gpu_count", "seed", "protocol", "runtime_install", "runtime_download",
    }
    if set(launch) != required_launch:
        raise ValueError(f"{task}: launch-request field set mismatch")
    if launch["task"] != task or launch["protocol"] != PROTOCOL:
        raise ValueError(f"{task}: launch request identity mismatch")
    if type(launch["seed"]) is not int or type(launch["gpu_count"]) is not int:
        raise ValueError(f"{task}: launch seed/GPU count must be integers")
    if launch["seed"] != 42 or launch["gpu_count"] != 1:
        raise ValueError(f"{task}: launch request must bind seed 42 and one GPU")
    if launch["image"] != PINNED_IMAGE:
        raise ValueError(f"{task}: launch request image is not the pinned digest")
    if launch["runtime_install"] is not False or launch["runtime_download"] is not False:
        raise ValueError(f"{task}: runtime install/download must be disabled")
    if (
        not isinstance(launch["run_id"], str)
        or SAFE_ID.fullmatch(launch["run_id"]) is None
        or not isinstance(launch["zone"], str)
        or SAFE_ID.fullmatch(launch["zone"]) is None
        or not isinstance(launch["source"], str)
        or not Path(launch["source"]).is_absolute()
        or not isinstance(launch["mount_root"], str)
        or not Path(launch["mount_root"]).is_absolute()
        or not isinstance(launch["priority"], str)
        or not launch["priority"]
    ):
        raise ValueError(f"{task}: malformed launch provenance")

    runtime = _load_json(anchor_dir / "runtime.json")
    required_runtime = {
        "event", "task", "zone", "image", "gpu_count", "gpu_name", "torch",
    }
    if set(runtime) != required_runtime:
        raise ValueError(f"{task}: runtime field set mismatch")
    if type(runtime["gpu_count"]) is not int:
        raise ValueError(f"{task}: runtime GPU count must be an integer")
    if runtime != {
        "event": "SIMP_ANCHOR_RUNTIME",
        "task": task,
        "zone": launch["zone"],
        "image": PINNED_IMAGE,
        "gpu_count": 1,
        "gpu_name": runtime.get("gpu_name"),
        "torch": runtime.get("torch"),
    }:
        raise ValueError(f"{task}: runtime identity mismatch")
    if not isinstance(runtime["gpu_name"], str) or not runtime["gpu_name"].strip():
        raise ValueError(f"{task}: missing runtime GPU name")
    if not isinstance(runtime["torch"], str) or not runtime["torch"].strip():
        raise ValueError(f"{task}: missing runtime torch version")

    baseline_lines = tuple(
        line.strip()
        for line in (anchor_dir / "baselines.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if baseline_lines != baselines:
        raise ValueError(
            f"{task}: worker baseline order {baseline_lines} != configured {baselines}"
        )
    observed_logs = {
        path.name.removesuffix(".worker.log")
        for path in anchor_dir.glob("*.worker.log")
    }
    if observed_logs != set(baselines):
        raise ValueError(f"{task}: raw log set does not match configured baselines")

    protocol_path = anchor_dir / "protocol.sha256"
    protocol_hashes = _read_protocol_hashes(
        protocol_path,
        _protocol_paths(task, config),
        config_status=str(config.get("calibration_status", "")),
    )
    parser_module = _load_module(task_dir / "parser.py", f"simp_anchor_parser_{task}")
    if (
        parser_module.PROTOCOL != PROTOCOL
        or parser_module.EXPECTED_TASK != task
        or parser_module.EXPECTED_COUNTS != EXPECTED_COUNTS
        or parser_module.INVENTORY_SHA256 != common.DATA_INVENTORY_SHA256
    ):
        raise ValueError(f"{task}: parser/common protocol constants disagree")

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    baseline_evidence: list[dict[str, Any]] = []
    used_models: set[str] = set()
    for baseline in baselines:
        _require_zero(anchor_dir / f"{baseline}.rc")
        _require_zero(anchor_dir / f"{baseline}.parser.rc")
        log_path = anchor_dir / f"{baseline}.worker.log"
        raw_sha = _sha256_file(log_path)
        _require_text(
            anchor_dir / f"{baseline}.worker.log.sha256",
            f"{raw_sha}  {log_path.name}",
        )
        surface_metrics, completion = _parse_surface_log(parser_module, log_path)
        if task == "simp-model-capacity" and completion["model"] != baseline:
            raise ValueError(
                f"{task}:{baseline}: completion model {completion['model']!r} "
                "does not match the materialized baseline"
            )
        used_models.add(completion["model"])
        metrics[baseline] = surface_metrics
        baseline_evidence.append(
            {
                "baseline": baseline,
                "rc": 0,
                "parser_rc": 0,
                "raw_log_path": f"anchor_evidence/{log_path.name}",
                "raw_log_sha256": raw_sha,
                "completion": completion,
                "metrics": surface_metrics,
            }
        )

    models = {}
    for choice in sorted(used_models):
        if choice not in common.MODEL_SPECS:
            raise ValueError(f"{task}: completion references unknown model {choice}")
        models[choice] = {
            "manifest_sha256": common.MODEL_SPEC_SHA256[choice],
            **common.MODEL_SPECS[choice],
        }

    evidence = {
        "schema": "mlsbench.text-simplification.anchor-evidence.v2",
        "task": task,
        "protocol": PROTOCOL,
        "surface": parser_module.EXPECTED_SURFACE,
        "seed": 42,
        "settings": EXPECTED_COUNTS,
        "data_inventory": common.DATA_INVENTORY,
        "data_inventory_sha256": common.DATA_INVENTORY_SHA256,
        "launch": launch,
        "runtime": runtime,
        "worker": worker,
        "started": started_raw,
        "success": success_raw,
        "task_rc": 0,
        "protocol_files": protocol_hashes,
        "protocol_sha256_file_sha256": _sha256_file(protocol_path),
        "models": models,
        "baselines": baseline_evidence,
    }
    return TaskAnchor(
        task=task,
        task_dir=task_dir,
        anchor_dir=anchor_dir,
        config=config,
        baselines=baselines,
        metrics=metrics,
        evidence=evidence,
        evidence_files=_evidence_bytes(anchor_dir, baselines),
        timestamp=success_raw,
    )


def _render_leaderboard(anchor: TaskAnchor) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        ["timestamp", "model", "is_final", "seed", "sari_asset", "sari_turk", "sari_wiki"]
    )
    for baseline in anchor.baselines:
        writer.writerow(
            [
                anchor.timestamp,
                f"baseline:{baseline}",
                "true",
                42,
                *(
                    f"{anchor.metrics[baseline][setting]['sari']:.6f}"
                    for setting in EXPECTED_COUNTS
                ),
            ]
        )
    return stream.getvalue()


def _render_score_spec(anchor: TaskAnchor, calibration: TaskCalibration) -> str:
    date = anchor.timestamp[:10]
    lines = [
        f'"""Fresh strict full-split score spec for {anchor.task}.',
        "",
        "Every configured baseline completed gem-full-test-v2 with rc=0 and",
        "strict parser acceptance on one GPU. The task config cryptographically",
        "binds the tracked raw logs, worker, image, protocol files, data inventory,",
        "model manifests, and positive finite completion times.",
        "",
        f"Measured endpoints ({date}):",
    ]
    for setting in EXPECTED_COUNTS:
        weak = anchor.metrics[calibration.weak][setting]["sari"]
        strong = anchor.metrics[calibration.strong][setting]["sari"]
        measured_floor = min(
            anchor.metrics[baseline][setting]["sari"]
            for baseline in anchor.baselines
        )
        if weak != measured_floor:
            raise ValueError(
                f"{anchor.task}: named weak endpoint {calibration.weak}={weak} "
                f"is not the measured floor {measured_floor} on {setting}"
            )
        if strong <= weak:
            raise ValueError(
                f"{anchor.task}: {calibration.strong} must exceed "
                f"{calibration.weak} on {setting}: {strong} <= {weak}"
            )
        lines.append(
            f"    {setting}: floor={weak:.6f} -> 0.0, "
            f"reference={strong:.6f} -> 0.5"
        )
    lines.extend(['"""', "from mlsbench.scoring.dsl import *", ""])
    for setting in EXPECTED_COUNTS:
        weak = anchor.metrics[calibration.weak][setting]["sari"]
        strong = anchor.metrics[calibration.strong][setting]["sari"]
        scale = (strong - weak) / math.log(3.0)
        lines.append(
            f'term("sari_{setting}", col("sari_{setting}").higher().id()'
            f'.sigmoid(ref=const({strong:.6f}), scale={scale:.15g}))'
        )
    lines.extend(
        [
            "",
            'setting("asset", weighted_mean(("sari_asset", 1.0)))',
            'setting("turk", weighted_mean(("sari_turk", 1.0)))',
            'setting("wiki", weighted_mean(("sari_wiki", 1.0)))',
            "",
            'task(gmean("asset", "turk", "wiki"))',
            "",
        ]
    )
    return "\n".join(lines)


def _render_config(
    anchor: TaskAnchor,
    evidence_sha256: str,
) -> str:
    config = dict(anchor.config)
    config["calibration_status"] = "measured_fresh_strict_full_official_split_anchors"
    config["calibration_protocol"] = PROTOCOL
    config["calibration_anchor_date"] = anchor.timestamp[:10]
    config["calibration_anchor_seed"] = 42
    config["calibration_anchor_counts"] = EXPECTED_COUNTS
    config["calibration_anchor_evidence_path"] = "anchor_evidence/manifest.json"
    config["calibration_anchor_evidence_sha256"] = evidence_sha256
    config.pop("calibration_pending_marker", None)
    return json.dumps(config, indent=2, sort_keys=False) + "\n"


def _write_or_check_bytes(path: Path, expected: bytes, *, write: bool) -> None:
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
        return
    if not path.is_file() or path.read_bytes() != expected:
        raise ValueError(f"stale or missing strict anchor evidence: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.anchor_root.is_dir():
        raise FileNotFoundError(args.anchor_root)

    common = _load_module(
        ROOT / "vendor/text-simplification/common.py",
        "simp_anchor_writeback_common",
    )
    if common.PROTOCOL != PROTOCOL or common.SETTING_COUNTS != EXPECTED_COUNTS:
        raise ValueError("writeback/common protocol constants disagree")

    # Validate the complete ten-task family before changing any tracked file.
    anchors = {
        task: _load_task_anchor(args.anchor_root, task, common)
        for task in CALIBRATIONS
    }
    outputs: dict[Path, bytes] = {}
    for task, calibration in CALIBRATIONS.items():
        anchor = anchors[task]
        evidence_text = (
            json.dumps(anchor.evidence, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        evidence_sha = _sha256_bytes(evidence_text)
        outputs[anchor.task_dir / "anchor_evidence/manifest.json"] = evidence_text
        for name, payload in anchor.evidence_files.items():
            outputs[anchor.task_dir / "anchor_evidence" / name] = payload
        outputs[anchor.task_dir / "leaderboard.csv"] = _render_leaderboard(
            anchor
        ).encode("utf-8")
        outputs[anchor.task_dir / "score_spec.py"] = _render_score_spec(
            anchor, calibration
        ).encode("utf-8")
        outputs[anchor.task_dir / "config.json"] = _render_config(
            anchor, evidence_sha
        ).encode("utf-8")

    for path, expected in sorted(outputs.items(), key=lambda item: str(item[0])):
        _write_or_check_bytes(path, expected, write=args.write)
    for anchor in anchors.values():
        marker = anchor.task_dir / "PENDING_FULL_OFFICIAL_ANCHORS"
        if args.write:
            marker.unlink(missing_ok=True)
        elif marker.exists():
            raise ValueError(f"measured task still has pending marker: {marker}")
        print(
            f"{anchor.task}: validated {len(anchor.baselines)} strict baselines; "
            f"worker={anchor.evidence['worker']}"
        )


if __name__ == "__main__":
    main()
