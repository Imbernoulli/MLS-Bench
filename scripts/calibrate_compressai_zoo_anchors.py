#!/usr/bin/env python3
"""Strictly parse three measured anchors and emit private calibration files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


FAMILIES = ("factorized", "hyperprior_scale", "meanscale")
SETTINGS = ("full", "low", "mid", "high")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_parser_types() -> None:
    try:
        from mlsbench.agent.parsers import OutputParser, ParseResult  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class OutputParser:
        pass

    @dataclass
    class ParseResult:
        feedback: str
        metrics: dict

    root = types.ModuleType("mlsbench")
    agent = types.ModuleType("mlsbench.agent")
    parsers = types.ModuleType("mlsbench.agent.parsers")
    parsers.OutputParser = OutputParser
    parsers.ParseResult = ParseResult
    root.agent = agent
    agent.parsers = parsers
    sys.modules.update(
        {
            "mlsbench": root,
            "mlsbench.agent": agent,
            "mlsbench.agent.parsers": parsers,
        }
    )


def load_parser(path: Path):
    install_parser_types()
    spec = importlib.util.spec_from_file_location("compress_calibration_parser", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("parser loader is unavailable")
    spec.loader.exec_module(module)
    return module


def logistic(value: float, ref: float, scale: float) -> float:
    return 1.0 / (1.0 + math.exp(-(value - ref) / scale))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parser", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for family in FAMILIES:
        parser.add_argument(f"--{family.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    if output.exists():
        raise SystemExit(f"refusing to reuse calibration output: {output}")
    output.mkdir(parents=True)
    parser_module = load_parser(args.parser)
    paths = {
        family: getattr(args, family)
        for family in FAMILIES
    }
    records = {}
    manifest = {
        "status": "success",
        "parser_sha256": sha256(args.parser),
        "anchors": {},
    }
    final_re = re.compile(r"^COMPRESS_FINAL .* family=(\w+) .* elapsed=([0-9.]+)$")
    for family, path in paths.items():
        raw = path.read_text()
        result = parser_module.Parser().parse("kodak24_q1q8", raw)
        if len(result.metrics) != 12:
            raise SystemExit(f"strict parse failed for {family}")
        finals = [final_re.fullmatch(line) for line in raw.splitlines() if line.startswith("COMPRESS_FINAL ")]
        if len(finals) != 1 or finals[0] is None or finals[0].group(1) != family:
            raise SystemExit(f"terminal family proof mismatch for {family}")
        if any(not math.isfinite(float(value)) for value in result.metrics.values()):
            raise SystemExit(f"non-finite parsed metric for {family}")
        records[family] = dict(result.metrics)
        records[family]["elapsed"] = float(finals[0].group(2))
        manifest["anchors"][family] = {
            "proof": str(path),
            "proof_sha256": sha256(path),
            "cases": 192,
            "models": 8,
            "metrics": result.metrics,
            "harness_elapsed": float(finals[0].group(2)),
        }

    calibration = {}
    predicted = {family: {} for family in FAMILIES}
    for setting in SETTINGS:
        key = f"mean_rd_utility_{setting}"
        weak = records["factorized"][key]
        native = records["hyperprior_scale"][key]
        strong = records["meanscale"][key]
        if not weak < native < strong:
            raise SystemExit(f"measured family ordering failed for {setting}")
        scale = max(native - weak, strong - native) / math.log(4.0)
        if not math.isfinite(scale) or scale <= 0:
            raise SystemExit(f"invalid measured scale for {setting}")
        calibration[setting] = {
            "weak_factorized": weak,
            "native_hyperprior_scale": native,
            "strong_meanscale": strong,
            "ref": native,
            "scale": scale,
        }
        for family in FAMILIES:
            predicted[family][setting] = logistic(records[family][key], native, scale)

    for family in FAMILIES:
        predicted[family]["combined"] = math.prod(
            predicted[family][setting] for setting in SETTINGS
        ) ** (1.0 / len(SETTINGS))

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metric_columns = [
        f"{metric}_{setting}"
        for setting in SETTINGS
        for metric in ("mean_rd_utility", "psnr", "bpp")
    ]
    with (output / "leaderboard.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "model", "is_final", "seed", *metric_columns, "elapsed"],
        )
        writer.writeheader()
        for family in FAMILIES:
            writer.writerow(
                {
                    "timestamp": timestamp,
                    "model": f"baseline:official_zoo:{family}",
                    "is_final": "true",
                    "seed": 42,
                    **{column: records[family][column] for column in metric_columns},
                    "elapsed": records[family]["elapsed"],
                }
            )

    score_lines = [
        '"""Measured CompressAI-1.2.8 Kodak24 q1..8 calibration.',
        "",
        "The official scale-hyperprior is the 0.5 midpoint for each required",
        "setting. Logistic width is the larger measured distance to the official",
        "factorized or mean-scale anchor divided by ln(4), so the farther extreme",
        "maps to 0.2 or 0.8 without a positive floor or fallback.",
        '"""',
        "from math import log",
        "",
        "from mlsbench.scoring.dsl import *",
        "",
        f"_CALIBRATION = {repr(calibration)}",
        "",
        "for _setting, _values in _CALIBRATION.items():",
        "    _name = f\"mean_rd_utility_{_setting}\"",
        "    term(",
        "        _name,",
        "        col(_name).higher().id().sigmoid(",
        "            ref=const(_values['native_hyperprior_scale']),",
        "            scale=_values['scale'],",
        "        ),",
        "    )",
        "    setting(_setting, weighted_mean((_name, 1.0)))",
        "",
        "task(gmean('full', 'low', 'mid', 'high'))",
        "",
    ]
    (output / "score_spec.py").write_text("\n".join(score_lines))
    report = {
        "status": "success",
        "protocol": parser_module.PROTOCOL,
        "calibration": calibration,
        "predicted_scores": predicted,
        "manifest": manifest,
    }
    (output / "calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "anchors_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with (output / "files.sha256").open("w") as handle:
        for name in ("leaderboard.csv", "score_spec.py", "calibration.json", "anchors_manifest.json"):
            handle.write(f"{sha256(output / name)}  {name}\n")
    print(
        "COMPRESS_CALIBRATION_READY "
        + " ".join(
            f"{family}={predicted[family]['combined']:.12f}" for family in FAMILIES
        )
    )


if __name__ == "__main__":
    main()
