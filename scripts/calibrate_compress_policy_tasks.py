#!/usr/bin/env python3
"""Replay accepted family proofs into measured policy-task calibrations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("factorized", "hyperprior_scale", "meanscale")
SETTINGS = ("full", "low", "mid", "high")
QUALITIES = tuple(range(1, 9))
GROUPS = {
    "low": {
        "kodim02.png", "kodim03.png", "kodim09.png", "kodim10.png",
        "kodim12.png", "kodim15.png", "kodim20.png", "kodim23.png",
    },
    "mid": {
        "kodim04.png", "kodim07.png", "kodim11.png", "kodim16.png",
        "kodim17.png", "kodim19.png", "kodim21.png", "kodim22.png",
    },
    "high": {
        "kodim01.png", "kodim05.png", "kodim06.png", "kodim08.png",
        "kodim13.png", "kodim14.png", "kodim18.png", "kodim24.png",
    },
}
ALL_IMAGES = set().union(*GROUPS.values())
QUALITY_BANDS = {
    "low": (1, 2, 3),
    "mid": (4, 5, 6),
    "high": (7, 8),
}
TARGET_BPP = {
    1: 0.15,
    2: 0.25,
    3: 0.35,
    4: 0.50,
    5: 0.70,
    6: 0.90,
    7: 1.20,
    8: 1.50,
}
TARGET_PENALTY = 12.0
EXPECTED_STREAMS = {
    "factorized": 1,
    "hyperprior_scale": 2,
    "meanscale": 2,
}
EXPECTED_PROOF_SHA = {
    "factorized": "83de02d6f531f29a0eb936f3726c83a1ae7bdd4328504c4aeb4d34b21db1b01a",
    "hyperprior_scale": "c4d117e664a9f8547e0495d1b0913985a8582d79f7971c2b45e2f28c5842bdaa",
    "meanscale": "3c5a68541e2710b21b98ecaf1088ec8e06f6d32a939399e1569c07a9c5827a6b",
}
TASK_SPECS = {
    "compress-content-dispatch": {"objective": "rd12"},
    "compress-quality-dispatch": {"objective": "rd12"},
    "compress-low-rate-policy": {"objective": "lowq_rd12"},
    "compress-parameter-budget": {
        "objective": "rd12",
        "constraint": "mean_params",
    },
    "compress-objective-policy": {"objective": "rd6"},
    "compress-robust-policy": {"objective": "rd18"},
    "compress-bitrate-policy": {"objective": "target_utility"},
    "compress-stream-budget": {
        "objective": "rd12",
        "constraint": "mean_streams",
    },
    "compress-high-rate-policy": {"objective": "highq_rd12"},
}
HEX64 = r"[0-9a-f]{64}"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module)
    return module


def mean(records: list[dict], key: str) -> float:
    if not records:
        raise ValueError(f"empty records for {key}")
    return sum(float(record[key]) for record in records) / len(records)


def parse_proof(family: str, path: Path) -> dict:
    if sha256(path) != EXPECTED_PROOF_SHA[family]:
        raise ValueError(f"accepted proof digest mismatch for {family}")
    raw = path.read_text()
    strict_parser = load_module(
        f"entropy_parser_{family}",
        ROOT / "tasks" / "compress-entropy-model" / "parser.py",
    ).Parser()
    parsed = strict_parser.parse("kodak24_q1q8", raw)
    if len(parsed.metrics) != 12:
        raise ValueError(f"strict proof parse failed for {family}: {parsed.feedback}")

    model_re = re.compile(
        rf"^COMPRESS_MODEL quality=(\d+) checkpoint_sha=({HEX64}) "
        rf"cdf_sha=({HEX64}) params=(\d+) updated=1$"
    )
    case_re = re.compile(
        rf"^COMPRESS_CASE quality=(\d+) image=(kodim\d{{2}}\.png) "
        rf"group=(low|mid|high) pixels=(\d+) bytes=(\d+) bpp=({NUMBER}) "
        rf"psnr=({NUMBER}) recon_sha=({HEX64})$"
    )
    final_re = re.compile(
        rf"^COMPRESS_FINAL .* family={family} images=24 qualities=8 cases=192 "
        rf"streams=(\d+) cases_sha=({HEX64}) checkpoints_sha=({HEX64}) "
        rf"settings_sha=({HEX64}) elapsed=({NUMBER})$"
    )
    models = []
    for line in raw.splitlines():
        match = model_re.fullmatch(line)
        if match:
            models.append(
                {
                    "quality": int(match.group(1)),
                    "checkpoint_sha": match.group(2),
                    "cdf_sha": match.group(3),
                    "params": int(match.group(4)),
                }
            )
    if [record["quality"] for record in models] != list(QUALITIES):
        raise ValueError(f"model inventory mismatch for {family}")
    model_by_quality = {record["quality"]: record for record in models}
    cases = []
    for line in raw.splitlines():
        match = case_re.fullmatch(line)
        if not match:
            continue
        quality = int(match.group(1))
        model = model_by_quality[quality]
        cases.append(
            {
                "family": family,
                "quality": quality,
                "image": match.group(2),
                "group": match.group(3),
                "checkpoint_sha": model["checkpoint_sha"],
                "params": model["params"],
                "streams": EXPECTED_STREAMS[family],
                "pixels": int(match.group(4)),
                "bytes": int(match.group(5)),
                "bpp": float(match.group(6)),
                "psnr": float(match.group(7)),
                "recon_sha": match.group(8),
            }
        )
    expected_order = [
        (quality, f"kodim{index:02d}.png")
        for quality in QUALITIES
        for index in range(1, 25)
    ]
    if [(case["quality"], case["image"]) for case in cases] != expected_order:
        raise ValueError(f"case inventory mismatch for {family}")
    finals = [
        final_re.fullmatch(line)
        for line in raw.splitlines()
        if line.startswith("COMPRESS_FINAL")
    ]
    if len(finals) != 1 or finals[0] is None:
        raise ValueError(f"terminal proof mismatch for {family}")
    final = finals[0]
    if int(final.group(1)) != 192 * EXPECTED_STREAMS[family]:
        raise ValueError(f"stream inventory mismatch for {family}")
    return {
        "family": family,
        "proof": str(path),
        "proof_sha256": EXPECTED_PROOF_SHA[family],
        "models": models,
        "cases": cases,
        "case_completion_sha": final.group(2),
        "model_completion_sha": final.group(3),
        "setting_completion_sha": final.group(4),
        "elapsed": float(final.group(5)),
    }


def replay_uniform(proof: dict) -> dict:
    cases = proof["cases"]
    rates = []
    settings = []
    for setting in SETTINGS:
        names = ALL_IMAGES if setting == "full" else GROUPS[setting]
        per_quality = []
        for quality in QUALITIES:
            selected = [
                case
                for case in cases
                if case["quality"] == quality and case["image"] in names
            ]
            total_pixels = sum(case["pixels"] for case in selected)
            total_bytes = sum(case["bytes"] for case in selected)
            bpp = 8.0 * total_bytes / total_pixels
            psnr = mean(selected, "psnr")
            record = {
                "setting": setting,
                "quality": quality,
                "count": len(selected),
                "pixels": total_pixels,
                "bytes": total_bytes,
                "bpp": float(f"{bpp:.12f}"),
                "psnr": float(f"{psnr:.9f}"),
                "rd6": float(f"{psnr - 6.0 * bpp:.9f}"),
                "rd12": float(f"{psnr - 12.0 * bpp:.9f}"),
                "rd18": float(f"{psnr - 18.0 * bpp:.9f}"),
                "target_utility": float(
                    f"{psnr - TARGET_PENALTY * abs(bpp - TARGET_BPP[quality]):.9f}"
                ),
                "mean_params": float(f"{mean(selected, 'params'):.3f}"),
                "mean_streams": float(f"{mean(selected, 'streams'):.6f}"),
            }
            rates.append(record)
            per_quality.append(record)

        def band_mean(band: str) -> float:
            records = [
                record
                for record in per_quality
                if record["quality"] in QUALITY_BANDS[band]
            ]
            return mean(records, "rd12")

        settings.append(
            {
                "setting": setting,
                "rd6": float(f"{mean(per_quality, 'rd6'):.9f}"),
                "rd12": float(f"{mean(per_quality, 'rd12'):.9f}"),
                "rd18": float(f"{mean(per_quality, 'rd18'):.9f}"),
                "lowq_rd12": float(f"{band_mean('low'):.9f}"),
                "midq_rd12": float(f"{band_mean('mid'):.9f}"),
                "highq_rd12": float(f"{band_mean('high'):.9f}"),
                "target_utility": float(f"{mean(per_quality, 'target_utility'):.9f}"),
                "mean_psnr": float(f"{mean(per_quality, 'psnr'):.9f}"),
                "mean_bpp": float(f"{mean(per_quality, 'bpp'):.12f}"),
                "mean_params": float(f"{mean(per_quality, 'mean_params'):.3f}"),
                "mean_streams": float(f"{mean(per_quality, 'mean_streams'):.6f}"),
            }
        )
    return {
        "family": proof["family"],
        "elapsed": proof["elapsed"],
        "rates": rates,
        "settings": settings,
        "metrics": {
            f"{key}_{record['setting']}": value
            for record in settings
            for key, value in record.items()
            if key != "setting"
        },
    }


def score_spec_source(task_spec: dict, calibration: dict, constraints: dict) -> str:
    objective = task_spec["objective"]
    lines = [
        '"""Measured replay calibration from three accepted 192-case family proofs."""',
        "from mlsbench.scoring.dsl import *",
        "",
        f"_CALIBRATION = {calibration!r}",
        "",
        f"for _setting in {SETTINGS!r}:",
        f"    _metric = f\"{objective}_{{_setting}}\"",
        "    _values = _CALIBRATION[_setting]",
        "    term(",
        "        _metric,",
        "        col(_metric).higher().id().sigmoid(",
        "            ref=const(_values['midpoint']),",
        "            scale=_values['scale'],",
        "        ),",
        "    )",
    ]
    constraint = task_spec.get("constraint")
    if constraint:
        constraint_values = constraints[constraint]
        lines.extend(
            [
                f"    _constraint = f\"{constraint}_{{_setting}}\"",
                f"    _constraint_values = {constraint_values!r}",
                "    term(",
                "        _constraint,",
                "        penalty_upper(",
                "            col(_constraint).lower().id(),",
                "            target=_constraint_values['target'],",
                "            sharpness=_constraint_values['sharpness'],",
                "        ),",
                "    )",
                "    setting(",
                "        _setting,",
                "        weighted_mean((_metric, 1.0)),",
                "        constraints=[_constraint],",
                "    )",
            ]
        )
    else:
        lines.append("    setting(_setting, weighted_mean((_metric, 1.0)))")
    lines.extend(["", 'task(gmean("full", "low", "mid", "high"))', ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--source-commit", required=True)
    for family in FAMILIES:
        parser.add_argument(f"--{family.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to reuse calibration output: {args.output}")
    args.output.mkdir(parents=True)

    proofs = {
        family: parse_proof(family, getattr(args, family))
        for family in FAMILIES
    }
    replays = {family: replay_uniform(proofs[family]) for family in FAMILIES}
    if sum(len(proofs[family]["cases"]) for family in FAMILIES) != 576:
        raise SystemExit("calibration replay did not consume all 576 measured cases")

    parameter_values = [
        replays[family]["metrics"]["mean_params_full"]
        for family in FAMILIES
    ]
    parameter_target = replays["hyperprior_scale"]["metrics"]["mean_params_full"]
    parameter_excess = max(max(parameter_values) - parameter_target, 1.0)
    constraints = {
        "mean_params": {
            "target": parameter_target,
            "sharpness": math.log(2.0) / parameter_excess,
        },
        "mean_streams": {
            "target": 1.0,
            "sharpness": math.log(2.0),
        },
    }
    manifest = {
        "status": "success",
        "source_commit": args.source_commit,
        "accepted_proofs": {
            family: {
                key: value
                for key, value in proofs[family].items()
                if key not in {"models", "cases"}
            }
            for family in FAMILIES
        },
        "replayed_family_cases": 576,
        "constraints": constraints,
        "tasks": {},
    }

    from mlsbench.scoring.anchors import BaselineAnchors
    from mlsbench.scoring.evaluate import score_record_details
    from mlsbench.scoring.spec import load_score_spec

    output_tasks = args.output / "tasks"
    output_tasks.mkdir()
    for task_id, task_spec in TASK_SPECS.items():
        objective = task_spec["objective"]
        calibration = {}
        for setting in SETTINGS:
            values = {
                family: replays[family]["metrics"][f"{objective}_{setting}"]
                for family in FAMILIES
            }
            midpoint = values["hyperprior_scale"]
            width = max(
                abs(midpoint - values["factorized"]),
                abs(values["meanscale"] - midpoint),
            )
            scale = width / math.log(4.0)
            if not math.isfinite(scale) or scale <= 0:
                raise SystemExit(f"degenerate measured calibration for {task_id}/{setting}")
            calibration[setting] = {
                "factorized": values["factorized"],
                "hyperprior_scale": values["hyperprior_scale"],
                "meanscale": values["meanscale"],
                "midpoint": midpoint,
                "scale": scale,
            }

        task_output = output_tasks / task_id
        task_output.mkdir()
        constraint = task_spec.get("constraint")
        metric_columns = [f"{objective}_{setting}" for setting in SETTINGS]
        if constraint:
            metric_columns.extend(f"{constraint}_{setting}" for setting in SETTINGS)
        leaderboard_path = task_output / "leaderboard.csv"
        with leaderboard_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp", "model", "is_final", "seed",
                    *metric_columns, "elapsed",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            for family in FAMILIES:
                metrics = replays[family]["metrics"]
                writer.writerow(
                    {
                        "timestamp": args.timestamp,
                        "model": f"baseline:uniform_{family}",
                        "is_final": "true",
                        "seed": 42,
                        **{column: metrics[column] for column in metric_columns},
                        "elapsed": replays[family]["elapsed"],
                    }
                )
        (task_output / "score_spec.py").write_text(
            score_spec_source(task_spec, calibration, constraints)
        )

        spec = load_score_spec(task_output)
        if spec is None:
            raise SystemExit(f"generated score spec failed to load for {task_id}")
        anchors = BaselineAnchors(task_output)
        predicted = {}
        rows = list(csv.DictReader(leaderboard_path.open(newline="")))
        for row in rows:
            record = {
                column: float(row[column])
                for column in metric_columns
            }
            score, setting_results, valid = score_record_details(spec, record, anchors)
            if not valid or len(setting_results) != 4 or not math.isfinite(score):
                raise SystemExit(f"generated baseline did not score for {task_id}")
            predicted[row["model"]] = score
        missing = {
            column: float(rows[-1][column])
            for column in metric_columns
        }
        del missing[metric_columns[0]]
        missing_score, _settings, missing_valid = score_record_details(spec, missing, anchors)
        if missing_valid or missing_score != 0.0:
            raise SystemExit(f"missing metric did not fail closed for {task_id}")

        calibration_record = {
            "task_id": task_id,
            "objective": objective,
            "constraint": constraint,
            "calibration": calibration,
            "predicted_scores": predicted,
            "proof_sha256": dict(EXPECTED_PROOF_SHA),
            "replayed_family_cases": 576,
        }
        (task_output / "calibration.json").write_text(
            json.dumps(calibration_record, indent=2, sort_keys=True) + "\n"
        )
        manifest["tasks"][task_id] = calibration_record

    for cache_dir in args.output.rglob("__pycache__"):
        shutil.rmtree(cache_dir)
    (args.output / "policy_calibration.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with (args.output / "files.sha256").open("w") as handle:
        for path in sorted(args.output.rglob("*")):
            if path.is_file() and path.name != "files.sha256":
                handle.write(f"{sha256(path)}  {path.relative_to(args.output)}\n")
    print(
        "COMPRESS_POLICY_CALIBRATION_READY "
        f"cases=576 tasks={len(TASK_SPECS)} "
        + " ".join(
            f"{family}_proof={EXPECTED_PROOF_SHA[family]}"
            for family in FAMILIES
        )
    )


if __name__ == "__main__":
    main()
