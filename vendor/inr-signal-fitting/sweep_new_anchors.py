#!/usr/bin/env python3
"""Measure checked-in INR candidate configurations without assuming their ranking.

Every candidate is evaluated through ``common.fit_surface``, the same fixed builder used
by the verifier. Results are written only after every requested run succeeds. This tool
does not assign weak/strong/SOTA roles and never rewrites tasks or score specifications.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import common


HERE = Path(__file__).resolve().parent
SIGNALS = ("low", "medium", "high")
SURFACES = {
    "inr-activation": "activation",
    "inr-eikonal-reg": "jacobian_reg",
    "inr-encoding-dim": "encoding_dim",
    "inr-fourier-frequency": "frequency",
    "inr-hash-grid": "hash_grid",
    "inr-lr-schedule": "lr_schedule",
    "inr-network-depth": "depth",
    "inr-network-width": "width",
    "inr-per-layer-w0": "per_layer_w0",
    "inr-skip-connections": "skip",
}
CANDIDATES = {
    "inr-activation": {
        "relu_mlp": {"family": "relu"},
        "fourier_mlp": {"family": "fourier"},
        "siren": {"family": "siren"},
    },
    "inr-eikonal-reg": {
        "jacobian_high": {"weight": 1.0},
        "jacobian_low": {"weight": 0.01},
        "jacobian_zero": {"weight": 0.0},
    },
    "inr-encoding-dim": {
        "nfreq2": {"num_freqs": 2},
        "nfreq32": {"num_freqs": 32},
        "nfreq128": {"num_freqs": 128},
    },
    "inr-fourier-frequency": {
        "sigma_low": {"sigma": 0.3},
        "sigma_high": {"sigma": 100.0},
        "sigma_tuned": {"sigma": 10.0},
    },
    "inr-hash-grid": {
        "collapsed": {"n_levels": 1, "base_res": 4, "finest_res": 4},
        "pyramid8": {"n_levels": 8, "base_res": 16, "finest_res": 256},
        "pyramid16": {"n_levels": 16, "base_res": 16, "finest_res": 256},
    },
    "inr-lr-schedule": {
        "lr_big_const": {"lr": 0.2, "schedule": "constant"},
        "lr_good_const": {"lr": 5e-4, "schedule": "constant"},
        "lr_good_cosine": {"lr": 5e-4, "schedule": "cosine"},
    },
    "inr-network-depth": {
        "depth1": {"n_layers": 1},
        "depth4": {"n_layers": 4},
        "depth10": {"n_layers": 10},
    },
    "inr-network-width": {
        "width8": {"hidden": 8},
        "width64": {"hidden": 64},
        "width256": {"hidden": 256},
    },
    "inr-per-layer-w0": {
        "w0_3": {"first": 3.0, "hidden": 3.0},
        "w0_15": {"first": 15.0, "hidden": 15.0},
        "w0_30": {"first": 30.0, "hidden": 30.0},
    },
    "inr-skip-connections": {
        "noskip": {"skip_at": None},
        "skip4": {"skip_at": 4},
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measure(surface: str, plan: dict, signal: str, seed: int) -> dict:
    common.set_seeds(seed)
    started = time.monotonic()
    coords, target, resolution = common.load_signal(signal)
    common.set_seeds(seed)
    predictor = common.fit_surface(surface, plan, coords, target, common.device())
    if not callable(predictor):
        raise TypeError("fixed surface builder did not return a predictor")
    with torch.no_grad():
        prediction = predictor(coords)
    prediction = common.require_rgb_prediction(
        prediction, coords.shape[0], "anchor prediction"
    )
    psnr = common.psnr_db(prediction, target)
    elapsed = time.monotonic() - started
    if not math.isfinite(psnr) or not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("anchor measurement produced non-finite output")
    return {"psnr": psnr, "elapsed": elapsed, "resolution": resolution}


def _atomic_json(path: Path, value: dict, overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing evidence file {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", choices=sorted(SURFACES))
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--signal", action="append", choices=SIGNALS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    tasks = args.task or sorted(SURFACES)
    signals = tuple(dict.fromkeys(args.signal or SIGNALS))
    requested_candidates = set(args.candidate or ())
    if requested_candidates and len(tasks) != 1:
        parser.error("--candidate requires exactly one --task")

    results = {}
    for task in tasks:
        candidates = CANDIDATES[task]
        names = list(candidates)
        if requested_candidates:
            unknown = requested_candidates - set(candidates)
            if unknown:
                parser.error(f"unknown candidates for {task}: {sorted(unknown)}")
            names = [name for name in names if name in requested_candidates]
        task_rows = {}
        for name in names:
            plan = candidates[name]
            signal_rows = {}
            for signal in signals:
                row = _measure(SURFACES[task], plan, signal, args.seed)
                signal_rows[signal] = row
                print(
                    f"MEASURED task={task} candidate={name} signal={signal} "
                    f"psnr={row['psnr']:.6f} elapsed={row['elapsed']:.1f}",
                    flush=True,
                )
            task_rows[name] = {"config": plan, "settings": signal_rows}
        rankings = {
            signal: sorted(
                names,
                key=lambda name: task_rows[name]["settings"][signal]["psnr"],
                reverse=True,
            )
            for signal in signals
        }
        rankings["mean_psnr"] = sorted(
            names,
            key=lambda name: sum(
                task_rows[name]["settings"][signal]["psnr"] for signal in signals
            ) / len(signals),
            reverse=True,
        )
        results[task] = {"candidates": task_rows, "rankings": rankings}

    data_root = Path(os.environ.get("INR_DATA", "/data/inr-signal-fitting"))
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "signals": list(signals),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "steps": common.STEPS,
        },
        "source_sha256": {
            "common.py": _sha256(HERE / "common.py"),
            "sweep_new_anchors.py": _sha256(Path(__file__).resolve()),
        },
        "data": {
            "root": str(data_root.resolve()),
            "npz_sha256": {
                signal: _sha256(data_root / f"{signal}.npz") for signal in signals
            },
        },
        "results": results,
    }
    _atomic_json(args.output, record, args.overwrite)
    print(f"WROTE {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
