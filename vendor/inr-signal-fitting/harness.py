#!/usr/bin/env python3
"""INR verification runtime component.











"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import torch

import common


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--signal", required=True, choices=["low", "medium", "high"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="eval")
    args = ap.parse_args()

    common.set_seeds(args.seed)
    t0 = time.time()
    dev = common.device()

    coords, target, res = common.load_signal(args.signal)
    print(f"DATA_INFO signal={args.signal} res={res} n_coords={coords.shape[0]} "
          f"dev={dev}", flush=True)

    common.set_seeds(args.seed)
    plan = common.load_surface_config(args.solution)
    surface = Path(args.solution).stem
    predict = common.fit_surface(surface, plan, coords, target, dev)
    if not callable(predict):
        raise TypeError("fixed surface builder must return callable predict(coords) -> [N, 3]")
    with torch.no_grad():
        probe = predict(coords[:8])
    common.require_rgb_prediction(probe, 8, "predict probe")

    with torch.no_grad():
        pred = predict(coords)
    pred = common.require_rgb_prediction(pred, coords.shape[0], "final prediction")

    psnr = common.psnr_db(pred, target)
    if not math.isfinite(psnr):
        raise RuntimeError("PSNR is non-finite")
    dt = time.time() - t0
    print(f"INR_METRICS signal={args.signal} psnr={psnr:.6f} res={res} "
          f"elapsed={dt:.1f}", flush=True)
    print(
        f"INR_DONE signal={args.signal} n_coords={coords.shape[0]} "
        f"steps={common.STEPS} seed={args.seed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
