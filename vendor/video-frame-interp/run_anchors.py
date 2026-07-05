#!/usr/bin/env python3
"""Anchor driver for the 10 NEW vfi-* surfaces (reuses the validated harness + data).

For each surface, run its baseline configs across the 3 motion settings (small/medium/large)
through the REAL harness (harness.run), 800 iters seed 42 -- exactly the protocol that
produced the committed vfi-synthesis anchors. Append every (surface, cfg, motion) result to a
RESUMABLE TSV on moonfs so a preemption/restart just skips finished keys.

Runs from inside the mlaunch worker. Data root = the staged vfi_anchor_data.
"""
import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

# (surface, hook-key, [config values in weak..strong order])
MATRIX = [
    ("flow",      "kind",   ["zero", "single", "refine"]),
    ("refine",    "depth",  ["none", "shallow", "deep"]),
    ("warp",      "kind",   ["none", "forward", "backward", "softsplat"]),
    ("occlusion", "kind",   ["avg", "time", "mask"]),
    ("loss",      "kind",   ["l2", "l1", "l1_census", "l1_warp"]),
    ("fusion",    "kind",   ["warps", "plus_flow", "full"]),
    ("context",   "kind",   ["none", "shallow", "pyramid"]),
    ("scale",     "levels", ["single", "two", "three"]),
    ("attention", "kind",   ["none", "se", "nonlocal"]),
    ("iters",     "n",      [1, 2, 4]),
]
MOTIONS = ["small", "medium", "large"]

TSV_COLS = ["surface", "cfg", "motion", "psnr", "psnr_gain", "blend_psnr", "ssim", "mse"]


class _CfgMod:
    """A fake solution module that returns a fixed config dict for the surface's hook."""
    def __init__(self, surface, key, value):
        hook = H._HOOK[surface]
        setattr(self, hook, lambda k=key, v=value: {k: v})


def load_done(tsv_path):
    done = set()
    if tsv_path.exists():
        for ln in tsv_path.read_text().splitlines()[1:]:
            p = ln.split("\t")
            if len(p) >= 3:
                done.add((p[0], p[1], p[2]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default="", help="comma surfaces subset")
    args = ap.parse_args()

    tsv = Path(args.out)
    tsv.parent.mkdir(parents=True, exist_ok=True)
    if not tsv.exists():
        tsv.write_text("\t".join(TSV_COLS) + "\n")
    done = load_done(tsv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE {device} torch {torch.__version__}", flush=True)

    only = set(s for s in args.only.split(",") if s)
    t0 = time.time()
    for surface, key, values in MATRIX:
        if only and surface not in only:
            continue
        for value in values:
            cfg_str = str(value)
            for motion in MOTIONS:
                k = (surface, cfg_str, motion)
                if k in done:
                    print(f"SKIP {k}", flush=True)
                    continue
                mod = _CfgMod(surface, key, value)
                try:
                    m = H.run(surface, mod, args.data_root, device,
                              args.iters, args.seed, motion=motion)
                except Exception as e:  # noqa: BLE001
                    print(f"ERROR {k}: {e!r}", flush=True)
                    raise
                row = [surface, cfg_str, motion,
                       f"{m['psnr']:.4f}", f"{m['psnr_gain']:.4f}",
                       f"{m['blend_psnr']:.4f}", f"{m['ssim']:.4f}", f"{m['mse']:.6f}"]
                with tsv.open("a") as fh:
                    fh.write("\t".join(row) + "\n")
                done.add(k)
                print(f"ANCHOR {surface} cfg={cfg_str} {motion} "
                      f"psnr={m['psnr']:.4f} blend={m['blend_psnr']:.4f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    print("ALL ANCHORS DONE", flush=True)


if __name__ == "__main__":
    main()
