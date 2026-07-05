#!/usr/bin/env python3
"""Anchor driver for the image-deshadow repo (6 surviving surfaces x 3 settings),
post real-ISTD-data-swap re-anchoring.

Runs every baseline of every surviving surface across all 3 severities (light/medium/heavy)
by invoking harness.py as a subprocess (reuses the exact, already-reviewed main() path).
Resumable: skips (surface, baseline, setting, seed) keys already present in --out.

Usage (from vendor/image-deshadow/):
    python3 run_anchors.py --out /path/to/anchor_real.tsv --iters 400 \
        --data-root /data/image-deshadow
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "baselines")
SOL = os.path.join(HERE, "solution")

SETTINGS = ["light", "medium", "heavy"]

# surface -> (harness --surface value, [(baseline_name, baseline_source_path)])
BASELINES = {
    "network-design": ("network", [
        ("copy", os.path.join(BASE, "network_copy.py")),
        ("unet_mask", os.path.join(BASE, "network_unet_mask.py")),
    ]),
    "mask-guidance": ("network", [
        ("unet_nomask", os.path.join(BASE, "network_unet_nomask.py")),
        ("unet_mask", os.path.join(BASE, "network_unet_mask.py")),
    ]),
    "mask": ("mask", [
        ("nomask", os.path.join(BASE, "mask_off.py")),
        ("mask", os.path.join(BASE, "mask_on.py")),
    ]),
    "dilation": ("dilation", [
        ("d1", os.path.join(BASE, "dilation_d1.py")),
        ("dilated", os.path.join(BASE, "dilation_dilated.py")),
    ]),
    "fusion": ("fusion", [
        ("last", os.path.join(BASE, "fusion_last.py")),
        ("dense", os.path.join(BASE, "fusion_dense.py")),
    ]),
    "upsampling": ("upsampling", [
        ("transpose", os.path.join(BASE, "upsampling_transpose.py")),
        ("bilinear", os.path.join(BASE, "upsampling_bilinear.py")),
    ]),
}
# NOTE: architecture/attention/loss/normalization/residual/multiscale are already
# _dropped=True in tasks/deshadow-* (non-monotone on the OLD SYNTHETIC data; never
# re-validated on real ISTD, kept for provenance only per harness.py's SURFACES doc).
# Not included here -- re-anchoring a surface that's already honestly dropped for a
# documented reason is out of scope; dropped stays dropped.

METRIC_RE = re.compile(
    r"DESHADOW_METRICS\s+surface=(\S+)\s+setting=(\S+)\s+psnr=([\d.eE+-]+)\s+"
    r"psnr_gain=([\d.eE+-]+)\s+shadow_psnr=([\d.eE+-]+)\s+ssim=([\d.eE+-]+)\s+"
    r"mse=([\d.eE+-]+)\s+full_psnr=([\d.eE+-]+)")


def run_one(harness_surface, setting, sol_file, seed, iters, data_root):
    cmd = [sys.executable, os.path.join(HERE, "harness.py"),
           "--data-root", os.path.join(data_root, setting),
           "--surface", harness_surface,
           "--label", setting,
           "--solution", sol_file,
           "--iters", str(iters),
           "--seed", str(seed)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=1800)
    dt = time.time() - t0
    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        return None, dt, out
    m = METRIC_RE.search(out)
    if not m:
        return None, dt, out
    return {
        "psnr": float(m.group(3)),
        "psnr_gain": float(m.group(4)),
        "shadow_psnr": float(m.group(5)),
        "ssim": float(m.group(6)),
        "mse": float(m.group(7)),
        "full_psnr": float(m.group(8)),
    }, dt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds (overrides --seed)")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--only-task", default=None,
                    help="comma-separated task names to run (default: all)")
    ap.add_argument("--data-root", default="/data/image-deshadow")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        for ln in open(args.out):
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 4 and p[0] != "task":
                done.add((p[0], p[1], p[2], p[3]))
    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    f = open(args.out, "a")
    if write_header:
        f.write("task\tbaseline\tsetting\tseed\tpsnr\tpsnr_gain\tshadow_psnr\tssim\tmse\t"
                 "full_psnr\telapsed\n")
        f.flush()

    tasks = list(BASELINES)
    if args.only_task:
        tasks = [t.strip() for t in args.only_task.split(",")]

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed])

    for seed in seeds:
        for task in tasks:
            harness_surface, blist = BASELINES[task]
            for bname, bpath in blist:
                for setting in SETTINGS:
                    key = (task, bname, setting, str(seed))
                    if key in done:
                        print(f"[skip] {key}", flush=True)
                        continue
                    res, dt, out = run_one(harness_surface, setting, bpath, seed,
                                           args.iters, args.data_root)
                    if res is None:
                        print(f"[FAIL] {key} dt={dt:.1f}s\n{out[-2000:]}", flush=True)
                        continue
                    row = (f"{task}\t{bname}\t{setting}\t{seed}\t"
                           f"{res['psnr']:.6f}\t{res['psnr_gain']:.6f}\t"
                           f"{res['shadow_psnr']:.6f}\t{res['ssim']:.6f}\t"
                           f"{res['mse']:.6f}\t{res['full_psnr']:.6f}\t{dt:.1f}")
                    f.write(row + "\n"); f.flush()
                    print("[done] " + row, flush=True)
    f.close()
    print("ANCHOR_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
