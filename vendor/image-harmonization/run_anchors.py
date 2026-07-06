#!/usr/bin/env python3
"""Anchor driver for the image-harmonization repo (6 shipped cv-harmonization-* surfaces
x 3 severities), cross-seed (42/123) re-anchoring on REAL iHarmony4 data.

Runs every baseline of every in-scope surface across all 3 severities (mild/medium/
strong) by invoking harness.py as a subprocess (reuses the exact, already-reviewed
main() path). Resumable: skips (task, baseline, severity, seed) keys already present in
--out.

Usage (from vendor/image-harmonization/):
    python3 run_anchors.py --out /path/to/anchor_real.tsv --seeds 42,123 --iters 500 \
        --data-root /data/image-harmonization
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

SEVERITIES = ["mild", "medium", "strong"]

# task -> (harness --surface value, solution file, [(baseline_name, baseline_source_path), ...])
BASELINES = {
    "cv-harmonization-region-norm": ("network", "solution/network.py", [
        ("copy", os.path.join(BASE, "network_copy.py")),
        ("blind", os.path.join(BASE, "network_blind.py")),
        ("mask", os.path.join(BASE, "network_mask.py")),
    ]),
    "cv-harmonization-mask-conditioning": ("maskcond", "solution/maskcond.py", [
        ("none", os.path.join(BASE, "maskcond_none.py")),
        ("gated", os.path.join(BASE, "maskcond_gated.py")),
        ("concat", os.path.join(BASE, "maskcond_concat.py")),
    ]),
    "cv-harmonization-loss-region": ("loss", "solution/loss.py", [
        ("bg", os.path.join(BASE, "loss_bg.py")),
        ("global", os.path.join(BASE, "loss_global.py")),
        ("fg", os.path.join(BASE, "loss_fg.py")),
    ]),
    "cv-harmonization-feature-fusion": ("fusion", "solution/fusion.py", [
        ("noskip", os.path.join(BASE, "fusion_noskip.py")),
        ("skip", os.path.join(BASE, "fusion_skip.py")),
    ]),
    "cv-harmonization-activation": ("activation", "solution/activation.py", [
        ("identity", os.path.join(BASE, "activation_identity.py")),
        ("gelu", os.path.join(BASE, "activation_gelu.py")),
        ("relu", os.path.join(BASE, "activation_relu.py")),
    ]),
    "cv-harmonization-input-norm": ("inputnorm", "solution/inputnorm.py", [
        ("bg_whiten", os.path.join(BASE, "inputnorm_bg_whiten.py")),
        ("none", os.path.join(BASE, "inputnorm_none.py")),
    ]),
}

METRIC_RE = re.compile(
    r"HARMONY_METRICS\s+surface=(\S+)\s+setting=(\S+)\s+fg_psnr=([\d.eE+-]+)\s+"
    r"fg_psnr_gain=([\d.eE+-]+)\s+comp_fg_psnr=([\d.eE+-]+)\s+fg_mse=([\d.eE+-]+)\s+"
    r"fg_ssim=([\d.eE+-]+)")


def run_one(harness_surface, severity, sol_file, seed, iters, data_root):
    cmd = [sys.executable, os.path.join(HERE, "harness.py"),
           "--data-root", data_root,
           "--surface", harness_surface,
           "--severity", severity,
           "--label", severity,
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
        "fg_psnr": float(m.group(3)),
        "fg_psnr_gain": float(m.group(4)),
        "comp_fg_psnr": float(m.group(5)),
        "fg_mse": float(m.group(6)),
        "fg_ssim": float(m.group(7)),
    }, dt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds (overrides --seed)")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--only-task", default=None,
                    help="comma-separated task names to run (default: all)")
    ap.add_argument("--data-root", default="/data/image-harmonization")
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
        f.write("task\tbaseline\tseverity\tseed\tfg_psnr\tfg_psnr_gain\tcomp_fg_psnr\t"
                 "fg_mse\tfg_ssim\telapsed\n")
        f.flush()

    tasks = list(BASELINES)
    if args.only_task:
        tasks = [t.strip() for t in args.only_task.split(",")]

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed])

    for seed in seeds:
        for task in tasks:
            harness_surface, sol_rel, blist = BASELINES[task]
            for bname, bpath in blist:
                for severity in SEVERITIES:
                    key = (task, bname, severity, str(seed))
                    if key in done:
                        print(f"[skip] {key}", flush=True)
                        continue
                    res, dt, out = run_one(harness_surface, severity, bpath, seed,
                                           args.iters, args.data_root)
                    if res is None:
                        print(f"[FAIL] {key} dt={dt:.1f}s\n{out[-2000:]}", flush=True)
                        continue
                    row = (f"{task}\t{bname}\t{severity}\t{seed}\t"
                           f"{res['fg_psnr']:.6f}\t{res['fg_psnr_gain']:.6f}\t"
                           f"{res['comp_fg_psnr']:.6f}\t{res['fg_mse']:.6f}\t"
                           f"{res['fg_ssim']:.6f}\t{dt:.1f}")
                    f.write(row + "\n"); f.flush()
                    print("[done] " + row, flush=True)
    f.close()
    print("HARMONY_ANCHOR_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
