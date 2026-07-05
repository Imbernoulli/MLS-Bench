#!/usr/bin/env python3
"""Anchor driver for the crowd-counting repo (11 cv-count-* surfaces x 3 scenes),
post real-ShanghaiTech-data-swap re-anchoring.

Runs every baseline of every surface across all 3 (or for upsample/multiscale: medium/
dense/superdense) scenes by invoking harness.py as a subprocess (reuses the exact,
already-reviewed main() path). Resumable: skips (surface, baseline, scene, seed) keys
already present in --out.

Usage (from vendor/crowd-counting/):
    python3 run_anchors.py --out /path/to/anchor_real.tsv --iters 450 \
        --data-root /data/crowd-counting
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

DEFAULT_SCENES = ["medium", "middense", "dense"]
TASK_SCENES = {
    "cv-count-upsample": ["medium", "dense", "superdense"],
    "cv-count-multiscale": ["medium", "dense", "superdense"],
}

# task -> (harness --surface value, [(baseline_name, baseline_source_path), ...])
BASELINES = {
    "cv-count-formulation": ("head", [
        ("scalar", os.path.join(BASE, "head_scalar.py")),
        ("density", os.path.join(BASE, "head_density.py")),
    ]),
    "cv-count-normalization": ("norm", [
        ("softmax", os.path.join(BASE, "norm_softmax.py")),
        ("free", os.path.join(BASE, "norm_free.py")),
    ]),
    "cv-count-architecture": ("arch", [
        ("plain", os.path.join(BASE, "arch_plain.py")),
        ("mcnn", os.path.join(BASE, "arch_mcnn.py")),
        ("csrnet", os.path.join(BASE, "arch_csrnet.py")),
    ]),
    "cv-count-loss": ("loss", [
        ("mse", os.path.join(BASE, "loss_mse.py")),
        ("count", os.path.join(BASE, "loss_count.py")),
    ]),
    "cv-count-kernel": ("sigma", [
        ("fixed", os.path.join(BASE, "sigma_fixed.py")),
        ("adaptive", os.path.join(BASE, "sigma_adaptive.py")),
    ]),
    "cv-count-dilation": ("dilation", [
        ("pooled", os.path.join(BASE, "dilation_pooled.py")),
        ("dilated", os.path.join(BASE, "dilation_dilated.py")),
    ]),
    "cv-count-upsample": ("upsample", [
        ("none", os.path.join(BASE, "upsample_none.py")),
        ("learned", os.path.join(BASE, "upsample_learned.py")),
    ]),
    "cv-count-attention": ("attention", [
        ("none", os.path.join(BASE, "attention_none.py")),
        ("spatial", os.path.join(BASE, "attention_spatial.py")),
    ]),
    "cv-count-multiscale": ("multiscale", [
        ("single", os.path.join(BASE, "multiscale_single.py")),
        ("context", os.path.join(BASE, "multiscale_context.py")),
    ]),
    "cv-count-batchnorm": ("batchnorm", [
        ("none", os.path.join(BASE, "batchnorm_none.py")),
        ("bn", os.path.join(BASE, "batchnorm_bn.py")),
    ]),
    "cv-count-depth": ("depth", [
        ("shallow", os.path.join(BASE, "depth_shallow.py")),
        ("deep", os.path.join(BASE, "depth_deep.py")),
    ]),
}

METRIC_RE = re.compile(
    r"COUNT_METRICS\s+surface=(\S+)\s+setting=(\S+)\s+mae=([\d.eE+-]+)\s+"
    r"rmse=([\d.eE+-]+)\s+nae=([\d.eE+-]+)\s+gt_mean=([\d.eE+-]+)\s+pred_mean=([\d.eE+-]+)")


def run_one(harness_surface, scene, sol_file, seed, iters, data_root):
    cmd = [sys.executable, os.path.join(HERE, "harness.py"),
           "--data-root", os.path.join(data_root, scene),
           "--surface", harness_surface,
           "--label", scene,
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
        "mae": float(m.group(3)),
        "rmse": float(m.group(4)),
        "nae": float(m.group(5)),
        "gt_mean": float(m.group(6)),
        "pred_mean": float(m.group(7)),
    }, dt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds (overrides --seed)")
    ap.add_argument("--iters", type=int, default=450)
    ap.add_argument("--only-task", default=None,
                    help="comma-separated task names to run (default: all)")
    ap.add_argument("--data-root", default="/data/crowd-counting")
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
        f.write("task\tbaseline\tscene\tseed\tmae\trmse\tnae\tgt_mean\tpred_mean\telapsed\n")
        f.flush()

    tasks = list(BASELINES)
    if args.only_task:
        tasks = [t.strip() for t in args.only_task.split(",")]

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed])

    for seed in seeds:
        for task in tasks:
            harness_surface, blist = BASELINES[task]
            scenes = TASK_SCENES.get(task, DEFAULT_SCENES)
            for bname, bpath in blist:
                for scene in scenes:
                    key = (task, bname, scene, str(seed))
                    if key in done:
                        print(f"[skip] {key}", flush=True)
                        continue
                    res, dt, out = run_one(harness_surface, scene, bpath, seed,
                                           args.iters, args.data_root)
                    if res is None:
                        print(f"[FAIL] {key} dt={dt:.1f}s\n{out[-2000:]}", flush=True)
                        continue
                    row = (f"{task}\t{bname}\t{scene}\t{seed}\t"
                           f"{res['mae']:.6f}\t{res['rmse']:.6f}\t"
                           f"{res['nae']:.6f}\t{res['gt_mean']:.6f}\t"
                           f"{res['pred_mean']:.6f}\t{dt:.1f}")
                    f.write(row + "\n"); f.flush()
                    print("[done] " + row, flush=True)
    f.close()
    print("ANCHOR_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
