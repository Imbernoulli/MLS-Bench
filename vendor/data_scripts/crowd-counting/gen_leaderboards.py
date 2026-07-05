#!/usr/bin/env python3
"""Generate leaderboard.csv for every cv-count-* task from the measured anchor MAEs.

Columns: timestamp,model,is_final,seed,mae_<scene>,rmse_<scene>,nae_<scene>,...,
elapsed_<scene>... . One row per baseline (weak/good, and plain/mcnn/csrnet for arch).
Only mae is measured here (rmse/nae/elapsed are placeholders the scorer ignores for
ranking; mae is the scored column).
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/lvbohan/projects/MLS-Bench")
TASKS = ROOT / "tasks"
DEFAULT_SCENES = ["medium", "middense", "dense"]
TASK_SCENES = {"cv-count-upsample": ["medium", "dense", "superdense"],
               "cv-count-multiscale": ["medium", "dense", "superdense"]}
# task -> (surface, [(baseline_name, role), ...])
TASKS_BASELINES = {
    "cv-count-formulation": ("head", [("scalar", "WEAK"), ("density", "GOOD")]),
    "cv-count-normalization": ("norm", [("softmax", "WEAK"), ("free", "GOOD")]),
    "cv-count-architecture": ("arch", [("plain", "WEAK"), ("mcnn", "MID"), ("csrnet", "SOTA")]),
    "cv-count-loss": ("loss", [("mse", "WEAK"), ("count", "GOOD")]),
    "cv-count-kernel": ("sigma", [("fixed", "WEAK"), ("adaptive", "GOOD")]),
    "cv-count-dilation": ("dilation", [("pooled", "WEAK"), ("dilated", "GOOD")]),
    "cv-count-upsample": ("upsample", [("none", "WEAK"), ("learned", "GOOD")]),
    "cv-count-attention": ("attention", [("none", "WEAK"), ("spatial", "GOOD")]),
    "cv-count-multiscale": ("multiscale", [("single", "WEAK"), ("context", "GOOD")]),
    "cv-count-batchnorm": ("batchnorm", [("none", "WEAK"), ("bn", "GOOD")]),
    "cv-count-depth": ("depth", [("shallow", "WEAK"), ("deep", "GOOD")]),
}
TS = "2026-07-03T00:00:00+00:00"


def load(path):
    A = defaultdict(lambda: defaultdict(dict))
    pat = re.compile(r"scene=(\S+) surface=(\S+) role=(\S+) sol=(\S+) mae=([\d.]+)")
    for ln in open(path):
        m = pat.search(ln)
        if m:
            A[m.group(2)][m.group(1)][m.group(3)] = float(m.group(5))
    return A


def main():
    A = load(sys.argv[1])
    for task, (surf, bls) in TASKS_BASELINES.items():
        scenes = TASK_SCENES.get(task, DEFAULT_SCENES)
        cols = ["timestamp", "model", "is_final", "seed"]
        for sc in scenes:
            cols += [f"mae_{sc}", f"rmse_{sc}", f"nae_{sc}"]
        cols += [f"elapsed_{sc}" for sc in scenes]
        rows = [",".join(cols)]
        for name, role in bls:
            vals = [TS, f"baseline:{name}", "true", "42"]
            for sc in scenes:
                mae = A[surf][sc][role]
                vals += [f"{mae:.4f}", f"{mae*1.1:.4f}", f"{mae/100:.4f}"]
            vals += ["0"] * len(scenes)
            rows.append(",".join(vals))
        (TASKS / task / "leaderboard.csv").write_text("\n".join(rows) + "\n")
        print(f"leaderboard {task}: {len(bls)} rows, scenes={scenes}")


if __name__ == "__main__":
    main()
