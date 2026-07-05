#!/usr/bin/env python3
"""Write score_spec.py for every cv-count-* task from the measured anchor MAEs.

Reads the anchor log (lines: ANCHOR scene=<s> surface=<su> role=<r> sol=<f> mae=<m>) and,
for each task, sets the sigmoid midpoint per RQ to the geometric mean of the WEAK and
GOOD (or plain/SOTA for arch) MAEs averaged appropriately across the three scenes, so the
GOOD baseline scores well above 0.5 and the WEAK / degenerate below.

Each score_spec scores counting MAE (lower better) with a sigmoid, one term per scene
(sparse/medium/dense), gmean over the three scenes.
"""
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/lvbohan/projects/MLS-Bench")
TASKS = ROOT / "tasks"
DEFAULT_SCENES = ["medium", "middense", "dense"]

# task -> (surface, weak_role, good_role). Most RQs weak=WEAK good=GOOD; arch weak=plain
# (WEAK) good=csrnet(SOTA). Roster validated across each task's 3 crowd-density settings;
# columns & patch were DROPPED (non-monotone: their "good" is genuinely worse at scale).
TASK_SURFACE = {
    "cv-count-formulation": ("head", "WEAK", "GOOD"),
    "cv-count-normalization": ("norm", "WEAK", "GOOD"),
    "cv-count-architecture": ("arch", "WEAK", "SOTA"),
    "cv-count-loss": ("loss", "WEAK", "GOOD"),
    "cv-count-kernel": ("sigma", "WEAK", "GOOD"),
    "cv-count-dilation": ("dilation", "WEAK", "GOOD"),
    "cv-count-upsample": ("upsample", "WEAK", "GOOD"),
    "cv-count-attention": ("attention", "WEAK", "GOOD"),
    "cv-count-multiscale": ("multiscale", "WEAK", "GOOD"),
    "cv-count-batchnorm": ("batchnorm", "WEAK", "GOOD"),
    "cv-count-depth": ("depth", "WEAK", "GOOD"),
}

# The output-stride & multi-scale gains are clearest at the EXTREME densities, so those
# two tasks use (medium, dense, superdense); everyone else uses (medium, middense, dense).
TASK_SCENES = {
    "cv-count-upsample": ["medium", "dense", "superdense"],
    "cv-count-multiscale": ["medium", "dense", "superdense"],
}


def load_anchors(path):
    # A[surface][scene][role] = mae
    A = defaultdict(lambda: defaultdict(dict))
    pat = re.compile(r"scene=(\S+) surface=(\S+) role=(\S+) sol=(\S+) mae=([\d.]+)")
    for ln in open(path):
        m = pat.search(ln)
        if m:
            A[m.group(2)][m.group(1)][m.group(3)] = float(m.group(5))
    return A


def spec_text(task, surface, scenes, midpoints, weak_maes, good_maes, degen_maes, scale):
    lines = [f'"""Score spec for {task}.\n',
             "Counting MAE (lower is better), scored on THREE crowd-density scenes",
             f"({' / '.join(scenes)}) as three validation settings; the score is the",
             "geometric mean over the scenes. Each scene uses a sigmoid on the counting MAE",
             "with a midpoint between the strong (good) and weak baselines, so the good",
             "baseline scores well above 0.5, the weak/degenerate below.\n",
             "MEASURED anchors (120 train / 40 val per scene, GPU torch 2.4, 450-step fit,"
             " seed 42):"]
    for sc in scenes:
        w = weak_maes.get(sc); g = good_maes.get(sc); d = degen_maes.get(sc)
        lines.append(f"  {sc:<10} weak mae={w:.2f}  good mae={g:.2f}"
                     + (f"  degenerate mae={d:.2f}" if d is not None else "")
                     + f"   (midpoint {midpoints[sc]:.1f})")
    lines.append('"""')
    body = ["from mlsbench.scoring.dsl import *", ""]
    for sc in scenes:
        body.append(f'term("mae_{sc}",')
        body.append(f'    col("mae_{sc}").lower().id()')
        body.append(f'    .sigmoid(ref=const({midpoints[sc]:.1f}), scale={scale:.1f}))')
        body.append("")
    for sc in scenes:
        body.append(f'setting("{sc}", weighted_mean(("mae_{sc}", 1.0)))')
    body.append("")
    body.append(f'task(gmean({", ".join(chr(34)+s+chr(34) for s in scenes)}))')
    return "\n".join(lines) + "\n" + "\n".join(body) + "\n"


def main():
    anchor_log = sys.argv[1]
    A = load_anchors(anchor_log)
    for task, (surface, weak_role, good_role) in TASK_SURFACE.items():
        scenes = TASK_SCENES.get(task, DEFAULT_SCENES)
        sc_data = A.get(surface, {})
        weak_maes, good_maes, degen_maes, midpoints = {}, {}, {}, {}
        ok = True
        for sc in scenes:
            d = sc_data.get(sc, {})
            w = d.get(weak_role); g = d.get(good_role)
            deg = d.get("DEGEN")
            if w is None or g is None:
                print(f"  [skip {task}/{sc}] missing weak/good ({weak_role}/{good_role}); have {list(d)}")
                ok = False
                continue
            weak_maes[sc] = w; good_maes[sc] = g
            if deg is not None:
                degen_maes[sc] = deg
            mid = math.sqrt(max(1e-3, g) * max(1e-3, w))
            midpoints[sc] = mid
        if not ok:
            print(f"!! {task}: incomplete anchors, NOT writing score_spec")
            continue
        spreads = [abs(weak_maes[sc] - good_maes[sc]) for sc in scenes]
        scale = max(4.0, sum(spreads) / len(spreads) / 4.0)
        txt = spec_text(task, surface, scenes, midpoints, weak_maes, good_maes, degen_maes, scale)
        (TASKS / task / "score_spec.py").write_text(txt)
        gm_good = math.exp(sum(math.log(good_maes[s]) for s in scenes) / 3)
        gm_weak = math.exp(sum(math.log(weak_maes[s]) for s in scenes) / 3)
        print(f"OK {task} [{'/'.join(scenes)}]: gmean(good)={gm_good:.1f} "
              f"gmean(weak)={gm_weak:.1f} mids={[round(midpoints[s],1) for s in scenes]} "
              f"scale={scale:.1f}")


if __name__ == "__main__":
    main()
