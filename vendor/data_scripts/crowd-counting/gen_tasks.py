#!/usr/bin/env python3
"""Self-contained generator for all cv-count-* RQ task scaffolds.

For each RQ: writes config.json, scripts/count_{sparse,medium,dense}.sh, parser.py, and
edits/*.edit.py (weak + strong). The edit _CONTENT is the hook function BODY extracted
from the corresponding vendor baseline file, so applying the edit reproduces that
baseline exactly. score_spec.py is written separately after anchors are measured.
"""
import json
import shutil
from pathlib import Path

ROOT = Path("/home/lvbohan/projects/MLS-Bench")
TASKS = ROOT / "tasks"
VEND = ROOT / "vendor" / "crowd-counting"
SOL = "crowd-counting/solution"
PARSER_SRC = TASKS / "cv-count-formulation" / "parser.py"
DEFAULT_SCENES = ["medium", "middense", "dense"]
# Output-stride & multi-scale gains are clearest at the extreme densities.
TASK_SCENES = {
    "cv-count-upsample": ["medium", "dense", "superdense"],
    "cv-count-multiscale": ["medium", "dense", "superdense"],
}

# Each RQ: task, surface, solution stub file, hook name, edit [start,end],
# and baselines: name -> (role, baseline_file, doc_first_line)
RQS = {
    "cv-count-loss": dict(
        surface="loss", stub="loss.py", hook="density_loss", edit=[31, 34],
        baselines={
            "count": ("GOOD", "loss_count.py",
                      "Good baseline for cv-count-loss: pixel MSE + COUNT-CONSISTENCY. "
                      "Directly supervises the integrated count -> lower MAE. "
                      "Ref: vendor/crowd-counting/baselines/loss_count.py"),
            "mse": ("WEAK", "loss_mse.py",
                    "Weak baseline for cv-count-loss: plain pixel MSE only. No count "
                    "supervision -> count drifts -> higher MAE. "
                    "Ref: vendor/crowd-counting/baselines/loss_mse.py"),
        }),
    "cv-count-kernel": dict(
        surface="sigma", stub="sigma.py", hook="gt_sigma", edit=[39, 40],
        baselines={
            "adaptive": ("GOOD", "sigma_adaptive.py",
                         "Good baseline for cv-count-kernel: geometry-ADAPTIVE k-NN "
                         "Gaussian kernel (MCNN/CSRNet, beta~0.3). Resolves dense scenes "
                         "-> lower MAE. Ref: vendor/crowd-counting/baselines/sigma_adaptive.py"),
            "fixed": ("WEAK", "sigma_fixed.py",
                      "Weak baseline for cv-count-kernel: OVERSIZED fixed Gaussian "
                      "kernel. Smears dense scenes -> higher MAE. "
                      "Ref: vendor/crowd-counting/baselines/sigma_fixed.py"),
        }),
    "cv-count-dilation": dict(
        surface="dilation", stub="dilation.py", hook="build_backbone_block", edit=[39, 53],
        baselines={
            "dilated": ("GOOD", "dilation_dilated.py",
                        "Good baseline for cv-count-dilation: DILATED large-RF block "
                        "(CSRNet). Enlarges receptive field without losing resolution -> "
                        "lower MAE. Ref: vendor/crowd-counting/baselines/dilation_dilated.py"),
            "pooled": ("WEAK", "dilation_pooled.py",
                       "Weak baseline for cv-count-dilation: POOLED small-RF block. "
                       "Loses resolution / context -> higher MAE. "
                       "Ref: vendor/crowd-counting/baselines/dilation_pooled.py"),
        }),
    "cv-count-upsample": dict(
        surface="upsample", stub="upsample.py", hook="build_decoder", edit=[36, 38],
        baselines={
            "learned": ("GOOD", "upsample_learned.py",
                        "Good baseline for cv-count-upsample: learned UPSAMPLING decoder "
                        "(finer output). Separates nearby objects in dense scenes -> "
                        "lower MAE. Ref: vendor/crowd-counting/baselines/upsample_learned.py"),
            "none": ("WEAK", "upsample_none.py",
                     "Weak baseline for cv-count-upsample: NO decoder (coarse stride-8). "
                     "Objects in one cell can't be separated -> higher MAE. "
                     "Ref: vendor/crowd-counting/baselines/upsample_none.py"),
        }),
    "cv-count-attention": dict(
        surface="attention", stub="attention.py", hook="build_attention", edit=[36, 37],
        baselines={
            "spatial": ("GOOD", "attention_spatial.py",
                        "Good baseline for cv-count-attention: spatial ATTENTION gate. "
                        "Suppresses distractor clutter -> lower MAE. "
                        "Ref: vendor/crowd-counting/baselines/attention_spatial.py"),
            "none": ("WEAK", "attention_none.py",
                     "Weak baseline for cv-count-attention: NO attention. Clutter not "
                     "suppressed -> higher MAE. "
                     "Ref: vendor/crowd-counting/baselines/attention_none.py"),
        }),
    "cv-count-multiscale": dict(
        surface="multiscale", stub="multiscale.py", hook="build_context", edit=[41, 43],
        baselines={
            "context": ("GOOD", "multiscale_context.py",
                        "Good baseline for cv-count-multiscale: MULTI-SCALE context "
                        "aggregation (CAN-style pyramid). Handles scale variation -> "
                        "lower MAE. Ref: vendor/crowd-counting/baselines/multiscale_context.py"),
            "single": ("WEAK", "multiscale_single.py",
                       "Weak baseline for cv-count-multiscale: SINGLE-scale context. "
                       "Mis-counts off-scale objects -> higher MAE. "
                       "Ref: vendor/crowd-counting/baselines/multiscale_single.py"),
        }),
    "cv-count-batchnorm": dict(
        surface="batchnorm", stub="batchnorm.py", hook="build_backbone", edit=[43, 60],
        baselines={
            "bn": ("GOOD", "batchnorm_bn.py",
                   "Good baseline for cv-count-batchnorm: BatchNorm backbone. Stabler "
                   "optimisation across the count range -> lower MAE. "
                   "Ref: vendor/crowd-counting/baselines/batchnorm_bn.py"),
            "none": ("WEAK", "batchnorm_none.py",
                     "Weak baseline for cv-count-batchnorm: NO normalization. Activation "
                     "stats drift -> higher MAE. "
                     "Ref: vendor/crowd-counting/baselines/batchnorm_none.py"),
        }),
    "cv-count-depth": dict(
        surface="depth", stub="depth.py", hook="build_deep_backbone", edit=[41, 55],
        baselines={
            "deep": ("GOOD", "depth_deep.py",
                     "Good baseline for cv-count-depth: DEEP backbone. More capacity to "
                     "resolve heavily crowded scenes -> lower MAE. "
                     "Ref: vendor/crowd-counting/baselines/depth_deep.py"),
            "shallow": ("WEAK", "depth_shallow.py",
                        "Weak baseline for cv-count-depth: SHALLOW backbone. Too little "
                        "capacity for dense crowds -> higher MAE. "
                        "Ref: vendor/crowd-counting/baselines/depth_shallow.py"),
        }),
}


def extract_body(baseline_path: Path, hook: str) -> str:
    """Extract the hook function BODY (lines after 'def hook(...):' to the function end,
    dedented by 0 -- kept at the baseline's indentation, i.e. body lines start at 4
    spaces). We take everything after the def line to the end of file (baselines have
    only the one function)."""
    lines = baseline_path.read_text().split("\n")
    start = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(f"def {hook}("):
            start = i + 1
            break
    assert start is not None, f"{hook} not found in {baseline_path}"
    body = lines[start:]
    # trim trailing blank lines
    while body and body[-1].strip() == "":
        body.pop()
    return "\n".join(body)


def script(task, surface, solution_file, scene):
    return f"""#!/bin/bash
# {task} ({scene} scene): train the agent surface on the {scene.upper()} crowd-density
# scene, score counting MAE on its higher-count val split.
set -e
export CUDA_VISIBLE_DEVICES=0
cd /workspace/crowd-counting
python harness.py \\
    --data-root ${{COUNT_DATA_ROOT:-/data/crowd-counting}}/{scene} \\
    --surface {surface} \\
    --label {scene} \\
    --solution {solution_file} \\
    --iters ${{COUNT_ITERS:-450}} \\
    --seed ${{SEED:-42}}
"""


def config(surface, solution_file, edit, baselines, scenes):
    tcs = []
    for i, scene in enumerate(scenes):
        tc = {"cmd": f"scripts/count_{scene}.sh", "label": scene, "group": 1,
              "compute": 1.0, "time": "0:30:00", "mem": 16, "package": "crowd-counting"}
        if i == len(scenes) - 1:   # last scene is the hidden held-out setting
            tc["hidden"] = True
        tcs.append(tc)
    cfg = {
        "allow_create": False, "rigorous_codebase": True, "seeds": [42],
        "test_cmds": tcs,
        "baselines": {n: {"edit_ops": f"edits/{n}.edit.py"} for n in baselines},
        "files": [{"filename": solution_file, "read": [{"start": -1, "end": -1}],
                   "edit": [{"start": edit[0], "end": edit[1]}]}],
    }
    return json.dumps(cfg, indent=2) + "\n"


def edit_py(solution_file, edit, body, doc):
    return (f'"""{doc}"""\n\n'
            f'_FILE = "{solution_file}"\n\n'
            f'_CONTENT = {body!r}\n\n'
            f'OPS = [\n'
            f'    {{"op": "replace", "file": _FILE, "start_line": {edit[0]}, '
            f'"end_line": {edit[1]}, "content": _CONTENT}},\n'
            f']\n')


def main():
    for task, spec in RQS.items():
        surface = spec["surface"]
        solution_file = f"{SOL}/{spec['stub']}"
        edit = spec["edit"]
        scenes = TASK_SCENES.get(task, DEFAULT_SCENES)
        d = TASKS / task
        (d / "scripts").mkdir(parents=True, exist_ok=True)
        (d / "edits").mkdir(parents=True, exist_ok=True)
        # write scripts for this task's scenes; remove any stale scene scripts
        for old in (d / "scripts").glob("count_*.sh"):
            old.unlink()
        for scene in scenes:
            (d / "scripts" / f"count_{scene}.sh").write_text(script(task, surface, solution_file, scene))
        (d / "config.json").write_text(config(surface, solution_file, edit, spec["baselines"], scenes))
        shutil.copy(PARSER_SRC, d / "parser.py")
        for name, (role, bfile, doc) in spec["baselines"].items():
            body = extract_body(VEND / "baselines" / bfile, spec["hook"])
            (d / "edits" / f"{name}.edit.py").write_text(edit_py(solution_file, edit, body, doc))
        print(f"built {task}: surface={surface} scenes={scenes} baselines={list(spec['baselines'])}")


if __name__ == "__main__":
    main()
