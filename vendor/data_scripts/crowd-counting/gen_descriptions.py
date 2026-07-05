#!/usr/bin/env python3
"""Generate task_description.md for each cv-count-* RQ task."""
from pathlib import Path

TASKS = Path("/home/lvbohan/projects/MLS-Bench/tasks")

# task -> (title, RQ paragraph, contract hook signature + good example, weak default note)
DESCS = {
    "cv-count-loss": (
        "Crowd/Object Counting: The Density-Map Training LOSS",
        "How should the density map be supervised? A plain per-pixel MSE is dominated by "
        "the huge ZERO background of a crowded density map, so the network under-shoots "
        "the foreground mass and **under-counts**. A better loss **up-weights the "
        "foreground pixels** (where the GT density is non-zero) and adds an explicit "
        "**count-consistency** term on the integrated mass — directly targeting what the "
        "counting metric measures. This mirrors count-aware / DM-Count-style losses "
        "(Wang et al., NeurIPS 2020). **Design the loss** that recovers counting accuracy.",
        "density_loss", "def density_loss(pred, gt):  # pred, gt: (B,h,w) density maps",
        "The default is plain pixel MSE; add foreground weighting + a count term to win."),
    "cv-count-kernel": (
        "Crowd/Object Counting: The GT-Density KERNEL (fixed vs geometry-adaptive)",
        "The ground-truth density map is rendered by blurring each annotated point with a "
        "Gaussian. An **oversized fixed** sigma over-smooths dense scenes — big kernels "
        "overlap and smear neighbouring objects together, so the target cannot resolve "
        "individuals. A **geometry-adaptive k-NN** kernel sets each point's sigma from the "
        "distance to its nearest neighbours (sigma = beta·mean_kNN_dist): small where the "
        "crowd is dense, larger where it is sparse — the MCNN / CSRNet kernel (beta≈0.3). "
        "**Design the kernel** that resolves crowded scenes.",
        "gt_sigma", "def gt_sigma(points, H, W):  # -> scalar or per-point sigma (px)",
        "The default is an oversized fixed sigma; switch to adaptive k-NN to win."),
    "cv-count-dilation": (
        "Crowd/Object Counting: DILATION / Receptive Field (CSRNet's core idea)",
        "After a fixed VGG-lite stem, how should the back-end enlarge context? A **pooled** "
        "block downsamples and uses plain convs — small receptive field, lost resolution, "
        "so dense scenes are under-counted. A **dilated** block (3×3 convs at dilation rate "
        "2) enlarges the receptive field **without** reducing resolution, aggregating "
        "large-scale context while keeping a dense density map — the founding CSRNet result "
        "(Li et al., CVPR 2018). **Design the back-end block**.",
        "build_backbone_block", "def build_backbone_block(cin):  # -> nn.Module, .out_channels",
        "The default is a pooled small-RF block; switch to dilated to win."),
    "cv-count-upsample": (
        "Crowd/Object Counting: OUTPUT STRIDE / Upsampling decoder",
        "The density map is produced at stride 8 (a 16×16 grid for a 128px image). In dense "
        "scenes many objects fall inside a single coarse cell and cannot be separated, so "
        "the count saturates. A learned **upsampling decoder** (transposed conv to a finer "
        "stride + refinement) produces a higher-resolution, higher-quality density map so "
        "nearby objects occupy separate cells — the lever behind TEDnet / SANet decoders. "
        "The count is resolution-invariant. **Design the decoder**.",
        "build_decoder", "def build_decoder(cin):  # -> nn.Module (features -> refined features)",
        "The default is identity (coarse stride-8); add an upsampling decoder to win."),
    "cv-count-attention": (
        "Crowd/Object Counting: Spatial ATTENTION (clutter suppression)",
        "The scenes contain unannotated distractor **clutter** that looks like objects but "
        "is not counted. Without attention, the counter spends density mass on this clutter "
        "and mis-counts. A learned **spatial-attention gate** predicts a per-pixel weight in "
        "[0,1] and multiplies the features by it, suppressing clutter and focusing on real "
        "objects — the idea behind SCAR / ADCrowdNet / SFANet. **Design the attention "
        "module**.",
        "build_attention", "def build_attention(cin):  # -> nn.Module (features -> gated features)",
        "The default is identity (no attention); add a spatial-attention gate to win."),
    "cv-count-multiscale": (
        "Crowd/Object Counting: MULTI-SCALE Context aggregation (CAN-style)",
        "Objects span a wide range of scales, but a single-scale feature map matches only "
        "one receptive field, so off-scale objects are mis-counted. A **multi-scale context** "
        "module pools the features at several block sizes (2×2 / 4×4 / 8×8), upsamples each "
        "back, and fuses them (residually) with the base features — explicit multi-scale "
        "context, as in CAN (Liu et al., CVPR 2019) / spatial-pyramid pooling. **Design the "
        "context module**.",
        "build_context", "def build_context(cin):  # -> nn.Module (features -> context-enriched)",
        "The default is identity (single-scale); add a multi-scale context module to win."),
    "cv-count-batchnorm": (
        "Crowd/Object Counting: Backbone NORMALIZATION (none vs BatchNorm)",
        "With a batch of crowded images spanning a wide count range, the backbone's "
        "activation statistics drift, so optimisation is less stable and the density "
        "calibration is noisier at a fixed step budget. Adding **BatchNorm** after each conv "
        "stabilises the statistics, converges better and calibrates the density — the "
        "CSRNet-with-BN (VGG16-BN) recipe for batched crowd-counting training. **Design the "
        "backbone normalization**. (Note: BN helps at moderate density but can destabilise "
        "the most extreme crowds — the score aggregates over three densities.)",
        "build_backbone", "def build_backbone():  # -> nn.Module (image -> features), .out_channels",
        "The default is a plain (no-norm) backbone; add BatchNorm to win overall."),
    "cv-count-depth": (
        "Crowd/Object Counting: Backbone DEPTH (shallow vs deep)",
        "A **shallow** feature extractor (one conv per pooling stage) has too little capacity "
        "to disentangle heavily crowded, occluded scenes and under-counts. A **deeper** "
        "backbone (two convs per stage + a post-pool refinement block) has the capacity to "
        "resolve dense crowds — depth is the standard lever behind VGG-16-based counters "
        "(CSRNet uses a 13-layer VGG front-end). **Design the backbone depth**.",
        "build_deep_backbone", "def build_deep_backbone():  # -> nn.Module (image -> features), .out_channels",
        "The default is a shallow backbone; deepen it to win on the crowded scenes."),
}

STUB = {
    "cv-count-loss": "loss.py", "cv-count-kernel": "sigma.py", "cv-count-dilation": "dilation.py",
    "cv-count-upsample": "upsample.py", "cv-count-attention": "attention.py",
    "cv-count-multiscale": "multiscale.py", "cv-count-batchnorm": "batchnorm.py",
    "cv-count-depth": "depth.py",
}


def body(task):
    title, rq, hook, sig, weak = DESCS[task]
    stub = STUB[task]
    return f"""# {title}

## Research Question
{rq}

The fixed harness handles the data, the rest of the network, the optimiser, the
iterations, the seed and the evaluation; your contribution is **this one surface only**,
so any change in counting MAE is attributable to your design.

## Background
The primary crowd-counting metric is the **counting MAE** = `mean |pred_count −
gt_count|` (RMSE secondary, both lower-is-better; standard MCNN/CSRNet convention). The
task is scored on **three crowd-density scenes** — `medium`, `middense`, `dense` — each a
count-**extrapolation** test (training counts LOW, held-out val counts HIGHER), so a
degenerate constant-mean predictor is hopeless by construction. The score is the
geometric mean over the three scenes.

## Implementation Contract
Modify `{hook}` in `crowd-counting/solution/{stub}` (see the file for the exact
signature and a worked strong example):

```python
{sig}
    ...
```

{weak} A malformed / crashing surface falls back to the harness default.

## Fixed Pipeline & Evaluation
- Data: three crowd-density scenes (`medium` / `middense` / `dense`), each 120 train
  (LOW counts) / 40 val (HIGHER counts) **REAL crowd photos** (128×128×3, ShanghaiTech
  Crowd Counting Dataset, Zhang et al. CVPR 2016 — real surveillance/street photos with
  every human head annotated by a single point; images are bucketed into each scene by
  their REAL annotated head count), exact GT counts. The three scenes are the **three
  validation settings**.
- Training: a few hundred steps with a fixed density loss (except where the loss IS the
  surface). **Only your surface changes.**
- Metric (lower is better): **counting MAE** on each scene's val split; RMSE and NAE
  recorded. The score is the geometric mean over the three scenes; the per-scene sigmoid
  midpoint sits between the strong and weak baselines, so you score above 0.5 only by a
  genuine design improvement.
"""


if __name__ == "__main__":
    for task in DESCS:
        (TASKS / task / "task_description.md").write_text(body(task))
        print("wrote", task)
