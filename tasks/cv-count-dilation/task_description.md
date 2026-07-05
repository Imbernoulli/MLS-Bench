# Crowd/Object Counting: DILATION / Receptive Field (CSRNet's core idea)

## Research Question
After a fixed VGG-lite stem, how should the back-end enlarge context? A **pooled** block downsamples and uses plain convs — small receptive field, lost resolution, so dense scenes are under-counted. A **dilated** block (3×3 convs at dilation rate 2) enlarges the receptive field **without** reducing resolution, aggregating large-scale context while keeping a dense density map — the founding CSRNet result (Li et al., CVPR 2018). **Design the back-end block**.

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
Modify `build_backbone_block` in `crowd-counting/solution/dilation.py` (see the file for the exact
signature and a worked strong example):

```python
def build_backbone_block(cin):  # -> nn.Module, .out_channels
    ...
```

The default is a pooled small-RF block; switch to dilated to win. A malformed / crashing surface falls back to the harness default.

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
