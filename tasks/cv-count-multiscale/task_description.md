# Crowd/Object Counting: MULTI-SCALE Context aggregation (CAN-style)

## Research Question
Objects span a wide range of scales, but a single-scale feature map matches only one receptive field, so off-scale objects are mis-counted. A **multi-scale context** module pools the features at several block sizes (2×2 / 4×4 / 8×8), upsamples each back, and fuses them (residually) with the base features — explicit multi-scale context, as in CAN (Liu et al., CVPR 2019) / spatial-pyramid pooling. **Design the context module**.

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
Modify `build_context` in `crowd-counting/solution/multiscale.py` (see the file for the exact
signature and a worked strong example):

```python
def build_context(cin):  # -> nn.Module (features -> context-enriched)
    ...
```

The default is identity (single-scale); add a multi-scale context module to win. A malformed / crashing surface falls back to the harness default.

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
