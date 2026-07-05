# Crowd/Object Counting: The GT-Density KERNEL (fixed vs geometry-adaptive)

## Research Question
The ground-truth density map is rendered by blurring each annotated point with a Gaussian. An **oversized fixed** sigma over-smooths dense scenes — big kernels overlap and smear neighbouring objects together, so the target cannot resolve individuals. A **geometry-adaptive k-NN** kernel sets each point's sigma from the distance to its nearest neighbours (sigma = beta·mean_kNN_dist): small where the crowd is dense, larger where it is sparse — the MCNN / CSRNet kernel (beta≈0.3). **Design the kernel** that resolves crowded scenes.

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
Modify `gt_sigma` in `crowd-counting/solution/sigma.py` (see the file for the exact
signature and a worked strong example):

```python
def gt_sigma(points, H, W):  # -> scalar or per-point sigma (px)
    ...
```

The default is an oversized fixed sigma; switch to adaptive k-NN to win. A malformed / crashing surface falls back to the harness default.

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
