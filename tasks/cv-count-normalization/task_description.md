# Crowd/Object Counting: Density-Head Spatial Aggregation (Free Field vs Normalized × Scalar)

## Research Question
A density-map counter outputs a per-pixel density whose **integral is the count**. But
how should the head produce that map? A tempting design factorises **where** objects
are from **how many** there are: predict a spatial **softmax** (a distribution that
sums to 1 — pure location) and multiply it by a single learned **count scalar** (the
total). This looks elegant, but it **bottlenecks the entire count through one scalar**:
trained on sparsely-populated images, that scalar saturates near the training mean and
**cannot scale up** to denser scenes — so on higher-count test images it under-counts,
exactly like a direct count regressor. A **free non-negative density field** (a
per-pixel softplus with **unbounded** integrated mass) has no such bottleneck: more
objects simply produce more integrated density, so it **extrapolates** to unseen crowd
sizes. **Choose the spatial-aggregation** that recovers counting accuracy.

The fixed harness handles the data, the frontend, the loss, the optimiser, the
iterations, the seed, and the evaluation; your contribution is the **density head's
spatial aggregation only** (both options return a non-negative `(B,h,w)` density map;
the harness counts by its integral).

## Background
The primary crowd-counting metric is the **counting MAE** = `mean |pred_count −
gt_count|` (RMSE secondary). As in the sibling formulation task, this is a
**count-extrapolation** test: **training counts are LOW (15–60)** and the held-out
**val counts are HIGHER (61–136)**. A softmax-normalized-×-scalar head learns a total
mass ≈ the training mean (~37) and scores near the constant-mean floor
(`CONST_MEAN_MAE ≈ 63`); a free density field integrates the correct larger mass and
scores a much lower MAE. A degenerate constant-mean predictor is hopeless on the
shifted val split by construction.

## Implementation Contract
Modify `build_density_head` in `crowd-counting/solution/norm.py` to return a
`torch.nn.Module` on the frontend features `(B, cin, h, w)` that outputs a
**non-negative** density map `(B, h, w)`:

```python
def build_density_head(cin):
    import torch.nn as nn, torch.nn.functional as F
    class Head(nn.Module):                          # FREE density field
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, 64, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(True),
                nn.Conv2d(32, 1, 1))
        def forward(self, feat):
            return F.softplus(self.net(feat)).squeeze(1)   # (B,h,w) density
    return Head()
```

- A malformed / crashing head falls back to the default free density head.

## Fixed Pipeline & Evaluation
- Data: three crowd-density scenes (`medium` / `middense` / `dense`), each 120 train
  (LOW counts) / 40 val (HIGHER counts) **REAL crowd photos** (128×128×3, ShanghaiTech
  Crowd Counting Dataset, Zhang et al. CVPR 2016 — real surveillance/street photos with
  every human head annotated by a single point; images are bucketed into each scene by
  their REAL annotated head count), exact GT counts. The three scenes are the **three
  validation settings**.
- Backbone: a **fixed VGG-lite frontend (stride 8)** + your density head, trained a few
  hundred steps with a fixed density loss. **Only the head's aggregation changes.**
- The score is the geometric mean over the three scenes (`dense` is hidden).
- Metric (lower is better): **counting MAE** on each scene's val split; RMSE and NAE
  recorded.
- The per-scene scoring midpoint sits between the free density field and the
  softmax-normalized / constant-mean floor: you score above 0.5 only by removing the
  single-scalar mass bottleneck.
