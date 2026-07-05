# Crowd/Object Counting: The Counting Formulation (Density Map vs Direct Regression)

## Research Question
How should a network predict the number of objects in an image? The founding result
of modern crowd counting (MCNN, CSRNet) is that you should **NOT** regress the count
directly. Instead you regress a per-pixel **density map** and take the count as its
**spatial integral**. A **direct global regressor** (global-average-pool the features
and map them through an MLP to a single scalar count) throws away the spatial
structure: it has only one supervision signal per image, no spatial inductive bias,
and it **memorises the training count distribution** — so when the test scenes are
denser than training, it regresses toward the (low) training mean and badly
**under-counts**. A **density map** predicts a translation-equivariant *local* density
and sums ~h·w densely-supervised local votes, so more objects produce proportionally
more integral and it **generalises** to unseen crowd sizes. **Choose the formulation**
that recovers counting accuracy.

The fixed harness handles the data, the frontend, the loss, the optimiser, the
iterations, the seed, and the evaluation; your contribution is the **count head only**.

## Background
The primary crowd-counting metric is the **counting MAE** = `mean |pred_count −
gt_count|` (RMSE secondary). Here the task is deliberately a **count-extrapolation**
test: **training images have LOW counts (15–60)** while the **held-out val images have
HIGHER counts (61–136)**. A direct scalar regressor learns to output ≈ the training
mean (~37) and so scores an MAE around the constant-mean floor (`CONST_MEAN_MAE ≈
63`); a density-map head, being fully convolutional and translation-equivariant,
integrates the correct larger mass and scores a much lower MAE. A degenerate
constant-mean predictor and an all-zero (collapsed) predictor are both hopeless on the
shifted val split by construction.

## Implementation Contract
Modify `build_count_head` in `crowd-counting/solution/head.py` to return a
`torch.nn.Module` on the frontend features `(B, cin, h, w)`. The harness scores
whichever the head returns:
- a **non-negative density map** `(B, h, w)` → the count is its spatial integral;
- a per-image **scalar** `(B,)` → used directly as the count.

```python
def build_count_head(cin):
    import torch.nn as nn, torch.nn.functional as F
    class Head(nn.Module):                       # DENSITY-MAP formulation
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

- A malformed / crashing head falls back to the default density head.

## Fixed Pipeline & Evaluation
- Data: three crowd-density scenes (`medium` / `middense` / `dense`), each 120 train
  (LOW counts) / 40 val (HIGHER counts) **REAL crowd photos** (128×128×3, ShanghaiTech
  Crowd Counting Dataset, Zhang et al. CVPR 2016 — real surveillance/street photos with
  every human head annotated by a single point; images are bucketed into each scene by
  their REAL annotated head count), with **exact** GT counts. The three scenes are the
  **three validation settings**.
- Backbone: a **fixed VGG-lite frontend (stride 8)** + your head, trained a few hundred
  steps with a fixed density loss. **Only the head changes.**
- The score is the geometric mean over the three scenes (`dense` is hidden).
- Metric (lower is better): **counting MAE** on each scene's val split; RMSE and NAE
  recorded.
- The per-scene scoring midpoint sits between the density head and the direct regressor /
  constant-mean floor: you score above 0.5 only by adopting the density-map formulation.
