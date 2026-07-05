# Crowd/Object Counting: The Counting ARCHITECTURE (plain CNN vs multi-column vs dilated)

## Research Question
Given the density-map counting formulation (predict a per-pixel density, count = its
spatial integral), **what backbone architecture** best regresses that density? This is
the central question of modern crowd counting, and the field answered it with a clear
weak→strong→SOTA progression:

- **Plain single-column CNN** (weak): one filter size, a small receptive field. It
  cannot cover the wide range of object scales and crowding, and its small receptive
  field under-counts dense, occluded regions — the **worst** counting MAE. (The
  single-column ablation in MCNN scores ShanghaiTech Part A MAE ≈ 141.)
- **Multi-column CNN (MCNN)** (mid): three parallel columns with **different filter
  sizes** (9×9 / 7×7 / 5×5) fused by a 1×1 conv, absorbing scale variation → lower MAE.
  (MCNN, Zhang et al. CVPR 2016: Part A MAE **110.2** / Part B **26.4**.)
- **Dilated backbone (CSRNet)** (SOTA): a VGG-style stem (three poolings, stride 8) with
  a back-end of **dilated convolutions** (rate 2) that enlarge the receptive field
  **without** reducing resolution, aggregating large-scale context while keeping a
  dense, high-quality density map → the **lowest** MAE. (CSRNet, Li et al. CVPR 2018:
  Part A MAE **68.2** / Part B **10.6** — strictly better than MCNN and the plain
  single column.)

The fixed harness handles the data, the loss, the optimiser, the iterations, the seed,
and the evaluation; **your contribution is the entire image→density counter** (backbone
+ density tail). **Design the architecture** that recovers counting accuracy across all
three crowd densities.

## Background
The primary crowd-counting metric is the **counting MAE** = `mean |pred_count −
gt_count|` (RMSE secondary, both lower-is-better; standard MCNN/CSRNet convention). The
task is scored on **three crowd-density scenes** — `medium`, `middense`, `dense` — each a
count-**extrapolation** test (training counts LOW, held-out val counts HIGHER), so a
degenerate **constant-mean** predictor (image-independent uniform density) is hopeless
by construction and must lose. The score is the geometric mean over the three scenes, so
the architecture ordering **plain < MCNN < CSRNet** must hold at every density.

## Implementation Contract
Modify `build_counter` in `crowd-counting/solution/arch.py` to return a
`torch.nn.Module` mapping an image batch `(B, 3, H, W)` to a **non-negative density
map** `(B, h, w)`; the count is its spatial integral (the harness divides by the fixed
`DENSITY_SCALE = 100`, and the count is resolution-invariant).

```python
def build_counter():
    import torch.nn as nn, torch.nn.functional as F
    def conv(ci, co, k=3, d=1):
        return nn.Conv2d(ci, co, k, padding=((k - 1) // 2) * d, dilation=d)
    class CSRNet(nn.Module):                          # dilated SOTA backbone
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2)
            self.b1 = nn.Sequential(conv(3, 32), nn.ReLU(True), conv(32, 32), nn.ReLU(True))
            self.b2 = nn.Sequential(conv(32, 64), nn.ReLU(True), conv(64, 64), nn.ReLU(True))
            self.b3 = nn.Sequential(conv(64, 64), nn.ReLU(True), conv(64, 64), nn.ReLU(True))
            self.backend = nn.Sequential(
                conv(64, 64, 3, d=2), nn.ReLU(True), conv(64, 64, 3, d=2), nn.ReLU(True),
                conv(64, 32, 3, d=2), nn.ReLU(True))
            self.out = nn.Conv2d(32, 1, 1)
        def forward(self, x):
            x = self.pool(self.b1(x)); x = self.pool(self.b2(x)); x = self.pool(self.b3(x))
            return F.softplus(self.out(self.backend(x))).squeeze(1)   # (B,h,w)
    return CSRNet()
```

- A malformed / crashing counter falls back to the default fixed VGG-lite frontend +
  density head.

## Fixed Pipeline & Evaluation
- Data: three crowd-density scenes (`medium` / `middense` / `dense`), each 120 train
  (LOW counts) / 40 val (HIGHER counts) **REAL crowd photos** (128×128×3, ShanghaiTech
  Crowd Counting Dataset, Zhang et al. CVPR 2016 — real surveillance/street photos with
  every human head annotated by a single point; images are bucketed into each scene by
  their REAL annotated head count), exact GT counts. The three scenes are the **three
  validation settings**.
- Training: a few hundred steps with a fixed density loss. **Only the architecture
  changes.**
- Metric (lower is better): **counting MAE** on each scene's val split; RMSE and NAE
  recorded. The score is the geometric mean over the three scenes.
- The scoring midpoint sits between the dilated (CSRNet) MAE and the plain-CNN /
  constant-mean MAE: you score high only by adopting the multi-column, and higher still
  with the dilated backbone — reproducing the plain < MCNN < CSRNet ordering.
