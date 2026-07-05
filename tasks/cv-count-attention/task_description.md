# Crowd/Object Counting: Spatial ATTENTION (clutter suppression)

## Research Question
The scenes contain unannotated distractor **clutter** that looks like objects but is not counted. Without attention, the counter spends density mass on this clutter and mis-counts. A learned **spatial-attention gate** predicts a per-pixel weight in [0,1] and multiplies the features by it, suppressing clutter and focusing on real objects — the idea behind SCAR / ADCrowdNet / SFANet. **Design the attention module**.

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
Modify `build_attention` in `crowd-counting/solution/attention.py` (see the file for the exact
signature and a worked strong example):

```python
def build_attention(cin):  # -> nn.Module (features -> gated features)
    ...
```

The default is identity (no attention); add a spatial-attention gate to win. A malformed / crashing surface falls back to the harness default.

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
