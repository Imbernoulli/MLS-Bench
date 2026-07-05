# Image Matting: Design the Alpha-Matte Loss (whole-image L1 vs unknown-band + composition)

## Research Question
Trimap-guided **image matting** predicts a **soft foreground alpha** `α ∈ [0,1]` per
pixel (NOT a hard segmentation mask) via the compositing equation
`I = α·F + (1−α)·B`. A trimap splits the image into definite-foreground (`α=1`),
definite-background (`α=0`), and an **unknown** transition band where the matte must
be solved. The training loss decides *where the network spends its capacity*. A naive
**uniform whole-image alpha-L1** averages the error over every pixel; the trivial
solid fg/bg regions (which any net nails immediately and which dominate the image
area) swamp the mean, so the gradient signal on the hard **unknown** band is diluted
and the soft transition is under-fit. The matting-standard practice is to (i) restrict
the loss to the **unknown band** and (ii) add a **composition loss**
`|I − (α·F + (1−α)·B)|` (Deep Image Matting, Xu et al. 2017 — equal weight, `w=0.5`),
optionally a Laplacian-pyramid / gradient term for sharper edges. **Redesign the loss**
to recover matte accuracy on a short fine-tune.

The fixed harness handles the data, the network (U-Net encoder + decoder), the trimap
conditioning, the optimiser, the iterations, the seed, and the evaluation; your
contribution is the **training loss only**.

## Background
The standard matting metric is the **alpha SAD** (sum of absolute alpha differences,
`/1000`) computed **only in the trimap unknown band** (lower is better); MSE and
gradient error are secondary. Because the ground-truth alpha in the unknown band is a
genuine **soft ramp** (spanning 0→1, mean ≠ 0.5), a **constant-0.5 / copy-trimap
predictor scores `CONST_HALF_SAD`** and a per-image mean-alpha predictor scores
`MEAN_ALPHA_SAD` — both far above any real matting net — so the metric is monotone in
matting quality (a trivial output is clearly beaten). Gradient error additionally
cannot be gamed by any constant predictor.

## Implementation Contract
Modify `get_matting_loss` in `image-matting/solution/loss.py` to return a callable
`loss_fn(pred, gt, image, fg, bg, trimap, unknown)`:
- `pred`, `gt`: `(B,H,W)` predicted / GT alpha in `[0,1]`
- `image`, `fg`, `bg`: `(B,3,H,W)` composite / foreground / background (`I = α·F + (1−α)·B`)
- `trimap`: `(B,H,W)` in `{0, 0.5, 1}`;  `unknown`: bool `(B,H,W)` (the scored band)

```python
def get_matting_loss():
    def loss_fn(pred, gt, image, fg, bg, trimap, unknown):
        u = unknown.float()
        d = u.sum(dim=(-2, -1)).clamp(min=1.0)
        alpha_l = ((pred - gt).abs() * u).sum(dim=(-2, -1)) / d          # alpha L1 (unknown)
        comp = pred.unsqueeze(1)*fg + (1 - pred.unsqueeze(1))*bg
        comp_l = ((comp - image).abs().mean(1) * u).sum(dim=(-2, -1)) / d # composition L1
        return (alpha_l + 0.5*comp_l).mean()
    return loss_fn
```

- The loss must return a finite scalar torch tensor. A malformed / crashing /
  non-finite loss falls back to the uniform whole-image alpha-L1.

## Fixed Pipeline & Evaluation
- Data: 100 train / 40 val **synthetic composites** (128×128), `I = α·F + (1−α)·B`
  with an **exact** soft GT alpha of a random blobby shape with fine hair-like detail,
  and a derived trimap (definite-fg / definite-bg / **unknown** band).
- Network: a **fixed U-Net matting net** (encoder + skip-connection decoder), fed
  RGB + the trimap, trained a short fine-tune with your loss. **Only the loss changes.**
- Settings: three **trimap-width** difficulties — `medium` (band width 6), `wide`
  (band width 9) and `xwide` (band width 12, thickest unknown band, hardest). The
  trimap is re-derived from the exact GT alpha at eval time; training uses the medium
  band. The score is the **gmean over all three settings** (`wide`/`xwide` hidden).
- Metric (lower is better): **alpha SAD** in the trimap unknown band on the val split;
  MSE and gradient error are also recorded.
- The scoring midpoint sits between the whole-image-L1 start and the unknown-band +
  composition loss: you score above 0.5 only by fixing the loss with headroom.
