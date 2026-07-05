# DROPPED surface: deshadow-loss (NOT shipped)

This editable surface was DESIGNED and GPU-VALIDATED on the proven image-deshadow harness but
DROPPED because it is not monotone (weak -> strong) on the synthetic near-affine
multiplicative cast-shadow data. The mask-guided residual deshadower is already near-ceiling on
this data, so this lever does not reliably improve shadow-region PSNR.

composite (L1+SSIM+color+composition) loss does not beat plain shadow-weighted L1 on this synthetic shadow; strong<weak on all three settings.

Reason: the shadow ground truth is an EXACT linear-model recovery, so plain shadow-up-weighted L1 already fits it; the extra SSIM/color/composition terms add gradient noise and marginally HURT every setting.

Per-setting SHADOW-REGION PSNR, weak vs strong (k1 H20, torch 2.4.1+cu121, 400 iters, seed 42):
  light  : weak= 33.40 -> strong= 33.18  (delta -0.23)
  medium : weak= 30.15 -> strong= 29.67  (delta -0.48)
  heavy  : weak= 28.16 -> strong= 27.93  (delta -0.22)
  gmean(light,medium,heavy): weak=30.494 strong=30.184 delta=-0.310

The surface code remains in vendor/image-deshadow/harness.py (SURFACES tuple) + the
solution/loss.py stub for provenance, but no task is shipped for it. See the SHIPPED,
robustly-monotone deshadow surfaces (deshadow-network-design / -mask-guidance / -mask /
-dilation / -multiscale) for the validated levers.
