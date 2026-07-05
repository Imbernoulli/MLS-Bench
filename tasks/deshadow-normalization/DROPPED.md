# DROPPED surface: deshadow-normalization (NOT shipped)

This editable surface was DESIGNED and GPU-VALIDATED on the proven image-deshadow harness but
DROPPED because it is not monotone (weak -> strong) on the synthetic near-affine
multiplicative cast-shadow data. The mask-guided residual deshadower is already near-ceiling on
this data, so this lever does not reliably improve shadow-region PSNR.

instance normalization inside the blocks HURTS the mask-guided deshadower on this synthetic shadow; strong<weak on all three settings (badly on medium).

Reason: a cast shadow is a per-image MULTIPLICATIVE illumination change that instance-norm partially normalises AWAY, destroying the very signal the deshadower must recover -> large drop, especially on medium.

Per-setting SHADOW-REGION PSNR, weak vs strong (k1 H20, torch 2.4.1+cu121, 400 iters, seed 42):
  light  : weak= 33.05 -> strong= 31.93  (delta -1.12)
  medium : weak= 29.86 -> strong= 26.18  (delta -3.68)
  heavy  : weak= 27.49 -> strong= 26.49  (delta -1.00)
  gmean(light,medium,heavy): weak=30.048 strong=28.082 delta=-1.966

The surface code remains in vendor/image-deshadow/harness.py (SURFACES tuple) + the
solution/normalization.py stub for provenance, but no task is shipped for it. See the SHIPPED,
robustly-monotone deshadow surfaces (deshadow-network-design / -mask-guidance / -mask /
-dilation / -multiscale) for the validated levers.
