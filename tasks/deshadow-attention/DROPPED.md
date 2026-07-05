# DROPPED surface: deshadow-attention (NOT shipped)

This editable surface was DESIGNED and GPU-VALIDATED on the proven image-deshadow harness but
DROPPED because it is not monotone (weak -> strong) on the synthetic near-affine
multiplicative cast-shadow data. The mask-guided residual deshadower is already near-ceiling on
this data, so this lever does not reliably improve shadow-region PSNR.

squeeze-excite channel attention does not help the mask-guided deshadower on this synthetic shadow; strong<weak on all three settings.

Reason: the mask-guided residual net already knows where/how-much to brighten (the mask supplies the spatial prior), so the SE channel gate adds parameters/noise without a discriminative signal and HURTS every setting.

Per-setting SHADOW-REGION PSNR, weak vs strong (k1 H20, torch 2.4.1+cu121, 400 iters, seed 42):
  light  : weak= 33.53 -> strong= 32.70  (delta -0.83)
  medium : weak= 30.96 -> strong= 29.29  (delta -1.67)
  heavy  : weak= 27.80 -> strong= 27.19  (delta -0.61)
  gmean(light,medium,heavy): weak=30.672 strong=29.639 delta=-1.033

The surface code remains in vendor/image-deshadow/harness.py (SURFACES tuple) + the
solution/attention.py stub for provenance, but no task is shipped for it. See the SHIPPED,
robustly-monotone deshadow surfaces (deshadow-network-design / -mask-guidance / -mask /
-dilation / -multiscale) for the validated levers.
