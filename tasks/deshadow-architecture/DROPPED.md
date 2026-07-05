# DROPPED surface: deshadow-architecture (NOT shipped)

This editable surface was DESIGNED and GPU-VALIDATED on the proven image-deshadow harness but
DROPPED because it is not monotone (weak -> strong) on the synthetic near-affine
multiplicative cast-shadow data. The mask-guided residual deshadower is already near-ceiling on
this data, so this lever does not reliably improve shadow-region PSNR.

deeper 2-level encoder-decoder does not help on 64px near-affine cast shadow; strong<weak on medium/heavy (aggregate strong<weak).

Reason: on this synthetic near-affine multiplicative shadow the shallow 1-level mask-guided U-Net is already near-ceiling; the extra downsampling level over-smooths the small 64px patch and HURTS the medium/heavy settings.

Per-setting SHADOW-REGION PSNR, weak vs strong (k1 H20, torch 2.4.1+cu121, 400 iters, seed 42):
  light  : weak= 32.92 -> strong= 33.28  (delta +0.35)
  medium : weak= 30.26 -> strong= 29.97  (delta -0.29)
  heavy  : weak= 27.28 -> strong= 26.13  (delta -1.15)
  gmean(light,medium,heavy): weak=30.067 strong=29.650 delta=-0.417

The surface code remains in vendor/image-deshadow/harness.py (SURFACES tuple) + the
solution/architecture.py stub for provenance, but no task is shipped for it. See the SHIPPED,
robustly-monotone deshadow surfaces (deshadow-network-design / -mask-guidance / -mask /
-dilation / -multiscale) for the validated levers.
