# DROPPED surface: deshadow-multiscale (NOT shipped)

This editable surface (single-scale vs coarse-to-fine pyramid) was DESIGNED and GPU-VALIDATED
on the proven image-deshadow harness but DROPPED because it is NOT CROSS-SEED ROBUST on the
synthetic near-affine multiplicative cast-shadow data. It looked aggregate-monotone on seed 42
but INVERTED on seed 1 (the pyramid loses on all three settings), so it is not a reliable
weak->strong lever.

Reason: on this synthetic 64px shadow the single-scale mask-guided U-Net already covers the
whole patch; the coarse-to-fine pyramid's half-resolution relight helps large shadows on some
seeds but adds a fusion path that hurts on others -> the ordering is seed-dependent.

Per-setting SHADOW-REGION PSNR, weak (single) vs strong (pyramid) (k1 H20, torch 2.4.1+cu121,
400 iters):
  seed 42:  light 33.23->33.45 (+)  medium 30.58->30.27 (-)  heavy 26.61->29.28 (+)  | gmean +0.93
  seed 1 :  light 34.21->34.08 (-)  medium 32.16->31.75 (-)  heavy 28.97->28.08 (-)  | gmean -0.51
The aggregate ordering FLIPS between seeds -> dropped.

The surface code remains in vendor/image-deshadow/harness.py (SURFACES tuple) + the
solution/multiscale.py stub for provenance, but no task is shipped. See the SHIPPED, robustly-
monotone deshadow surfaces (network-design / mask-guidance / mask / dilation / fusion /
physics / upsampling).
