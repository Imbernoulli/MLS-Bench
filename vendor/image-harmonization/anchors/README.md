# Image-Harmonization Repo Expansion — Anchor Provenance

> **STALE DATA NOTICE (2026-07-05):** everything below this line documents anchors measured
> against the OLD fully-synthetic CIFAR + affine-appearance-shift dataset. The repo has since
> been swapped to REAL iHarmony4 composites (Cong et al., DoveNet, CVPR 2020) — see
> `prepare_data.py` and the harness docstring for the new data path. The synthetic-scale
> DROPPED-surface findings below (normalization/colorhead/upsampling/dilation/attention/
> bgstats) have NOT been re-validated on real data and should not be assumed to still hold.
> The 6 SHIPPED tasks' `leaderboard.csv` numbers are also from the OLD synthetic pipeline and
> need GPU RE-ANCHORING on the real iHarmony4 npz before being trusted; CPU smoke tests on
> real data (mild severity, reduced iters, see this repo's data-swap notes) confirm the SAME
> weak<strong ORDERING direction holds for all 6 surfaces, but absolute dB numbers will differ
> substantially (real photographic mismatches are harder than the synthetic affine shift) and
> must be re-measured on GPU at the production iteration budget before shipping.

Self-contained synthetic-composite harmonization harness (`../harness.py`), multi-surface,
config-driven. Data: 2000tr/400val 64x64 CIFAR patches, a KNOWN per-channel affine
appearance shift (gain/contrast + brightness bias + colour tint) applied ONLY inside a
foreground blob mask -> the composite; the un-shifted patch is the EXACT harmonized GT.
Three settings = three appearance-mismatch severities (mild/medium/strong). PRIMARY metric
= FOREGROUND-region PSNR of the harmonized output vs the real GT (higher better), measured
only inside the foreground mask; the copy-composite do-nothing floor (`comp_fg_psnr`) is
reported so a degenerate scores gain 0.

Reference methods: DoveNet (Cong et al., CVPR 2020, iHarmony4) = mask-conditioned residual
U-Net; RainNet (Ling et al., CVPR 2021) = region-aware AdaIN (RAIN) transferring background
feature stats onto the foreground.

## GPU validation

- k1 H20, image `pytorch:2.4.1-cuda12.1-cudnn9-runtime`, torch 2.4.1+cu121, 500 iters.
- `anchor_seed42.tsv` — full seed-42 sweep of EVERY surface x EVERY variant x 3 severities
  (99 rows) through the real harness.
- `anchor_seed1.tsv` — seed-1 margin-robustness confirmation for the 6 SHIPPED tasks'
  weak+strong variants (36 rows).
- `oracle_e2e.tsv` — the ORACLE PATH: each shipped task's weak/strong EDIT applied to a fresh
  copy of the real solution file, then run through the harness exactly as the task scripts do
  (36 rows). Confirms edit -> config -> metric.

## Shipped tasks (6) — weak < strong on ALL 3 severities and BOTH seeds

| task | surface | weak | strong (SOTA-aligned) | seed42 weak->strong fg_psnr (mild/med/strong) |
|------|---------|------|-----------------------|-----------------------------------------------|
| cv-harmonization-region-norm      | network    | copy (do-nothing)     | mask-conditioned U-Net (DoveNet) | 15.02/12.92/11.13 -> 20.68/19.49/18.52 |
| cv-harmonization-mask-conditioning| maskcond   | mask-blind            | mask-concat (DoveNet)            | 16.50/15.31/14.11 -> 20.25/19.42/17.35 |
| cv-harmonization-loss-region      | loss       | background-only (degen)| whole+foreground-emphasis L1    | 13.64/11.21/10.23 -> 17.01/18.76/17.80 |
| cv-harmonization-feature-fusion   | fusion     | no skips              | U-Net skip connections           | 19.61/17.15/12.86 -> 20.27/19.85/18.17 |
| cv-harmonization-activation       | activation | identity (linear)     | ReLU                             | 16.21/14.99/13.56 -> 20.14/19.92/17.42 |
| cv-harmonization-input-norm       | inputnorm  | bg-whiten (corrupts)  | raw composite                    | 11.17/10.92/10.68 -> 20.05/18.97/18.12 |

## Surfaces DROPPED as scored tasks (non-monotone at this synthetic scale — honest)

The synthetic composite is a GLOBAL per-channel affine appearance shift, so refinement /
context / normalization surfaces have little headroom once the net is mask-conditioned. The
following surfaces are KEPT in the harness (documented editable surfaces / future higher-scale
validation) but NOT shipped as scored tasks because their strong variant does NOT beat the
weak one on all 3 severities:

- **normalization** (RAIN/instance/batch/none): plain `none` beats everything; RAIN does NOT
  win here (its advantage needs the real, complex, spatially-varying iHarmony4 gaps — same as
  the derain/dehaze/deshadow finding). `batch<none` is monotone but tells a misleading "no-norm
  wins" story, so dropped.
- **colorhead** (residual/affine_global/affine_spatial): affine heads win at mild/medium but
  INVERT at strong (residual best there). Non-monotone.
- **upsampling** (nearest/transpose/bilinear): nearest WINS at mild (transpose loses). Non-monotone.
- **dilation** (1/4/8): rate-1 (no dilation) is best everywhere — at 64px the 3-level U-Net
  already sees enough context. Hypothesis inverted.
- **attention** (SE on/off): attention helps at medium/strong but HURTS at mild. Non-monotone.
- **bgstats** (background-mean input channels): helps only at strong; inverts at mild/medium.

Only ship surfaces whose weak default genuinely cripples the fixed-correct rest of the
pipeline, with the order preserved across all 3 severities and both seeds.
