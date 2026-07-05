# image-deshadow anchors

## Real-data swap (2026-07-04/05)

The `vendor/image-deshadow/` harness and `vendor/data_scripts/image-deshadow/prepare_data.py`
were swapped from a FULLY SYNTHETIC dataset (CIFAR-10-derived clean patches + a synthetic
physics-based linear-illumination cast-shadow model) to REAL ISTD (Wang, Li & Yang, "Stacked
Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow
Removal", CVPR 2018) shadow/shadow-free/mask photo triplets.

- **Source**: HF Hub dataset `Donghyun99/ISTD` (parquet mirror of the official ISTD release),
  fetched via `hf_hub_download` through `hf-mirror.com` (proxy-friendly mirror; the canonical
  `huggingface.co` endpoint 401'd through this environment's proxy). 6 parquet shards
  (4 train + 2 test), ~2.1GB, downloaded to
  `/mnt/moonfs/lvbohan-b0/deshadow-real/istd_raw/data/` (B0-side moonfs staging; NOT on local
  disk, which was at 97% full).
- **3 severities** (light / medium / heavy) are TERCILES of each triplet's MEASURED shadow
  attenuation ratio `mean(shadow_img[mask]) / mean(shadow_free_target[mask])`, pooled over
  ISTD's own train+test triplets, then re-split back into ISTD's own DISJOINT-SCENE train/test
  for the task's train/val split (89 vs 46 scenes, 0 overlap). Each 640x480 triplet is
  center-cropped to a square and bilinear-resized to 64x64 (the harness's existing convention);
  the mask is re-binarized at 0.5 after resize. Triplets with too few shadow pixels (< 50 at
  full res or < ~12 at 64x64) are dropped (10/1330 train, 3/540 test).
- Resulting copy-floor (do-nothing) PSNR is close to the old synthetic floors (light
  17.25/17.44, medium 13.16/13.66, heavy 11.13/11.33 dB, real/synthetic) -- a strong sanity
  check that the real severity ladder is a faithful, comparable replacement.
- Only the DATA changed. The mask-conditioned residual deshadower architecture, the
  shadow-region-PSNR metric, and all 12 editable surfaces are byte-identical to before the
  swap (only comments/docstrings describing the data source were updated).

## CPU smoke-test re-check (ordering only, not final anchors)

Because the actual GPU re-anchor requires k1 (out of scope for the B0-side data-swap pass),
all 7 previously-shipped surfaces (network x2 configs, mask, dilation, fusion, physics,
upsampling) were re-checked for weak<strong ORDERING on a CPU using the harness's real
`_resolve_hook` / `_build_configured` / `psnr_masked` functions, with the model width shrunk
(`H.BASE` 32->8..14) and iteration count reduced purely for CPU tractability under heavy
machine contention -- this affects absolute PSNR values, not the real scoring path, and is
NOT a substitute for a full GPU re-anchor. Full provenance/log: `real_istd_cpu_smoke.log`
(multiple passes at increasing fidelity as ambiguous cases were re-checked; final decisive
runs used BASE=14, 30-60 iters, 2 seeds 42/123, val cap 60-100).

Result: 5 of 7 surfaces (network, mask, dilation, fusion) reconfirmed weak<strong on every
setting at the higher-fidelity re-check. `upsampling` stays aggregate-monotone (gmean
strong>weak) on both seeds but is now the weakest lever (light flips weak>strong on both
seeds; medium/heavy carry the aggregate). `physics` INVERTS at the aggregate level on both
seeds on real data (was aggregate-monotone on the old synthetic data) and has been
re-classified DROPPED -- see `tasks/deshadow-physics/DROPPED.md`.

## Still pending (NOT done in this pass)

A full GPU re-anchor of the 6 surviving shipped tasks' `score_spec.py` `_WEAK`/`_STRONG`
PSNR constants on the real ISTD data, on k1 (this environment's B0 devmachine cannot mlaunch
k1 GPUs directly; the raw ISTD parquet shards would need to be rsync'd to a k1-side moonfs
path and pointed to via `ISTD_PARQUET_DIR`, then `vendor/data_scripts/image-deshadow/
_validate_gpu.sh` run there). The pinned anchors in the 6 shipped score_spec.py files are
STILL the old synthetic-GPU numbers; they may need retuning (e.g. adjusted `_MID`/`_SCALE` or
`_WEAK`/`_STRONG`/`_FLOOR` constants) once real full-fidelity GPU numbers are available, since
absolute PSNR on real photos need not match the old synthetic floors exactly even though the
CPU-smoke-test ordering direction is confirmed to hold.
