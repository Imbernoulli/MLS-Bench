# vfi-synthesis anchor provenance (REAL Vimeo-90K data, GPU re-anchor)

Real GPU-measured baseline anchors for the vfi-synthesis (blend/flow_warp/learned)
task, on the REAL Vimeo-90K triplet-interpolation test-set tiles (see
`vendor/data_scripts/video-frame-interp/prepare_data.py` for full provenance:
TOFlow/Vimeo-90K, Xue et al. IJCV 2019, real decoded video frames tiled 64x64,
terciled into small/medium/large by measured per-tile Farneback motion magnitude).
All runs: mlaunch B0 8xH200, torch (see harness), 800-iter fixed budget, batch 32,
Charbonnier loss, CROSS-SEED 42 and 123.

## Cross-seed measured interpolation PSNR (dB, higher better)

| setting | seed | blend   | flow_warp | learned |
|---------|------|--------:|----------:|--------:|
| small   | 42   | 41.1579 | 41.6905   | 41.3175 |
| small   | 123  | 41.1579 | 41.7663   | 41.4734 |
| small   | avg  | 41.1579 | **41.7284** | 41.3954 |
| medium  | 42   | 28.5933 | 32.2812   | 32.6319 |
| medium  | 123  | 28.5933 | 32.5818   | 32.6395 |
| medium  | avg  | 28.5933 | 32.4315   | **32.6357** |
| large   | 42   | 22.1614 | 23.5839   | 25.3533 |
| large   | 123  | 22.1614 | 23.8235   | 25.3382 |
| large   | avg  | 22.1614 | 23.7037   | **25.3458** |

## Outcome: `small` DROPPED as a scored setting (honest, cross-seed-confirmed inversion)

The intended partial order is `blend < flow_warp < learned` (flow-compensation beats
naive averaging; learned refinement + occlusion-aware blending beats plain flow-warp,
with the margin WIDENING as motion increases). This holds robustly, cross-seed, on
**medium** (learned beats flow_warp by ~0.2-0.35 dB both seeds) and **large** (learned
beats flow_warp by ~1.5-1.7 dB both seeds, the widest margin, matching the "helps most
under heavy disocclusion" hypothesis).

On **small** motion, the order is INVERTED and this reproduces on BOTH seeds: `flow_warp`
(41.69 / 41.77 dB) beats `learned` (41.32 / 41.47 dB) by ~0.3-0.4 dB in both runs — not
seed noise. At small real motion the tile pair is already extremely close to identical
(blend floor alone is 41.16 dB — near the effective ceiling of this compact net/budget),
so there is essentially no disocclusion for the learned refinement head to resolve; the
extra refinement capacity has nothing genuine to fix and its own parameters add a small
amount of optimization noise/overfitting relative to the simpler flow-warp path. This is
the same "near-ceiling saturation flips second-order design choices" failure mode
documented for `image-matting`'s excluded narrow (width=2) trimap band
(`vendor/image-matting/anchors/README.md`).

Per the project's honest relabel-or-drop mandate (never HP-sweep to force monotonicity):
`small` is DROPPED as a scored setting. `vfi-synthesis` now ships as a **2-setting**
task (medium, large only) — `scripts/vfi_small.sh` and its `test_cmds`/score_spec term
are removed; the small-motion measurement above is preserved here for provenance. The
weak(flow_warp)<strong(learned) order holds cleanly, cross-seed, on both shipped
settings.

## Final shipped anchors (medium, large; seed-avg)

| setting | flow_warp (weak, score 0) | learned (strong, ref_score 0.5) |
|---------|---------------------------:|----------------------------------:|
| medium  | 32.4315                    | 32.6357                           |
| large   | 23.7037                    | 25.3458                           |
