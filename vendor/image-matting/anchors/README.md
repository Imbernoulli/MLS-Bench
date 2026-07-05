# image-matting anchor provenance

> **RE-ANCHORED 2026-07-05 on REAL PPM-100, CROSS-SEED (B0 8xH200).** The previous
> STALE notice (below, kept for provenance) flagged that `prepare_data.py` had been
> rewritten to build the dataset from REAL photos + real hand-annotated alpha mattes
> (PPM-100) at 160x160 (up from the old 128x128 synthetic composites), and that a GPU
> re-anchor was required before trusting `score_spec`/`leaderboard.csv` numbers. That
> re-anchor is now DONE: full budget (400 iters), cross-seed 42/123, all 8 surfaces x
> all 3 trimap widths x all baselines, on B0 8xH200 (torch 2.4.1). Result: 4 of the 8
> previously-shipped surfaces (arch, loss-design, norm, skip) are CONFIRMED robustly
> monotone (weak<strong SAD) on all 3 widths, both seeds, and have been refreshed with
> real numbers below. The other 4 (attention, decoder-design, dilation, upsampling) do
> NOT meet the cross-seed per-setting robustness bar on real data (each has at least one
> trimap width that sign-flips between seeds or robustly inverts) and have been DROPPED
> — see each task's `DROPPED.md` for the full per-setting/per-seed numbers and
> reasoning. The old CPU-smoke-test finding below (that loss-design/norm/upsampling
> showed transient wrong-direction ordering at low iters that resolved by iters=60/30)
> was about UNDER-TRAINING noise at short budgets, which is a DIFFERENT and weaker
> concern than the GPU cross-seed robust inversions found here at the FULL 400-iter
> budget; loss-design and norm both turned out robust cross-seed (confirming the old
> smoke-test's optimistic read), while upsampling did not (its xwide setting
> seed-flips at full budget, which the old single-seed smoke test could not have caught).
>
> <details><summary>Old STALE notice (2026-07-04/05, superseded)</summary>
>
> A CPU smoke test (6/20/60/150 train iterations, CPU, batch 8, real PPM-100 val split,
> 90 train/10 val samples at 160x160) confirms the harness trains and evaluates cleanly
> on the new real data end-to-end for all 8 shipped tasks, and that weak<strong (SAD)
> holds for all 8 at iters=6 EXCEPT `cv-matting-loss-design` (unk_comp vs whole_l1) and
> `cv-matting-norm` (batch vs identity), which were INVERTED at iters=6 but which a
> longer CPU run confirmed is under-training noise, not a genuine real-data ordering
> failure: at iters=20 the gap between weak/strong widened in the WRONG direction
> (loss-design 2.22 vs 2.86; norm 2.04 vs 2.86), but by iters=60 both surfaces flipped to
> the CORRECT weak>strong direction with a clear margin (loss-design: whole_l1=1.312 >
> unk_comp=0.616; norm: identity=1.009 > batch=0.616) — i.e. BatchNorm / unknown-band
> composition loss need more than a handful of steps to show their benefit on this real,
> harder data (real hair/edge alpha transitions are less trivial to fit than the old
> rendered proxy), but the intended ordering re-emerges with a slightly longer budget,
> well within the GPU re-anchor's planned 400-step regime. A third, much smaller
> near-inversion at iters=6 (`cv-matting-upsampling`, medium width only: nearest=3.2741
> vs learned=3.2785, a 0.0044 SAD gap — within noise, and wide/xwide already ordered
> correctly at iters=6) was independently confirmed as noise: at iters=30, medium width,
> nearest(weak)=1.2208 clearly worse than learned(strong)=0.9656. See
> `/tmp/matting_smoke_results.json` (this environment) for the full 6-iteration
> 8-task/all-widths smoke-test table used to reach this conclusion.
>
> </details>

Real GPU-measured baseline anchors for the trimap-guided image-matting tasks
(`tasks/cv-matting-*`). CURRENT (2026-07-05): B0 8xH200, torch 2.4.1, REAL PPM-100
photos + real hand-annotated alpha mattes at 160x160, 100 train / 40 val, lr 1e-3 +
grad-clip, CROSS-SEED 42/123, 400-step full-budget fit. OLD (superseded, kept below
for the still-shipped surfaces' historical section only): mlaunch on k1 H20, image
`msai-cn-beijing.cr.volces.com/public/pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`,
torch 2.4.0 + cuda, 100 train / 40 val 128x128 synthetic composites, lr 1e-3 +
grad-clip, seed 42, 400-step fit (250 for the loss surface). Metric = alpha **SAD**
(/1000) in the trimap UNKNOWN band (LOWER is better).

## Setting scheme (the 3 validation settings)

The trimap is RE-DERIVED from the exact GT alpha at eval time by eroding the solid
fg/bg regions by a band `width`. The three SHIPPED settings are:

| setting | band width | unk_frac |
|---------|-----------|----------|
| medium  | 6         | ~0.39    |
| wide    | 9         | ~0.47    |
| xwide   | 12        | ~0.54    |

Training always uses the medium (width-6) band; only the scored val band changes.

Two regimes were measured and EXCLUDED (see anchor_widths_2_6_12.tsv +
anchor_width_18.tsv): a NARROW band (width 2) is too easy — every method saturates and
second-order design choices invert; an EXTREME band (width >=16, unk_frac >0.65) is so
hard the refinement/attention levers add noise and the order flattens/inverts. The
weak<strong order holds cleanly across the three moderate-to-wide bands {6,9,12}.

## Files

- `anchor_widths_2_6_12.tsv`  — the main run: ALL 12 surface baselines at widths
  {2 (narrow), 6 (medium), 12 (old-wide)}. Format: `surface|baseline|setting  sad  mse  grad`.
- `anchor_width_18.tsv`       — supplementary xwide=18 (extreme) run for all 12 surfaces.
- `anchor_width_9.tsv`        — supplementary width-9 run for the 8 kept surfaces.
- `anchor_final_scheme_6_9_12.tsv` — the combined anchors used by the score_specs
  (medium=width6, wide=width9, xwide=width12) for the 8 SHIPPED surfaces.

## Shipped tasks (4, CURRENT real-data cross-seed, seed-averaged medium/wide/xwide)

| task (surface)              | weak                          | strong (SOTA)                                |
|-----------------------------|-------------------------------|-----------------------------------------------|
| cv-matting-arch             | plain enc-dec .4288/.7531/1.3797 | DIM skips+refine (Xu 2017) .2613/.6364/1.0736 |
| cv-matting-loss-design      | whole-img L1 .3421/.7642/1.2797  | unk-band+composition (Xu 2017) .2683/.6355/1.0745 |
| cv-matting-norm             | no-norm .4159/.9442/1.7256       | BatchNorm .2712/.6241/1.0539              |
| cv-matting-skip             | drop-skip .4416/.7903/1.4944     | full concat skip .2736/.6334/1.1033       |

All 4 above are robustly monotone (weak SAD > strong SAD) on ALL THREE trimap widths,
on BOTH seeds (42/123) — see each task's `score_spec.py` docstring for the full
per-seed table. `cv-matting-arch`'s `constant`/copy-trimap degenerate baseline remains
seed-invariant and far worse than either real net (medium/wide/xwide =
3.2950/4.3704/5.3597), confirming the metric stays monotone end-to-end.

## Dropped surfaces (12 total: 4 never-shipped + 4 re-dropped 2026-07-05, non-monotone cross-seed on real data)

Never shipped as tasks: `trimap` (trimap-blind beats one-hot — the synthetic fg/bg
colour overlap is too weak for the trimap to be necessary), `refine` (parameter-free
residual sharpening hurts an already-trained matte), `propagation` (image-guided
filter over-smooths the trained matte), `fgpred` (the aux FG head doesn't reliably
help). Their editable surfaces remain in the harness + `solution/` as documented RQs
(fail-safe), but no task ships.

RE-DROPPED 2026-07-05 (previously shipped on synthetic data, now fail the real-data
cross-seed bar — see each task's `DROPPED.md` for full numbers):
- `cv-matting-attention` (gap vs selfattn): only 1/3 widths (wide) robust; medium
  sign-flips across seeds, xwide robustly inverts both seeds.
- `cv-matting-decoder-design` (bilinear vs unet): 2/3 widths robust (medium, xwide),
  but `wide` robustly inverts on BOTH seeds with a real margin — not noise.
- `cv-matting-dilation` (single vs aspp): 0/3 widths robust; task-level gmean is
  robustly wrong (aspp scores worse than single) on both seeds — the worst case.
- `cv-matting-upsampling` (nearest vs learned): 2/3 widths robust (medium, wide), but
  `xwide` is a degenerate near-tie that sign-flips at seed123 (small margin).

These were shipped on the OLD synthetic dataset under a looser single-seed
"2/3 settings + noise-level inversion on the third" bar (see the superseded STALE
notice above); the current cross-seed re-anchor applies a stricter per-setting,
both-seeds-robust bar (matching the `image-deblur` repo's contemporaneous re-anchor),
under which none of these 4 qualify for a clean relabel or ship-as-is, so they were
dropped rather than HP-swept or cherry-picked into a partial fix.
