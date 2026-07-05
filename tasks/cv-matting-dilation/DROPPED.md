# DROPPED surface: cv-matting-dilation (NOT shipped)

This editable surface (single 3x3 bottleneck conv `single` vs dilated multi-rate ASPP
block `aspp`, Chen 2017) was DESIGNED and validated on a fully-synthetic 128x128
composite dataset with a single seed, but on the REAL PPM-100 cross-seed re-anchor
(B0 8xH200, torch 2.4.1, full budget 400 iters, seeds 42/123) the intended
weak(single)<strong(aspp) ordering (lower alpha SAD is better) holds robustly on
0 of 3 trimap-width settings:

Alpha SAD (unknown band, /1000, LOWER is better):
  medium: seed42  single=0.2654 aspp=0.2571 (aspp wins,   Δ=+0.0083)
          seed123 single=0.2412 aspp=0.2695 (single wins, Δ=-0.0283)
          -- SIGN FLIPS across seeds; not robust either direction.
  wide:   seed42  single=0.6657 aspp=0.6486 (aspp wins,   Δ=+0.0171)
          seed123 single=0.5598 aspp=0.7155 (single wins, Δ=-0.1557)
          -- SIGN FLIPS across seeds, with a LARGE seed123 margin; not robust.
  xwide:  seed42  single=1.1311 aspp=1.2454 (single wins, Δ=-0.1143)
          seed123 single=1.0039 aspp=1.2065 (single wins, Δ=-0.2026)
          -- single robustly wins BOTH seeds (inverts the intended ordering, large
             margin).

The intended ordering (aspp > single) holds robustly on 0/3 settings; the OPPOSITE
ordering (single > aspp) holds robustly only on xwide (1/3), with the other 2
settings seed-unstable (sign flips, one with a very large seed123 swing on "wide").
Neither direction is a clean relabel candidate, and the task-level gmean is itself
robustly WRONG in the intended direction on both seeds (aspp scores worse than single
at the task level, seed42 gmean single=0.5846<aspp=0.5922, seed123
single=0.5137<aspp=0.6150) -- the worst case of the 4 non-robust matting surfaces.
Per the project mandate (never HP-sweep to force monotonicity; drop honestly when the
signal is degenerate/seed-unstable rather than a clean relabel), this surface is
dropped. Plausibly, the short 400-iter fine-tune is too brief for the ASPP block's
extra multi-rate dilated-conv capacity to reliably converge on real photographic
texture, so which design "wins" is dominated by per-run optimization noise rather
than a genuine capacity advantage at this budget. The surface code remains in
`image-matting/solution/dilation.py` (lines 35-47) + this task's
`edits/`/`scripts/`/`task_description.md` for provenance, but no task is shipped for
it.
