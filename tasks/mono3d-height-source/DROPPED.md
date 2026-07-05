# DROPPED surface: mono3d-height-source (NOT shipped)

This editable surface (global-constant object height H0=1.5m vs per-object predicted height
`pred_H` for the geometry depth decode `Z=f·H/h2d`) was DESIGNED and GPU-VALIDATED on a
fully-synthetic dataset with a wide, uniform mix of car/pedestrian/cyclist heights (where a
global constant badly mis-scales non-car objects), but on the REAL KITTI 3D Object Detection
dataset cross-seed re-anchor (B0 8xH200, torch 2.4.1, 1200 steps, seeds 42/123) the intended
weak(constant)<strong(perobject) ordering ROBUSTLY INVERTS on 2 of 3 difficulty tiers, and the
third tier is seed-unstable:

AP3D@0.25:
  easy:     seed42 constant=0.316058 perobject=0.320438 (perobject wins, Δ=+0.0044)
            seed123 constant=0.348905 perobject=0.331387 (constant wins, Δ=+0.0175)
            -- SIGN FLIPS across seeds; not robust either direction.
  moderate: seed42 constant=0.168086 perobject=0.164679 (constant wins, Δ=+0.0034)
            seed123 constant=0.176036 perobject=0.167518 (constant wins, Δ=+0.0085)
            -- constant robustly wins both seeds (inverts the intended ordering).
  hard:     seed42 constant=0.176414 perobject=0.171620 (constant wins, Δ=+0.0048)
            seed123 constant=0.184084 perobject=0.168744 (constant wins, Δ=+0.0153)
            -- constant robustly wins both seeds (inverts the intended ordering).

The intended ordering (perobject > constant) holds on 0/3 settings robustly; the OPPOSITE
ordering (constant > perobject) holds robustly on 2/3 settings (moderate, hard) but the third
(easy) flips sign between seeds and is not distinguishable from noise. This is neither a clean
"honest relabel" candidate (constant does not robustly beat perobject on all 3 settings) nor a
validation of the original design (perobject does not robustly beat constant on any setting).
It is the "degenerate near-tie / seed-unstable" case explicitly called out in the project
mandate: per that mandate ("if it's just a degenerate near-tie / seed-unstable... DROP it
honestly"), this surface is dropped rather than HP-swept or cherry-picked into a partial
relabel. Plausibly, on real KITTI the per-object height predictor's benefit (correcting for
class-height variance) is swamped by real bbox/occlusion noise at this scale, and the
per-object head's own prediction error occasionally exceeds the class-constant's bias, making
the two methods practically indistinguishable/noise-dominated on 2 of 3 tiers and mildly,
consistently reversed on the other 2. The surface code remains in
`mono3d-detection/solution/height_source.py` (lines 30-41) + this task's
`edits/`/`scripts/`/`task_description.md` for provenance, but no task is shipped for it.
