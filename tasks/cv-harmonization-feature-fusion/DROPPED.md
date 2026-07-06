# DROPPED surface: cv-harmonization-feature-fusion (NOT shipped)

## Research question (from task_description.md)
Encoder->Decoder Skip Connections (Feature Fusion): does re-injecting encoder detail into
the decoder via U-Net skip connections ('skip') sharpen the foreground recolour vs a
no-skip bottleneck-only decoder ('noskip')?

This editable surface (U-Net skip connections vs no skips) was re-anchored on REAL
iHarmony4 composites (Cong et al., DoveNet, CVPR 2020; mild=HCOCO, medium=Hday2night,
strong=HFlickr) via a full GPU cross-seed (42/123) sweep
(vendor/image-harmonization/run_anchors.py), but is NOT robustly monotone (strong=skip
beats weak=noskip on every severity, every seed) at the package-default 500-iter budget.

Foreground PSNR (dB, higher better), weak=noskip, strong=skip, at 500 iters:
  mild    seed42:  strong=18.371  weak=19.579  [FAIL]
  mild    seed123: strong=17.269  weak=15.887  [OK]
  medium  seed42:  strong=18.962  weak=17.110  [OK]
  medium  seed123: strong=15.768  weak=16.611  [FAIL]
  strong  seed42:  strong=17.662  weak=12.896  [OK]
  strong  seed123: strong=15.883  weak=15.160  [OK]
  ==> 4/6 cells clean at 500 iters.

A training-budget diagnostic at 2000 iters (4x longer, NOT an HP-sweep -- a single
independent longer-budget rerun, same seeds/harness) does not resolve this cleanly:

Foreground PSNR (dB, higher better), weak=noskip, strong=skip, at 2000 iters:
  mild    seed42:  strong=20.055  weak=19.624  [OK]   (was FAIL at 500 iters)
  mild    seed123: strong=19.211  weak=19.342  [FAIL] (was OK at 500 iters)
  medium  seed42:  strong=21.366  weak=19.537  [OK]
  medium  seed123: strong=18.348  weak=19.168  [FAIL] (still FAIL at 500 iters)
  strong  seed42:  strong=20.425  weak=19.952  [OK]
  strong  seed123: strong=19.186  weak=16.663  [OK]
  ==> 4/6 cells clean at 2000 iters.

The failure at 'mild' RELOCATES from seed42 (500 iters) to seed123 (2000 iters) rather
than shrinking to zero, and the failure at 'medium'/seed123 PERSISTS unchanged at both
budgets. This rules out an undertraining artifact and instead indicates that, on this
64x64 real-iHarmony4 operating point, the skip-connection margin is small enough (both
baselines are already close to their PSNR ceiling once past ~500 iters) that its sign is
dominated by run-to-run seed variance rather than a genuine, stable ordering.

Per the project's never-HP-sweep mandate, this surface is dropped honestly rather than
shipped with a forced ordering. The surface code remains in
vendor/image-harmonization/solution/fusion.py + this task's edits/scripts/
task_description.md/leaderboard.csv (updated to the real 500-iter seed-42 measurements)
for provenance; no task is shipped for this surface (config.json baselines/test_cmds are
emptied, score_spec.py is a stub). See cv-harmonization-region-norm,
cv-harmonization-mask-conditioning, cv-harmonization-loss-region,
cv-harmonization-activation, and cv-harmonization-input-norm for the five shipped,
robustly cross-seed-monotone-at-500-iters harmonization surfaces on this real data.

Full per-anchor-line provenance:
  - seed-42/123, 500-iter re-anchor: GPU sweep on k1 H20 (torch 2.4.0), task=
    cv-harmonization-feature-fusion rows in anchor_real_full.tsv.
  - seed-42/123, 2000-iter diagnostic: GPU sweep on k1 H20 (torch 2.4.0), rows in
    anchor_diag_fusion_2000.tsv.
