# DROPPED surface: stereo-loss-design (NOT shipped)

This editable surface (squared-L2 vs smooth-L1 disparity regression loss, RQ:
"what regression loss minimizes EPE for a FIXED small GC-Net-style stereo
net?") was DESIGNED and GPU-anchored on the OLD SYNTHETIC stereo data (see git
history for the previous leaderboard.csv/score_spec.py that shipped it,
weak(l2)>strong(smooth_l1) on all 3 settings). After the data pipeline was
swapped to REAL rectified stereo photographs from the Middlebury Stereo
Datasets 2005/2006 (structured-light ground-truth disparity), a full GPU
re-anchor (k1 H20, NVIDIA H20, torch 2.4.0, seed 42, package-default 1200
steps, 20-pair val set) showed the `medium` setting INVERTS:

  seed 42 (1200 steps): easy   l2 4.011 > smooth_l1 3.677   (weak>strong, OK)
                         medium l2 10.688 < smooth_l1 12.070 (weak<strong, FAIL)
                         hard   l2 7.453 > smooth_l1 5.441   (weak>strong, OK)

Per the same diagnostic-training-budget methodology used to distinguish
undertraining artifacts from genuine confounds elsewhere in this repo (see
`vendor/stereo-matching/anchors/README.md` -- e.g. `aggregation`/`refine`
resolved cleanly at a longer 3000-step schedule), a diagnostic re-run of
`loss` and `refine` at 3000 steps was launched to test whether the `medium`
inversion was a training-budget artifact:

  seed 42 (3000 steps): easy   l2 3.125 > smooth_l1 3.072   (weak>strong, OK)
                         medium l2 9.784 < smooth_l1 12.375  (weak<strong, FAIL --
                                gap WIDENED from -1.38 at 1200 steps to -2.59
                                at 3000 steps)
                         hard   l2 6.207 > smooth_l1 4.262   (weak>strong, OK)

The `medium`-setting inversion PERSISTS AND WORSENS with more training budget
(2x the steps roughly doubles the gap in the wrong direction), unlike
`aggregation`/`refine` where the equivalent diagnostic fully resolved the
inversion. This rules out "undertraining artifact" and indicates a genuine
structural confound: squared-L2's large-error-dominated gradient apparently
does something that specifically benefits the `medium` real-scene tercile
(mid-range disparities, ~70px) as training progresses further, while it still
loses (as expected from the RQ's premise) on `easy` and `hard`. Task-level
gmean is therefore NOT monotone weak(l2)>strong(smooth_l1) at either step
count on real Middlebury data.

Reason (real data): the synthetic degradation/geometry the surface was
originally designed and anchored against evidently made squared-L2's
large-error over-weighting uniformly harmful; real Middlebury scenes have a
different, scene-content-dependent distribution of occlusion/discontinuity
pixel fractions across the three real-disparity terciles, and the `medium`
tercile's specific mix apparently rewards squared-L2's mean-disparity bias
rather than penalizing it once the network is given enough steps to actually
exploit that gradient. This is a genuine content/geometry confound between
the synthetic RQ design and the real data, not a training-budget artifact.

Full per-anchor-line provenance:
  - seed-42, 1200-step re-anchor: vendor/stereo-matching/anchors/anchor_real.tsv
    (task=loss rows)
  - seed-42, 3000-step diagnostic: vendor/stereo-matching/anchors/diag_refine_loss_3000.tsv
    (task=loss rows)

The surface code remains in vendor/stereo-matching/solution/loss.py and
vendor/stereo-matching/baselines/loss_{l2,smooth_l1}.py for provenance; no
task is shipped for it.
