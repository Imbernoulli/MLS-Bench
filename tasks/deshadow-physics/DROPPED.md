# DROPPED surface: deshadow-physics (NOT shipped)

This editable surface (free 3-ch residual vs the SP+M-Net affine illumination-inverse
parameterisation `J = w*I + b`) was DESIGNED and GPU-VALIDATED on the OLD SYNTHETIC
near-affine multiplicative cast-shadow data (aggregate gmean monotone strong>weak on both
seeds 42/1 -- see git history for the previous task_description.md / leaderboard.csv /
score_spec.py that shipped it). After the data pipeline was swapped to REAL ISTD (Wang, Li &
Yang, CVPR 2018) shadow/shadow-free/mask photo triplets, a CPU smoke-test re-check (harness's
real `_build_configured`/`psnr_masked` path, model width shrunk BASE=32->14 and iters
reduced purely for CPU tractability -- ordering only, not final numbers) shows the ordering
INVERTS on the harder settings:

  seed 42  (BASE=14, 30 iters, val cap 100):
    light  : weak(residual)=24.9366 -> strong(physics)=25.0775   (delta +0.14, weak<strong OK)
    medium : weak(residual)=25.5543 -> strong(physics)=25.2451   (delta -0.31, weak>strong FAIL)
    heavy  : weak(residual)=22.6237 -> strong(physics)=22.0870   (delta -0.54, weak>strong FAIL)

  seed 123 (BASE=14, 50 iters, val cap 100):
    light  : weak(residual)=25.6825 -> strong(physics)=25.1707   (delta -0.51, weak>strong FAIL)
    medium : weak(residual)=25.7644 -> strong(physics)=24.7798   (delta -0.98, weak>strong FAIL)
    heavy  : weak(residual)=22.4355 -> strong(physics)=21.6285   (delta -0.81, weak>strong FAIL)

Task-level gmean (light,medium,heavy): seed42 weak=24.338 -> strong=24.092 (delta -0.25,
INVERTED); seed123 weak=24.577 -> strong=23.805 (delta -0.77, INVERTED). Both seeds now
invert at the aggregate level on real ISTD data, unlike the old synthetic data.

Reason (real data): the synthetic degradation was an EXACT linear multiplicative model, so the
physics-constrained affine output fit it perfectly by construction. Real ISTD shadows are not
exactly affine (camera auto-exposure / white-balance / sensor noise differences between the two
captures of the same scene), so the hard constraint `J = w*I + b` is now a slight MISSPECIFIED
prior that a free residual can fit better than the constrained parameterisation, especially on
the harder medium/heavy severities.

Full per-anchor-line provenance: vendor/image-deshadow/anchors/real_istd_cpu_smoke.log
(search for `surface=physics`). The surface code remains in vendor/image-deshadow/harness.py
(SURFACES tuple, `get_physics_config` hook) for provenance; no task is shipped for it.
