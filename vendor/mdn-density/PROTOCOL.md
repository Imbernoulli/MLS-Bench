# MDN Conditional-Density Protocol

This repository uses a custom, fixed conditional-density benchmark. It does not
claim to reproduce a paper-scale training recipe or four canonical community
datasets.

- `inverse_sine` adapts the inverse problem in Bishop, *Mixture Density
  Networks* (1994), Figure 5.
- `two_branch`, `spiral`, and `rot_bimodal` are local synthetic extensions used
  to exercise bimodality, input-dependent multimodality, and full covariance.
- Every active setting uses 20,000 frozen training rows, 20,000 verifier-only
  test rows, seed 42, batch size 512, and 4,000 Adam updates.
- This is 2.048 million sampled training presentations, about 102.4 passes over
  the 20,000-row inventory. That arithmetic is the scale claim; it is not
  described as uniformly more compute than public full-batch tutorials.
- Train and test archives are bound by byte-level SHA-256 contracts. Trusted
  harness code emits the hashes, counts, seed, budget, completion, and metrics.
- Agent source is parsed as a bounded JSON literal AST and is never executed.
- Every proof binds the exact task, surface, target, seed, counts, budget, and
  data hashes. The `mdn-component-balance` sibling changes only a bounded
  minibatch component-usage regularization coefficient; scoring still uses the
  same unregularized held-out mixture NLL.

The representative `mdn-density-bench` task evaluates one density-family choice
on `inverse_sine`, `two_branch`, and `spiral`. Other task-specific surfaces stay
unanchored and naturally score zero until their own final-protocol baselines are
measured; they contain no explicit score override.

## Representative Evidence

Mangrove task `96377` / container `4927284` ran the native point-density choice
on one H20. All three settings reached update 3,999 and reproduced native NLL
`-0.073271 / 1.578776 / -0.084000`. Environment setup was 49.541 seconds,
verification 36.854 seconds, and trial wall time 87.068 seconds.

The pre-repair strong recipe was measured on H20 worker
`dev-rf4vk-2696400-worker-0`: K=5 softplus mixture, 4,000 updates per setting,
NLL `-1.169754 / -1.023576 / -1.806954`, and per-setting elapsed time
`11.9 / 12.4 / 12.0` seconds. The archived raw log SHA-256 is
`84431f92f9f8289ba7d2a3c4c6d135fcc4d42b75c4d4addb8b95ab00d8c7d0e9`;
this is numerical recipe evidence, not a claim that the final
rendered verifier image has already been replayed.
