"""Score spec for robo-diffusion-sampling-method.

The research question is *efficient* sampling: high D4RL return at as few NFEs
(sampling_steps) as possible. So the score is

    final = sigmoid(normalized_score) * penalty_upper(sampling_steps, target=10)

with the NFE penalty applied per-env as a constraint.

Reference NFEs across baselines:
  - default (DDPM)      : 100 steps
  - ddim                :  20 steps
  - dpm_solver (++ 2M)  :  10 steps   (target = no penalty)

penalty_upper formula: ``exp(-sharpness * (steps - 10))`` for steps > 10.
With ``sharpness = 0.015``:
  - 10 steps  -> 1.000  (full credit)
  - 20 steps  -> 0.861  (modest 14% penalty for 2x cost)
  - 50 steps  -> 0.549  (medium penalty for 5x cost)
  - 100 steps -> 0.259  (heavy penalty for 10x cost)

D4RL normalized_score is return-like and can exceed nominal expert scale, so
sigmoid normalization is used.
"""
from mlsbench.scoring.dsl import *

# ----- Per-env quality terms (sigmoid on normalized_score) -----
term("hopper_quality",
    col("hopper_normalized_score").higher().id().sigmoid())
term("walker2d_quality",
    col("walker2d_normalized_score").higher().id().sigmoid())
term("halfcheetah_quality",
    col("halfcheetah_normalized_score").higher().id().sigmoid())

# ----- Per-env NFE penalty constraints -----
# target = 10 (DPM-Solver++ floor); penalty kicks in beyond that.
NFE_TARGET = 10.0
NFE_SHARPNESS = 0.015

term("hopper_nfe",
    penalty_upper(col("hopper_sampling_steps"), target=NFE_TARGET, sharpness=NFE_SHARPNESS))
term("walker2d_nfe",
    penalty_upper(col("walker2d_sampling_steps"), target=NFE_TARGET, sharpness=NFE_SHARPNESS))
term("halfcheetah_nfe",
    penalty_upper(col("halfcheetah_sampling_steps"), target=NFE_TARGET, sharpness=NFE_SHARPNESS))

# ----- One setting per env so the NFE penalty multiplies env-wise -----
setting("hopper",
    weighted_mean(("hopper_quality", 1.0)),
    constraints=["hopper_nfe"])
setting("walker2d",
    weighted_mean(("walker2d_quality", 1.0)),
    constraints=["walker2d_nfe"])
setting("halfcheetah",
    weighted_mean(("halfcheetah_quality", 1.0)),
    constraints=["halfcheetah_nfe"])

# ----- Task aggregation: gmean across envs -----
task(gmean("hopper", "walker2d", "halfcheetah"))
