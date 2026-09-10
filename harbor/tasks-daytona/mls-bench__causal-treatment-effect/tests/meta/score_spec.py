"""Score spec for causal-treatment-effect.

Causal inference task estimating conditional average treatment effects (CATE).
Three synthetic DGP settings: ihdp_synth, jobs_synth, acic_synth. Each has two metrics:
  - PEHE (Precision in Estimation of Heterogeneous Effects): lower is better, bounded at 0
  - ATE_error (Average Treatment Effect error): lower is better, bounded at 0

Best baselines (mean across seeds):
  ihdp_synth:  PEHE: causal_forest 0.771, s_learner 0.803; ATE_error: r_learner 0.071
  jobs_synth:  PEHE: causal_forest 358.6, r_learner 476.0; ATE_error: t_learner 35.3
  acic_synth:  PEHE: r_learner 0.428, causal_forest 0.499; ATE_error: r_learner 0.021

ref values set near best baseline for each metric.
"""
from mlsbench.scoring.dsl import *

# ---- IHDP-inspired synthetic DGP ----
term("pehe_ihdp_synth",
    col("PEHE_ihdp_synth").lower().id()
    .bounded_power(bound=0.0))
term("ate_ihdp_synth",
    col("ATE_error_ihdp_synth").lower().id()
    .bounded_power(bound=0.0))

setting("ihdp_synth", weighted_mean(
    ("pehe_ihdp_synth", 1.0), ("ate_ihdp_synth", 1.0)))

# ---- Jobs/LaLonde-inspired synthetic DGP ----
term("pehe_jobs_synth",
    col("PEHE_jobs_synth").lower().log()
    .bounded_power(bound=0.0))
term("ate_jobs_synth",
    col("ATE_error_jobs_synth").lower().log()
    .bounded_power(bound=0.0))

setting("jobs_synth", weighted_mean(
    ("pehe_jobs_synth", 1.0), ("ate_jobs_synth", 1.0)))

# ---- ACIC-inspired synthetic DGP ----
term("pehe_acic_synth",
    col("PEHE_acic_synth").lower().id()
    .bounded_power(bound=0.0))
term("ate_acic_synth",
    col("ATE_error_acic_synth").lower().id()
    .bounded_power(bound=0.0))

setting("acic_synth", weighted_mean(
    ("pehe_acic_synth", 1.0), ("ate_acic_synth", 1.0)))

# Task: geometric mean across dataset settings
task(gmean("ihdp_synth", "jobs_synth", "acic_synth"))
