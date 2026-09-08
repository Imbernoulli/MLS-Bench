"""Score spec for llm-scaling-law-discovery.

Scaling law discovery task: predict LLM performance from compute/data/model
parameters. Three harder dataset settings recommended by the SLDBench
authors: sld-vocab, sld-lrbsz, sld-dataconstrained.

Each setting has four metrics:
  - r2: higher is better, bounded above by 1.0 (can be negative), bounded_power
  - mae: lower is better, bounded at 0, bounded_power
  - rmse: lower is better, bounded at 0, bounded_power
  - nmae: lower is better, bounded at 0, bounded_power

Reference baselines (symbolic, fit per-group) on seed=42 — rough numbers from
an initial human_exact / sldagent_style dry run; refresh after a full baseline
sweep:
  sld-vocab:             r2 ~ 0.928, mae ~ 0.169, rmse ~ 0.226, nmae ~ 0.200
  sld-lrbsz:             r2 often < 0 on held-out split (hard!); use mae/rmse
                         as primary diagnostic. Refs below are conservative.
  sld-dataconstrained:   r2 ~ 0.93 (best symbolic), mae ~ 0.13, rmse ~ 0.15,
                         nmae ~ 0.24

r2 uses bounded_power with bound=1.0 (theoretical maximum). For r2,
higher is better and bound is the best possible value, so the transform
maps improvement toward bound=1.0.
"""
from mlsbench.scoring.dsl import *

# ---- sld-vocab ----
term("r2_vocab",
    col("r2_sld_vocab").higher().id()
    .bounded_power(bound=1.0))
term("mae_vocab",
    col("mae_sld_vocab").lower().id()
    .bounded_power(bound=0.0))
term("rmse_vocab",
    col("rmse_sld_vocab").lower().id()
    .bounded_power(bound=0.0))
term("nmae_vocab",
    col("nmae_sld_vocab").lower().id()
    .bounded_power(bound=0.0))

setting("sld-vocab", weighted_mean(
    ("r2_vocab", 2.0),
    ("mae_vocab", 1.0),
    ("rmse_vocab", 1.0),
    ("nmae_vocab", 1.0),
))

# ---- sld-lrbsz ----
term("r2_lrbsz",
    col("r2_sld_lrbsz").higher().id()
    .bounded_power(bound=1.0))
term("mae_lrbsz",
    col("mae_sld_lrbsz").lower().id()
    .bounded_power(bound=0.0))
term("rmse_lrbsz",
    col("rmse_sld_lrbsz").lower().id()
    .bounded_power(bound=0.0))
term("nmae_lrbsz",
    col("nmae_sld_lrbsz").lower().id()
    .bounded_power(bound=0.0))

setting("sld-lrbsz", weighted_mean(
    ("r2_lrbsz", 2.0),
    ("mae_lrbsz", 1.0),
    ("rmse_lrbsz", 1.0),
    ("nmae_lrbsz", 1.0),
))

# ---- sld-dataconstrained ----
term("r2_dataconstrained",
    col("r2_sld_dataconstrained").higher().id()
    .bounded_power(bound=1.0))
term("mae_dataconstrained",
    col("mae_sld_dataconstrained").lower().id()
    .bounded_power(bound=0.0))
term("rmse_dataconstrained",
    col("rmse_sld_dataconstrained").lower().id()
    .bounded_power(bound=0.0))
term("nmae_dataconstrained",
    col("nmae_sld_dataconstrained").lower().id()
    .bounded_power(bound=0.0))

setting("sld-dataconstrained", weighted_mean(
    ("r2_dataconstrained", 2.0),
    ("mae_dataconstrained", 1.0),
    ("rmse_dataconstrained", 1.0),
    ("nmae_dataconstrained", 1.0),
))

# Task: geometric mean across scaling law datasets
task(gmean("sld-vocab", "sld-lrbsz", "sld-dataconstrained"))
