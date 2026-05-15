"""Score spec for optimization-multi-objective.

Multi-objective optimization task with 4 problem settings: zdt1, zdt3, dtlz2, dtlz1.
Each setting has three metrics:
  - HV (hypervolume): higher is better — unbounded, sigmoid
  - IGD (inverted generational distance): lower is better, optimum 0
  - Spread: lower is better, optimum 0
"""
from mlsbench.scoring.dsl import *

# ---- ZDT1 ----
term("hv_zdt1",
    col("hv_zdt1").higher().id()
    .sigmoid())
term("igd_zdt1",
    col("igd_zdt1").lower().id()
    .bounded_power(bound=0.0))
term("spread_zdt1",
    col("spread_zdt1").lower().id()
    .bounded_power(bound=0.0))

setting("zdt1", weighted_mean(
    ("hv_zdt1", 1.0), ("igd_zdt1", 1.0), ("spread_zdt1", 1.0)))

# ---- ZDT3 ----
term("hv_zdt3",
    col("hv_zdt3").higher().id()
    .sigmoid())
term("igd_zdt3",
    col("igd_zdt3").lower().id()
    .bounded_power(bound=0.0))
term("spread_zdt3",
    col("spread_zdt3").lower().id()
    .bounded_power(bound=0.0))

setting("zdt3", weighted_mean(
    ("hv_zdt3", 1.0), ("igd_zdt3", 1.0), ("spread_zdt3", 1.0)))

# ---- DTLZ2 ----
term("hv_dtlz2",
    col("hv_dtlz2").higher().id()
    .sigmoid())
term("igd_dtlz2",
    col("igd_dtlz2").lower().id()
    .bounded_power(bound=0.0))
term("spread_dtlz2",
    col("spread_dtlz2").lower().id()
    .bounded_power(bound=0.0))

setting("dtlz2", weighted_mean(
    ("hv_dtlz2", 1.0), ("igd_dtlz2", 1.0), ("spread_dtlz2", 1.0)))

# ---- DTLZ1 ----
term("hv_dtlz1",
    col("hv_dtlz1").higher().id()
    .sigmoid())
term("igd_dtlz1",
    col("igd_dtlz1").lower().id()
    .bounded_power(bound=0.0))
term("spread_dtlz1",
    col("spread_dtlz1").lower().id()
    .bounded_power(bound=0.0))

setting("dtlz1", weighted_mean(
    ("hv_dtlz1", 1.0), ("igd_dtlz1", 1.0), ("spread_dtlz1", 1.0)))

# Task: geometric mean across all 4 problem settings
task(gmean("zdt1", "zdt3", "dtlz2", "dtlz1"))
