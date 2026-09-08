"""Score spec for cv-vae-loss.

Per-size best_rfid only (lower is better, bound=0). PSNR/SSIM dropped
because agents and baselines did not produce consistent per-size data.
"""
from mlsbench.scoring.dsl import *

term("best_rfid_small",
    col("best_rfid_small").lower().id()
    .bounded_power(bound=0.0))

term("best_rfid_medium",
    col("best_rfid_medium").lower().id()
    .bounded_power(bound=0.0))

term("best_rfid_large",
    col("best_rfid_large").lower().id()
    .bounded_power(bound=0.0))

setting("train_small",  weighted_mean(("best_rfid_small", 1.0)))
setting("train_medium", weighted_mean(("best_rfid_medium", 1.0)))
setting("train_large",  weighted_mean(("best_rfid_large", 1.0)))

task(gmean("train_small", "train_medium", "train_large"))
