"""Score spec for optimization-convex-concave."""
from mlsbench.scoring.dsl import *

# final_gradient_norm: lower is better (convergence to 0)
# bilinear_final_gradient_norm: lower is better
# delta_nu_final_gradient_norm: lower is better
# auc_log_iteration_log_grad: lower is better (less area = faster convergence)
# score: higher is better (summary score)
# num_runs: informational count — dropped
# refs from best baseline means

term("final_gradient_norm_default_noise",
    col("final_gradient_norm_default-noise").lower().id()
    .sigmoid())

term("score_default_noise",
    col("score_default-noise").higher().id()
    .sigmoid())

term("auc_log_iteration_log_grad_default_noise",
    col("auc_log_iteration_log_grad_default-noise").lower().id()
    .sigmoid())

term("bilinear_final_gradient_norm_default_noise",
    col("bilinear_final_gradient_norm_default-noise").lower().id()
    .sigmoid())

term("delta_nu_final_gradient_norm_default_noise",
    col("delta_nu_final_gradient_norm_default-noise").lower().id()
    .sigmoid())

term("final_gradient_norm_low_noise",
    col("final_gradient_norm_low-noise").lower().id()
    .sigmoid())

term("score_low_noise",
    col("score_low-noise").higher().id()
    .sigmoid())

term("auc_log_iteration_log_grad_low_noise",
    col("auc_log_iteration_log_grad_low-noise").lower().id()
    .sigmoid())

term("bilinear_final_gradient_norm_low_noise",
    col("bilinear_final_gradient_norm_low-noise").lower().id()
    .sigmoid())

term("delta_nu_final_gradient_norm_low_noise",
    col("delta_nu_final_gradient_norm_low-noise").lower().id()
    .sigmoid())

term("final_gradient_norm_high_noise",
    col("final_gradient_norm_high-noise").lower().id()
    .sigmoid())

term("score_high_noise",
    col("score_high-noise").higher().id()
    .sigmoid())

term("auc_log_iteration_log_grad_high_noise",
    col("auc_log_iteration_log_grad_high-noise").lower().id()
    .sigmoid())

term("bilinear_final_gradient_norm_high_noise",
    col("bilinear_final_gradient_norm_high-noise").lower().id()
    .sigmoid())

term("delta_nu_final_gradient_norm_high_noise",
    col("delta_nu_final_gradient_norm_high-noise").lower().id()
    .sigmoid())

setting("default-noise", weighted_mean(
    ("final_gradient_norm_default_noise", 1.0),
    ("score_default_noise", 1.0),
    ("auc_log_iteration_log_grad_default_noise", 1.0),
    ("bilinear_final_gradient_norm_default_noise", 1.0),
    ("delta_nu_final_gradient_norm_default_noise", 1.0),
))
setting("low-noise", weighted_mean(
    ("final_gradient_norm_low_noise", 1.0),
    ("score_low_noise", 1.0),
    ("auc_log_iteration_log_grad_low_noise", 1.0),
    ("bilinear_final_gradient_norm_low_noise", 1.0),
    ("delta_nu_final_gradient_norm_low_noise", 1.0),
))
setting("high-noise", weighted_mean(
    ("final_gradient_norm_high_noise", 1.0),
    ("score_high_noise", 1.0),
    ("auc_log_iteration_log_grad_high_noise", 1.0),
    ("bilinear_final_gradient_norm_high_noise", 1.0),
    ("delta_nu_final_gradient_norm_high_noise", 1.0),
))

task(gmean("default-noise", "low-noise", "high-noise"))
