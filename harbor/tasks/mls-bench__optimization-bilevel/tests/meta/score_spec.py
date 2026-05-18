"""Score spec for optimization-bilevel."""
from mlsbench.scoring.dsl import *

# toy-convergence: lower steps/residual = faster convergence; success_rate higher is better
# hyperclean-linear/mlp: test_accuracy on 0-100 scale; f1 on 0-100; precision/recall on 0-1
# refs from best baseline (g_pbgd for toy, rhg/g_pbgd for hyperclean) means
# Dropped: runtime_sec, total_runtime_sec, best_step (informational)
# Dropped: best_accuracy_*/best_f1_* prefixed columns (redundant snapshots)
# Dropped: score_* aggregate fields (redundant with individual metrics)

term("convergence_steps_toy_convergence",
    col("convergence_steps_toy_convergence").lower().id()
    .sigmoid())

term("final_residual_toy_convergence",
    col("final_residual_toy_convergence").lower().id()
    .sigmoid())

term("final_projected_grad_toy_convergence",
    col("final_projected_grad_toy_convergence").lower().id()
    .sigmoid())

term("success_rate_toy_convergence",
    col("success_rate_toy_convergence").higher().id()
    .bounded_power(bound=1.0))

term("test_accuracy_hyperclean_linear",
    col("test_accuracy_hyperclean_linear").higher().id()
    .bounded_power(bound=100.0))

term("f1_score_hyperclean_linear",
    col("f1_score_hyperclean_linear").higher().id()
    .bounded_power(bound=100.0))

term("cleaner_precision_hyperclean_linear",
    col("cleaner_precision_hyperclean_linear").higher().id()
    .bounded_power(bound=1.0))

term("cleaner_recall_hyperclean_linear",
    col("cleaner_recall_hyperclean_linear").higher().id()
    .bounded_power(bound=1.0))

term("test_accuracy_hyperclean_mlp",
    col("test_accuracy_hyperclean_mlp").higher().id()
    .bounded_power(bound=100.0))

term("f1_score_hyperclean_mlp",
    col("f1_score_hyperclean_mlp").higher().id()
    .bounded_power(bound=100.0))

term("cleaner_precision_hyperclean_mlp",
    col("cleaner_precision_hyperclean_mlp").higher().id()
    .bounded_power(bound=1.0))

term("cleaner_recall_hyperclean_mlp",
    col("cleaner_recall_hyperclean_mlp").higher().id()
    .bounded_power(bound=1.0))

setting("toy-convergence", weighted_mean(
    ("convergence_steps_toy_convergence", 1.0),
    ("final_residual_toy_convergence", 1.0),
    ("final_projected_grad_toy_convergence", 1.0),
    ("success_rate_toy_convergence", 1.0),
))
setting("hyperclean-linear", weighted_mean(
    ("test_accuracy_hyperclean_linear", 1.0),
    ("f1_score_hyperclean_linear", 1.0),
    ("cleaner_precision_hyperclean_linear", 1.0),
    ("cleaner_recall_hyperclean_linear", 1.0),
))
setting("hyperclean-mlp", weighted_mean(
    ("test_accuracy_hyperclean_mlp", 1.0),
    ("f1_score_hyperclean_mlp", 1.0),
    ("cleaner_precision_hyperclean_mlp", 1.0),
    ("cleaner_recall_hyperclean_mlp", 1.0),
))

task(gmean("toy-convergence", "hyperclean-linear", "hyperclean-mlp"))
