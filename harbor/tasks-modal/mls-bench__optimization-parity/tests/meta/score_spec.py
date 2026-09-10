"""Score spec for optimization-parity."""
from mlsbench.scoring.dsl import *

# score_* is the primary metric (higher better, unbounded -> sigmoid)
# test_accuracy_* is secondary (higher better, bounded [0,1])
# test_accuracy_std, mean_steps, num_runs are informational -> dropped

term("score_n32_k8",
    col("score_n32-k8").higher().id()
    .sigmoid())

term("test_accuracy_n32_k8",
    col("test_accuracy_n32-k8").higher().id()
    .bounded_power(bound=1.0))

term("score_n50_k8",
    col("score_n50-k8").higher().id()
    .sigmoid())

term("test_accuracy_n50_k8",
    col("test_accuracy_n50-k8").higher().id()
    .bounded_power(bound=1.0))

term("score_n64_k8",
    col("score_n64-k8").higher().id()
    .sigmoid())

term("test_accuracy_n64_k8",
    col("test_accuracy_n64-k8").higher().id()
    .bounded_power(bound=1.0))

setting("n32-k8", weighted_mean(("score_n32_k8", 1.0), ("test_accuracy_n32_k8", 1.0)))
setting("n50-k8", weighted_mean(("score_n50_k8", 1.0), ("test_accuracy_n50_k8", 1.0)))
setting("n64-k8", weighted_mean(("score_n64_k8", 1.0), ("test_accuracy_n64_k8", 1.0)))

task(gmean("n32-k8", "n50-k8", "n64-k8"))
