"""Score spec for optimization-variance-reduction."""
from mlsbench.scoring.dsl import *

# total_grad_comps is informational (fixed budget per method) -> dropped
# final_test_mse_conditioned ref was corrupted (3.6e34) -> use best_test_mse ref instead

term("best_test_accuracy_logistic",
    col("best_test_accuracy_logistic").higher().id()
    .bounded_power(bound=100.0))

term("final_test_accuracy_logistic",
    col("final_test_accuracy_logistic").higher().id()
    .bounded_power(bound=100.0))

term("best_test_accuracy_mlp",
    col("best_test_accuracy_mlp").higher().id()
    .bounded_power(bound=100.0))

term("final_test_accuracy_mlp",
    col("final_test_accuracy_mlp").higher().id()
    .bounded_power(bound=100.0))

term("best_test_mse_conditioned",
    col("best_test_mse_conditioned").lower().id()
    .bounded_power(bound=0.0))

term("final_test_mse_conditioned",
    col("final_test_mse_conditioned").lower().id()
    .bounded_power(bound=0.0))

setting("logistic", weighted_mean(("best_test_accuracy_logistic", 1.0), ("final_test_accuracy_logistic", 1.0)))
setting("mlp", weighted_mean(("best_test_accuracy_mlp", 1.0), ("final_test_accuracy_mlp", 1.0)))
setting("conditioned", weighted_mean(("best_test_mse_conditioned", 1.0), ("final_test_mse_conditioned", 1.0)))

task(gmean("logistic", "mlp", "conditioned"))
