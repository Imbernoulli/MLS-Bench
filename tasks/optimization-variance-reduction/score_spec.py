"""Score spec for optimization-variance-reduction."""
from mlsbench.scoring.dsl import *

# total_grad_comps is informational (fixed budget per method) -> dropped
#
# conditioned (test MSE, lower is better): log scale, sigmoid calibrated so the
# best baseline scores 0.5. The floor is the worst baseline, and svrg diverges
# on this problem (best_test_mse 2582.67, final_test_mse 8.5e10, against
# storm 3.5 / storm_plus 0.0147). On the raw scale that floor made the term
# blind: any run that does not diverge scored ~0.5 whether its MSE was 0.01 or
# 100 (storm 3.9 and storm_plus 0.0196 both scored 0.500). MSE spans orders of
# magnitude between converged and stalled optimizers, so the log is the natural
# unit; the Bayes floor of the task's data (noise std 0.1 -> MSE 0.01) keeps
# log(MSE) bounded above in practice, and storm_plus already sits near it.

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
    col("best_test_mse_conditioned").lower().log()
    .sigmoid())

term("final_test_mse_conditioned",
    col("final_test_mse_conditioned").lower().log()
    .sigmoid())

setting("logistic", weighted_mean(("best_test_accuracy_logistic", 1.0), ("final_test_accuracy_logistic", 1.0)))
setting("mlp", weighted_mean(("best_test_accuracy_mlp", 1.0), ("final_test_accuracy_mlp", 1.0)))
setting("conditioned", weighted_mean(("best_test_mse_conditioned", 1.0), ("final_test_mse_conditioned", 1.0)))

task(gmean("logistic", "mlp", "conditioned"))
