"""Score spec for optimization-variance-reduction."""
import math

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
#
# logistic best_test_accuracy: fixed sigmoid scale. The baselines span only
# 0.07 pt (storm 92.55, svrg 92.58, storm_plus 92.62), below the run-to-run
# noise: re-running the three baselines at seeds 42/123/456 gives a pooled
# within-baseline std of 0.073 pt (storm_plus alone 92.62/92.47/92.43).
# Calibrating the term on that spread (best baseline = 0.5) made it a step
# function of seed noise. The floor stays the weakest baseline; the scale is
# fixed so that +0.3 pt over it (about 4 std) scores 0.5
# (2 * sigmoid(gain / scale) - 1 = 0.5, i.e. scale = gain / ln 3). The final
# accuracy spans 0.23 pt (about 5 of its 0.043 pt std) and keeps the default.
SCALE_LOGISTIC_BEST_03PT = 0.3 / math.log(3.0)

term("best_test_accuracy_logistic",
    col("best_test_accuracy_logistic").higher().id()
    .sigmoid(scale=SCALE_LOGISTIC_BEST_03PT))

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
