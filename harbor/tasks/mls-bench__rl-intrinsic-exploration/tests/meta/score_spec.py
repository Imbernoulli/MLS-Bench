"""Score spec for rl-intrinsic-exploration."""
from mlsbench.scoring.dsl import *

# auc = area under return curve: higher better, unbounded -> sigmoid (NOT bounded_power)
# nonzero_rate: bounded [0,1] -> bounded_power is correct
# eval_return, best_eval_return: higher better, unbounded -> sigmoid

term("eval_return_frostbite_v5",
    col("eval_return_frostbite_v5").higher().id()
    .sigmoid())

term("auc_frostbite_v5",
    col("auc_frostbite_v5").higher().id()
    .sigmoid())

term("nonzero_rate_frostbite_v5",
    col("nonzero_rate_frostbite_v5").higher().id()
    .bounded_power(bound=1.0))

term("best_eval_return_frostbite_v5",
    col("best_eval_return_frostbite_v5").higher().id()
    .sigmoid())

term("eval_return_private_eye_v5",
    col("eval_return_private_eye_v5").higher().id()
    .sigmoid())

term("auc_private_eye_v5",
    col("auc_private_eye_v5").higher().id()
    .sigmoid())

term("nonzero_rate_private_eye_v5",
    col("nonzero_rate_private_eye_v5").higher().id()
    .bounded_power(bound=1.0))

term("best_eval_return_private_eye_v5",
    col("best_eval_return_private_eye_v5").higher().id()
    .sigmoid())

term("eval_return_tutankham_v5",
    col("eval_return_tutankham_v5").higher().id()
    .sigmoid())

term("auc_tutankham_v5",
    col("auc_tutankham_v5").higher().id()
    .sigmoid())

term("nonzero_rate_tutankham_v5",
    col("nonzero_rate_tutankham_v5").higher().id()
    .bounded_power(bound=1.0))

term("best_eval_return_tutankham_v5",
    col("best_eval_return_tutankham_v5").higher().id()
    .sigmoid())

setting("frostbite-v5", weighted_mean(("eval_return_frostbite_v5", 1.0), ("auc_frostbite_v5", 1.0), ("nonzero_rate_frostbite_v5", 1.0), ("best_eval_return_frostbite_v5", 1.0)))
setting("private-eye-v5", weighted_mean(("eval_return_private_eye_v5", 1.0), ("auc_private_eye_v5", 1.0), ("nonzero_rate_private_eye_v5", 1.0), ("best_eval_return_private_eye_v5", 1.0)))
setting("tutankham-v5", weighted_mean(("eval_return_tutankham_v5", 1.0), ("auc_tutankham_v5", 1.0), ("nonzero_rate_tutankham_v5", 1.0), ("best_eval_return_tutankham_v5", 1.0)))

task(gmean("tutankham-v5", "frostbite-v5", "private-eye-v5"))
