"""Score spec for mdn-density-bench (the repo's PRIMARY strict-bar task).

One density-family choice is evaluated on three custom multimodal inverse
targets. The task score is the geometric mean of per-setting logistics on
held-out mixture NLL (nats; lower is better). The native point-density recipe
maps to 0.1 and the measured K=5 softplus-mixture recipe maps to 0.5.

Measured anchors (seed 42, 4000 steps, H20 GPU):
    inverse_sine   native -0.073271 -> mixture -1.169754
    two_branch     native +1.578776 -> mixture -1.023576
    spiral         native -0.084000 -> mixture -1.806954

Native values were reproduced by Mangrove task 96377. Mixture values come from
H20 worker dev-rf4vk-2696400-worker-0 using the same model, data, optimizer,
batch size, and 4,000-step numerical path; the final protocol adds literal-AST
loading and hash-bound trusted terminal proof without changing that numerical
recipe.
"""
from mlsbench.scoring.dsl import *

term("nll_inverse_sine",
    col("nll_inverse_sine").lower().id()
    .sigmoid(ref=const(-1.169754), scale=0.49903091896473717))
term("nll_two_branch",
    col("nll_two_branch").lower().id()
    .sigmoid(ref=const(-1.023576), scale=1.1843814359454017))
term("nll_spiral",
    col("nll_spiral").lower().id()
    .sigmoid(ref=const(-1.806954), scale=0.7841501582368079))

setting("inverse_sine", weighted_mean(("nll_inverse_sine", 1.0)))
setting("two_branch", weighted_mean(("nll_two_branch", 1.0)))
setting("spiral", weighted_mean(("nll_spiral", 1.0)))

task(gmean("inverse_sine", "two_branch", "spiral"))
