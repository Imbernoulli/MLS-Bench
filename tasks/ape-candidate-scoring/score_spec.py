"""Measured calibration for the full official APE candidate-scoring protocol.

Frozen Qwen2.5-0.5B-Instruct; train-only proposal pool 128 and selection set
200; exactly one selected instruction evaluated on all 7,600 official AG News
test rows and all 872 labeled SST-2 validation rows. Both settings participate.

Measured on one NVIDIA H20, seed 42, worker dev-bpqbw-4911-worker-0:
    agnews: weak 0.628421053 -> strong 0.783157895
    sst2:   weak 0.816513761 -> strong 0.887614679

For each setting, sigmoid(ref=strong, scale=(strong-weak)/ln(9)) maps the
measured weak baseline to 0.1 and measured strong baseline to 0.5. There is no
hardcoded positive reward and no scoring floor.
"""
from mlsbench.scoring.dsl import *

term(
    "test_acc_agnews",
    col("test_acc_agnews").higher().id().sigmoid(
        ref=const(0.783157895), scale=0.07042377169637958
    ),
)
term(
    "test_acc_sst2",
    col("test_acc_sst2").higher().id().sigmoid(
        ref=const(0.887614679), scale=0.03235942230638912
    ),
)

setting("agnews", weighted_mean(("test_acc_agnews", 1.0)))
setting("sst2", weighted_mean(("test_acc_sst2", 1.0)))
task(gmean("agnews", "sst2"))
