"""Score spec for graph-graph-classification."""
from mlsbench.scoring.dsl import *

# test_acc is on [0, 100] scale (percentage); macro_f1 is also on [0, 100] scale here
term("test_acc_MUTAG",
    col("test_acc_MUTAG").higher().id()
    .bounded_power(bound=100.0))

term("macro_f1_MUTAG",
    col("macro_f1_MUTAG").higher().id()
    .bounded_power(bound=100.0))

term("test_acc_PROTEINS",
    col("test_acc_PROTEINS").higher().id()
    .bounded_power(bound=100.0))

term("macro_f1_PROTEINS",
    col("macro_f1_PROTEINS").higher().id()
    .bounded_power(bound=100.0))

term("test_acc_NCI1",
    col("test_acc_NCI1").higher().id()
    .bounded_power(bound=100.0))

term("macro_f1_NCI1",
    col("macro_f1_NCI1").higher().id()
    .bounded_power(bound=100.0))

setting("MUTAG", weighted_mean(("test_acc_MUTAG", 1.0), ("macro_f1_MUTAG", 1.0)))
setting("PROTEINS", weighted_mean(("test_acc_PROTEINS", 1.0), ("macro_f1_PROTEINS", 1.0)))
setting("NCI1", weighted_mean(("test_acc_NCI1", 1.0), ("macro_f1_NCI1", 1.0)))

task(gmean("MUTAG", "PROTEINS", "NCI1"))
