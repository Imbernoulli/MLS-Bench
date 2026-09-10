"""Score spec for ai4bio-protein-structure-repr."""
from mlsbench.scoring.dsl import *

# accuracy_EC and accuracy_Fold are on [0, 1] scale (values like 0.78, 0.33)
term("accuracy_EC",
    col("accuracy_EC").higher().id()
    .bounded_power(bound=1.0))

term("test_loss_EC",
    col("test_loss_EC").lower().id()
    .bounded_power(bound=0.0))

term("f1_max_GO_BP",
    col("f1_max_GO-BP").higher().id()
    .bounded_power(bound=1.0))

term("test_loss_GO_BP",
    col("test_loss_GO-BP").lower().id()
    .bounded_power(bound=0.0))

term("accuracy_Fold",
    col("accuracy_Fold").higher().id()
    .bounded_power(bound=1.0))

term("test_loss_Fold",
    col("test_loss_Fold").lower().id()
    .bounded_power(bound=0.0))

setting("EC", weighted_mean(("accuracy_EC", 1.0), ("test_loss_EC", 1.0)))
setting("GO-BP", weighted_mean(("f1_max_GO_BP", 1.0), ("test_loss_GO_BP", 1.0)))
setting("Fold", weighted_mean(("accuracy_Fold", 1.0), ("test_loss_Fold", 1.0)))

task(gmean("EC", "GO-BP", "Fold"))
