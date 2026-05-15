"""Score spec for jepa-regularizer (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("val_acc_resnet18",
    col("val_acc_resnet18").higher().id()
    .bounded_power(bound=100.0))

term("val_acc_resnet34",
    col("val_acc_resnet34").higher().id()
    .bounded_power(bound=100.0))

term("val_acc_resnet50",
    col("val_acc_resnet50").higher().id()
    .bounded_power(bound=100.0))

setting("resnet18", weighted_mean(("val_acc_resnet18", 1.0)))
setting("resnet34", weighted_mean(("val_acc_resnet34", 1.0)))
setting("resnet50", weighted_mean(("val_acc_resnet50", 1.0)))

task(gmean("resnet18", "resnet34", "resnet50"))
