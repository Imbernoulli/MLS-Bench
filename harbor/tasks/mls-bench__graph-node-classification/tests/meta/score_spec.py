"""Score spec for graph-node-classification."""
from mlsbench.scoring.dsl import *

# accuracy values are on [0, 1] scale (e.g., 0.826, 0.717, 0.777)
term("accuracy_Cora",
    col("accuracy_Cora").higher().id()
    .bounded_power(bound=1.0))

term("macro_f1_Cora",
    col("macro_f1_Cora").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_CiteSeer",
    col("accuracy_CiteSeer").higher().id()
    .bounded_power(bound=1.0))

term("macro_f1_CiteSeer",
    col("macro_f1_CiteSeer").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_PubMed",
    col("accuracy_PubMed").higher().id()
    .bounded_power(bound=1.0))

term("macro_f1_PubMed",
    col("macro_f1_PubMed").higher().id()
    .bounded_power(bound=1.0))

setting("Cora", weighted_mean(("accuracy_Cora", 1.0), ("macro_f1_Cora", 1.0)))
setting("CiteSeer", weighted_mean(("accuracy_CiteSeer", 1.0), ("macro_f1_CiteSeer", 1.0)))
setting("PubMed", weighted_mean(("accuracy_PubMed", 1.0), ("macro_f1_PubMed", 1.0)))

task(gmean("Cora", "CiteSeer", "PubMed"))
