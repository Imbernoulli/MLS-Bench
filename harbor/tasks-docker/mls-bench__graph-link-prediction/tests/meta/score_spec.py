"""Score spec for graph-link-prediction."""
from mlsbench.scoring.dsl import *

# AUC, MRR, Hits@20 are all on [0, 100] percentage scale; all higher is better
term("AUC_Cora",
    col("AUC_Cora").higher().id()
    .bounded_power(bound=100.0))

term("MRR_Cora",
    col("MRR_Cora").higher().id()
    .bounded_power(bound=100.0))

term("Hits_20_Cora",
    col("Hits@20_Cora").higher().id()
    .bounded_power(bound=100.0))

term("AUC_CiteSeer",
    col("AUC_CiteSeer").higher().id()
    .bounded_power(bound=100.0))

term("MRR_CiteSeer",
    col("MRR_CiteSeer").higher().id()
    .bounded_power(bound=100.0))

term("Hits_20_CiteSeer",
    col("Hits@20_CiteSeer").higher().id()
    .bounded_power(bound=100.0))

term("Hits_50_ogbl_collab",
    col("Hits@50_ogbl-collab").higher().id()
    .bounded_power(bound=100.0))

term("MRR_ogbl_collab",
    col("MRR_ogbl-collab").higher().id()
    .bounded_power(bound=100.0))

setting("Cora", weighted_mean(("AUC_Cora", 1.0), ("MRR_Cora", 1.0), ("Hits_20_Cora", 1.0)))
setting("CiteSeer", weighted_mean(("AUC_CiteSeer", 1.0), ("MRR_CiteSeer", 1.0), ("Hits_20_CiteSeer", 1.0)))
setting("ogbl-collab", weighted_mean(("Hits_50_ogbl_collab", 1.0), ("MRR_ogbl_collab", 1.0)))

task(gmean("Cora", "CiteSeer", "ogbl-collab"))
