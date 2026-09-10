"""Score spec for ai4sci-vs-contrastive-scoring (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("auc_mean_dude",
    col("auc_mean_dude").higher().id()
    .bounded_power(bound=1.0))

term("bedroc_mean_dude",
    col("bedroc_mean_dude").higher().id()
    .bounded_power(bound=1.0))

term("ef005_mean_dude",
    col("ef005_mean_dude").higher().id()
    .sigmoid())

term("ef01_mean_dude",
    col("ef01_mean_dude").higher().id()
    .sigmoid())

term("ef05_mean_dude",
    col("ef05_mean_dude").higher().id()
    .sigmoid())

term("ef0005_mean_dude",
    col("ef0005_mean_dude").higher().id()
    .sigmoid())

term("ef001_mean_dude",
    col("ef001_mean_dude").higher().id()
    .sigmoid())

term("ef002_mean_dude",
    col("ef002_mean_dude").higher().id()
    .sigmoid())

term("auc_mean_dekois",
    col("auc_mean_dekois").higher().id()
    .bounded_power(bound=1.0))

term("bedroc_mean_dekois",
    col("bedroc_mean_dekois").higher().id()
    .bounded_power(bound=1.0))

term("ef005_mean_dekois",
    col("ef005_mean_dekois").higher().id()
    .sigmoid())

term("ef01_mean_dekois",
    col("ef01_mean_dekois").higher().id()
    .sigmoid())

term("ef05_mean_dekois",
    col("ef05_mean_dekois").higher().id()
    .sigmoid())

term("ef0005_mean_dekois",
    col("ef0005_mean_dekois").higher().id()
    .sigmoid())

term("ef001_mean_dekois",
    col("ef001_mean_dekois").higher().id()
    .sigmoid())

term("ef002_mean_dekois",
    col("ef002_mean_dekois").higher().id()
    .sigmoid())

term("auc_mean_lit_pcba",
    col("auc_mean_lit-pcba").higher().id()
    .bounded_power(bound=1.0))

term("bedroc_mean_lit_pcba",
    col("bedroc_mean_lit-pcba").higher().id()
    .bounded_power(bound=1.0))

term("ef005_mean_lit_pcba",
    col("ef005_mean_lit-pcba").higher().id()
    .sigmoid())

term("ef01_mean_lit_pcba",
    col("ef01_mean_lit-pcba").higher().id()
    .sigmoid())

term("ef05_mean_lit_pcba",
    col("ef05_mean_lit-pcba").higher().id()
    .sigmoid())

term("ef0005_mean_lit_pcba",
    col("ef0005_mean_lit-pcba").higher().id()
    .sigmoid())

term("ef001_mean_lit_pcba",
    col("ef001_mean_lit-pcba").higher().id()
    .sigmoid())

term("ef002_mean_lit_pcba",
    col("ef002_mean_lit-pcba").higher().id()
    .sigmoid())

setting("dude", weighted_mean(("auc_mean_dude", 1.0), ("bedroc_mean_dude", 1.0), ("ef005_mean_dude", 1.0), ("ef01_mean_dude", 1.0), ("ef05_mean_dude", 1.0), ("ef0005_mean_dude", 1.0), ("ef001_mean_dude", 1.0), ("ef002_mean_dude", 1.0)))
setting("dekois", weighted_mean(("auc_mean_dekois", 1.0), ("bedroc_mean_dekois", 1.0), ("ef005_mean_dekois", 1.0), ("ef01_mean_dekois", 1.0), ("ef05_mean_dekois", 1.0), ("ef0005_mean_dekois", 1.0), ("ef001_mean_dekois", 1.0), ("ef002_mean_dekois", 1.0)))
setting("lit-pcba", weighted_mean(("auc_mean_lit_pcba", 1.0), ("bedroc_mean_lit_pcba", 1.0), ("ef005_mean_lit_pcba", 1.0), ("ef01_mean_lit_pcba", 1.0), ("ef05_mean_lit_pcba", 1.0), ("ef0005_mean_lit_pcba", 1.0), ("ef001_mean_lit_pcba", 1.0), ("ef002_mean_lit_pcba", 1.0)))

task(gmean("dude", "dekois", "lit-pcba"))
