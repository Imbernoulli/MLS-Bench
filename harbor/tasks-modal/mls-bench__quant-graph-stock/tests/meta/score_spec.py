"""Score spec for quant-graph-stock."""
from mlsbench.scoring.dsl import *

# ic, icir, rank_ic, rank_icir: higher better, unbounded -> sigmoid
# annualized_return: higher better, unbounded -> sigmoid
# max_drawdown: higher better when reported as negative drawdown, because closer to 0 is better -> sigmoid
# information_ratio: higher better (like Sharpe ratio), unbounded -> sigmoid

# csi300 setting
term("ic_csi300",
    col("ic_csi300").higher().id()
    .sigmoid())

term("icir_csi300",
    col("icir_csi300").higher().id()
    .sigmoid())

term("rank_ic_csi300",
    col("rank_ic_csi300").higher().id()
    .sigmoid())

term("rank_icir_csi300",
    col("rank_icir_csi300").higher().id()
    .sigmoid())

term("annualized_return_csi300",
    col("annualized_return_csi300").higher().id()
    .sigmoid())

term("max_drawdown_csi300",
    col("max_drawdown_csi300").higher().id()
    .sigmoid())

term("information_ratio_csi300",
    col("information_ratio_csi300").higher().id()
    .sigmoid())

# csi100 setting
term("ic_csi100",
    col("ic_csi100").higher().id()
    .sigmoid())

term("icir_csi100",
    col("icir_csi100").higher().id()
    .sigmoid())

term("rank_ic_csi100",
    col("rank_ic_csi100").higher().id()
    .sigmoid())

term("rank_icir_csi100",
    col("rank_icir_csi100").higher().id()
    .sigmoid())

term("annualized_return_csi100",
    col("annualized_return_csi100").higher().id()
    .sigmoid())

term("max_drawdown_csi100",
    col("max_drawdown_csi100").higher().id()
    .sigmoid())

term("information_ratio_csi100",
    col("information_ratio_csi100").higher().id()
    .sigmoid())

# csi300_recent setting
term("ic_csi300_recent",
    col("ic_csi300_recent").higher().id()
    .sigmoid())

term("icir_csi300_recent",
    col("icir_csi300_recent").higher().id()
    .sigmoid())

term("rank_ic_csi300_recent",
    col("rank_ic_csi300_recent").higher().id()
    .sigmoid())

term("rank_icir_csi300_recent",
    col("rank_icir_csi300_recent").higher().id()
    .sigmoid())

term("annualized_return_csi300_recent",
    col("annualized_return_csi300_recent").higher().id()
    .sigmoid())

term("max_drawdown_csi300_recent",
    col("max_drawdown_csi300_recent").higher().id()
    .sigmoid())

term("information_ratio_csi300_recent",
    col("information_ratio_csi300_recent").higher().id()
    .sigmoid())

setting("csi300", weighted_mean(
    ("ic_csi300", 1.0),
    ("icir_csi300", 1.0),
    ("rank_ic_csi300", 1.0),
    ("rank_icir_csi300", 1.0),
    ("annualized_return_csi300", 1.0),
    ("max_drawdown_csi300", 1.0),
    ("information_ratio_csi300", 1.0),
))
setting("csi100", weighted_mean(
    ("ic_csi100", 1.0),
    ("icir_csi100", 1.0),
    ("rank_ic_csi100", 1.0),
    ("rank_icir_csi100", 1.0),
    ("annualized_return_csi100", 1.0),
    ("max_drawdown_csi100", 1.0),
    ("information_ratio_csi100", 1.0),
))
setting("csi300_recent", weighted_mean(
    ("ic_csi300_recent", 1.0),
    ("icir_csi300_recent", 1.0),
    ("rank_ic_csi300_recent", 1.0),
    ("rank_icir_csi300_recent", 1.0),
    ("annualized_return_csi300_recent", 1.0),
    ("max_drawdown_csi300_recent", 1.0),
    ("information_ratio_csi300_recent", 1.0),
))

task(gmean("csi300", "csi100", "csi300_recent"))
