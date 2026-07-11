"""Pending calibration score spec for a text-simplification sibling.

Fresh gem-full-test-v2 anchors have not yet produced task-bound, terminal,
cryptographically bound worker evidence. The leaderboard intentionally contains
only its schema header. With no baseline floor, every finite metric maps to
exactly zero; missing or rejected verification output also maps to zero.
"""
from mlsbench.scoring.dsl import *

term("sari_asset", col("sari_asset").higher().id().sigmoid(ref=const(1000.0), scale=1.0))
term("sari_turk", col("sari_turk").higher().id().sigmoid(ref=const(1000.0), scale=1.0))
term("sari_wiki", col("sari_wiki").higher().id().sigmoid(ref=const(1000.0), scale=1.0))

setting("asset", weighted_mean(("sari_asset", 1.0)))
setting("turk", weighted_mean(("sari_turk", 1.0)))
setting("wiki", weighted_mean(("sari_wiki", 1.0)))

task(gmean("asset", "turk", "wiki"))
