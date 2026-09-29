"""Score spec for ai4sci-vs-contrastive-scoring.

Scored metrics are the five that the fixed evaluation code prints on its
``TEST_METRICS`` line and that the task description names: AUROC, BEDROC and
EF at 0.5 % / 1 % / 5 % (columns ``ef005`` / ``ef01`` / ``ef05``). The parser
also records ``ef0005`` / ``ef001`` / ``ef002`` from the per-cutoff summary
lines; ``ef0005`` and ``ef001`` are the same EF@0.5 % and EF@1 % values under
another name, and EF@2 % is not a documented metric, so none of the three is
scored (scoring the duplicates gave EF 6 of 8 weights per setting).

DEKOIS: the three baselines are within seed noise of each other on BEDROC
(0.711-0.715) and EF (0.3-7 % relative), so calibrating those terms on the
baseline spread made the setting a coin flip. They use a fixed scale in log
space instead: the weakest baseline is the floor and a 10 % relative gain over
it scores 0.5 (2 * sigmoid(ln 1.1 / scale) - 1 = 0.5), so noise-level
differences move a term by a few hundredths. DEKOIS AUROC (0.892-0.922) and
all DUD-E / LIT-PCBA terms keep baseline-calibrated anchors.
"""
import math

from mlsbench.scoring.dsl import *

# 10 % relative gain over the weakest baseline -> 0.5.
DEKOIS_LOG_SCALE = math.log(1.1) / math.log(3.0)

for _name, _setting in (("dude", "dude"), ("dekois", "dekois"), ("lit_pcba", "lit-pcba")):
    term(f"auc_mean_{_name}",
        col(f"auc_mean_{_setting}").higher().id()
        .bounded_power(bound=1.0))

    for _metric in ("bedroc", "ef005", "ef01", "ef05"):
        if _setting == "dekois":
            _spec = col(f"{_metric}_mean_{_setting}").higher().log().sigmoid(
                scale=DEKOIS_LOG_SCALE)
        elif _metric == "bedroc":
            _spec = col(f"{_metric}_mean_{_setting}").higher().id().bounded_power(bound=1.0)
        else:
            _spec = col(f"{_metric}_mean_{_setting}").higher().id().sigmoid()
        term(f"{_metric}_mean_{_name}", _spec)

    setting(_setting, weighted_mean(
        (f"auc_mean_{_name}", 1.0),
        (f"bedroc_mean_{_name}", 1.0),
        (f"ef005_mean_{_name}", 1.0),
        (f"ef01_mean_{_name}", 1.0),
        (f"ef05_mean_{_name}", 1.0),
    ))

task(gmean("dude", "dekois", "lit-pcba"))
