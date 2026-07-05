"""Score spec for simp-length-control.

Corpus SARI (0-100, higher is better) of a FROZEN t5-base simplifier on THREE FIXED
simplification test settings (asset / turk / wiki); beam is fixed and only the
LENGTH / COMPRESSION window varies. Task score = geometric mean of the per-setting
SARI.

## SOTA=0.5 anchor convention (2026-07-05 fix)

Each per-setting logistic is anchored so the STRONGEST baseline (`tuned`) scores
EXACTLY 0.5: `ref=const(<tuned SARI>)`, `scale=(tuned - long) / ln(9)` (`long` =
the weakest baseline). This makes `long` score ~0.1 and preserves the measured
ordering long < tuned, with headroom above 0.5 for any agent config that beats
tuned.

Measured anchors (mrm8488/t5-base-finetuned-turk-text-simplification, seed 42,
300-sentence slice per setting; VALIDATED on k1 H20, 2026-07-03):
    setting   long(length_penalty=2.5, min=40, max=160)   tuned(lp=1.0, max=96)=SOTA
    asset     41.64  (lenratio 2.83 -> over-long)          45.14
    turk      41.24  (lenratio 3.20)                        43.68
    wiki      38.26  (lenratio 1.90)                        43.32
The over-long window pads outputs to ~3x the input length (few DELETE credits) and
trails the tuned compressive window (the SOTA=0.5 anchor) on every setting.

Verified via score_record_details: tuned (strongest) -> ~0.5000 on all 3 settings;
long (weakest) -> ~0.10 on all 3 settings.
"""
from mlsbench.scoring.dsl import *

# SOTA=0.5: ref = strongest baseline (tuned) SARI, scale = (tuned - long) / ln(9).
term("sari_asset", col("sari_asset").higher().id().sigmoid(ref=const(45.136026), scale=1.5907))
term("sari_turk",  col("sari_turk").higher().id().sigmoid(ref=const(43.676860), scale=1.1101))
term("sari_wiki",  col("sari_wiki").higher().id().sigmoid(ref=const(43.321471), scale=2.3039))

setting("asset", weighted_mean(("sari_asset", 1.0)))
setting("turk",  weighted_mean(("sari_turk", 1.0)))
setting("wiki",  weighted_mean(("sari_wiki", 1.0)))

task(gmean("asset", "turk", "wiki"))
