"""Score spec for simp-nucleus-sampling.

Corpus SARI (0-100, higher is better) of a FROZEN t5-base simplifier on THREE FIXED
simplification test settings (asset / turk / wiki), decoded via SAMPLING
(do_sample=True, num_beams=1, temperature=1.0, no_repeat_ngram_size=3 all FIXED);
only the NUCLEUS (top-p) mass varies. Task score = geometric mean of the per-setting
SARI.

## SOTA=0.5 anchor convention (2026-07-05 fix)

Each per-setting logistic is anchored so the STRONGEST baseline (`tight`) scores
EXACTLY 0.5: `ref=const(<tight SARI>)`, `scale=(tight - wide) / ln(9)` (`wide` = the
weakest baseline). This makes `wide` score ~0.1 and preserves the measured ordering
wide < mid < tight, with headroom above 0.5 for any agent config that beats tight.

Measured anchors (mrm8488/t5-base-finetuned-turk-text-simplification, seed 42,
300-sentence slice per setting; VALIDATED on k1 H20, 2026-07-05, image
pytorch:2.5.1-cuda12.4-cudnn9-runtime pinned):
    setting   wide(p=1.0)   mid(p=0.9)   tight(p=0.6)=SOTA
    asset       32.10         34.60         37.76
    turk        32.08         33.29         36.80
    wiki        27.01         29.89         35.28
Monotone wide < mid < tight on all three settings.

Verified via score_record_details: tight (strongest) -> ~0.5000 on all 3 settings;
wide (weakest) -> ~0.10 on all 3 settings.
"""
from mlsbench.scoring.dsl import *

# SOTA=0.5: ref = strongest baseline (tight) SARI, scale = (tight - wide) / ln(9).
term("sari_asset", col("sari_asset").higher().id().sigmoid(ref=const(37.760657), scale=2.5777))
term("sari_turk",  col("sari_turk").higher().id().sigmoid(ref=const(36.801646), scale=2.1481))
term("sari_wiki",  col("sari_wiki").higher().id().sigmoid(ref=const(35.281394), scale=3.7631))

setting("asset", weighted_mean(("sari_asset", 1.0)))
setting("turk",  weighted_mean(("sari_turk", 1.0)))
setting("wiki",  weighted_mean(("sari_wiki", 1.0)))

task(gmean("asset", "turk", "wiki"))
