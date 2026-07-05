"""Score spec for simp-decoding-temperature.

Corpus SARI (0-100, higher is better) of a FROZEN t5-base simplifier on THREE FIXED
simplification test settings (asset / turk / wiki), decoded via SAMPLING
(do_sample=True, num_beams=1, no_repeat_ngram_size=3 all FIXED); only the softmax
TEMPERATURE varies. Task score = geometric mean of the per-setting SARI.

## SOTA=0.5 anchor convention (2026-07-05 fix)

Each per-setting logistic is anchored so the STRONGEST baseline (`cold`) scores
EXACTLY 0.5: `ref=const(<cold SARI>)`, `scale=(cold - hot) / ln(9)` (`hot` = the
weakest baseline). This makes `hot` score ~0.1 and preserves the measured ordering
hot < mid < cold, with headroom above 0.5 for any agent config that beats cold.

Measured anchors (mrm8488/t5-base-finetuned-turk-text-simplification, seed 42,
300-sentence slice per setting; VALIDATED on k1 H20, 2026-07-05, image
pytorch:2.5.1-cuda12.4-cudnn9-runtime pinned):
    setting   hot(T=2.0)   mid(T=1.0)   cold(T=0.3)=SOTA
    asset      24.09        32.10        37.98
    turk       23.09        32.08        36.67
    wiki       15.43        27.01        35.79
Monotone hot < mid < cold on all three settings.

Verified via score_record_details: cold (strongest) -> ~0.5000 on all 3 settings;
hot (weakest) -> ~0.10 on all 3 settings.
"""
from mlsbench.scoring.dsl import *

# SOTA=0.5: ref = strongest baseline (cold) SARI, scale = (cold - hot) / ln(9).
term("sari_asset", col("sari_asset").higher().id().sigmoid(ref=const(37.976371), scale=6.3178))
term("sari_turk",  col("sari_turk").higher().id().sigmoid(ref=const(36.670040), scale=6.1822))
term("sari_wiki",  col("sari_wiki").higher().id().sigmoid(ref=const(35.790406), scale=9.2654))

setting("asset", weighted_mean(("sari_asset", 1.0)))
setting("turk",  weighted_mean(("sari_turk", 1.0)))
setting("wiki",  weighted_mean(("sari_wiki", 1.0)))

task(gmean("asset", "turk", "wiki"))
