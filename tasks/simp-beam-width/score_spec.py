"""Score spec for simp-beam-width.

Corpus SARI (0-100, higher is better) of a FROZEN t5-base simplifier on THREE FIXED
simplification test settings (asset / turk / wiki), decoded under a FIXED
repetition/length config (no_repeat_ngram_size=3, length_penalty=1.0, max_length=128,
early_stopping=True all FIXED); only the beam WIDTH (num_beams) varies. This isolates
the beam-width lever from no_repeat_ngram_size (which simp-decoding-beam varies
jointly with num_beams). Task score = geometric mean of the per-setting SARI.

## SOTA=0.5 anchor convention (2026-07-05 fix)

Each per-setting logistic is anchored so the STRONGEST baseline (`wide`) scores
EXACTLY 0.5: `ref=const(<wide SARI>)`, `scale=(wide - narrow) / ln(9)` (`narrow` =
the weakest baseline). This makes `narrow` score ~0.1 and preserves the measured
ordering narrow < mid < wide, with headroom above 0.5 for any agent config that
beats wide.

Measured anchors (mrm8488/t5-base-finetuned-turk-text-simplification, seeds 42 and
123 -- IDENTICAL, beam search is deterministic given a fixed decode config and does
not depend on the RNG seed; 300-sentence slice per setting; VALIDATED on k1 H20,
2026-07-05, image pytorch:2.5.1-cuda12.4-cudnn9-runtime pinned):
    setting   narrow(beams=2)   mid(beams=4)   wide(beams=8)=SOTA
    asset         40.81            44.26           46.31
    turk          40.42            43.26           44.08
    wiki          39.61            43.86           44.73
Monotone narrow < mid < wide on all three settings, both seeds.

Verified via score_record_details: wide (strongest) -> ~0.5000 on all 3 settings;
narrow (weakest) -> ~0.10 on all 3 settings.
"""
from mlsbench.scoring.dsl import *

# SOTA=0.5: ref = strongest baseline (wide) SARI, scale = (wide - narrow) / ln(9).
term("sari_asset", col("sari_asset").higher().id().sigmoid(ref=const(46.307912), scale=2.5011))
term("sari_turk",  col("sari_turk").higher().id().sigmoid(ref=const(44.081533), scale=1.6654))
term("sari_wiki",  col("sari_wiki").higher().id().sigmoid(ref=const(44.729053), scale=2.3306))

setting("asset", weighted_mean(("sari_asset", 1.0)))
setting("turk",  weighted_mean(("sari_turk", 1.0)))
setting("wiki",  weighted_mean(("sari_wiki", 1.0)))

task(gmean("asset", "turk", "wiki"))
