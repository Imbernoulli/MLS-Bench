"""Score spec for simp-model-capacity.

Corpus SARI (0-100, higher is better) of the agent's CHOSEN FROZEN pretrained
simplifier on THREE FIXED simplification test settings (asset / turk / wiki), under
a FIXED strong beam decode (num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0,
max_length=128); only WHICH CHECKPOINT is loaded varies. Task score = geometric mean
of the per-setting SARI.

## SOTA=0.5 anchor convention (2026-07-05 fix)

Each per-setting logistic is anchored so the STRONGEST checkpoint (`base_turk`)
scores EXACTLY 0.5: `ref=const(<base_turk SARI>)`, `scale=(base_turk -
small_wikiauto) / ln(9)` (`small_wikiauto` = the weakest checkpoint). This makes
`small_wikiauto` score ~0.1 and preserves the measured ordering small_wikiauto <
small_turk < base_turk, with headroom above 0.5 for any checkpoint that beats
base_turk.

Measured anchors (seed 42, 300-sentence slice per setting; VALIDATED on k1 H20,
2026-07-05, image pytorch:2.5.1-cuda12.4-cudnn9-runtime pinned):
    setting   small_wikiauto(weak)   small_turk(mid)   base_turk(strong)=SOTA
    asset          39.88                 41.92              45.14
    turk           39.09                 41.99              43.68
    wiki           38.16                 42.23              43.32
Monotone small_wikiauto < small_turk < base_turk on all three settings: model
CAPACITY (t5-base, ~220M, 3.7x params vs t5-small's 60M) AND the fine-tuning-data
family (turk-focused vs the broader wiki_auto_asset_turk mix) both matter here;
base_turk (the checkpoint every other simp-* task uses) is the clean strongest
choice on all three settings and the SOTA=0.5 anchor.

Verified via score_record_details: base_turk (strongest) -> ~0.5000 on all 3
settings; small_wikiauto (weakest) -> ~0.10 on all 3 settings.
"""
from mlsbench.scoring.dsl import *

# SOTA=0.5: ref = strongest checkpoint (base_turk) SARI, scale = (base_turk - small_wikiauto) / ln(9).
term("sari_asset", col("sari_asset").higher().id().sigmoid(ref=const(45.136026), scale=2.3927))
term("sari_turk",  col("sari_turk").higher().id().sigmoid(ref=const(43.676860), scale=2.0880))
term("sari_wiki",  col("sari_wiki").higher().id().sigmoid(ref=const(43.321471), scale=2.3473))

setting("asset", weighted_mean(("sari_asset", 1.0)))
setting("turk",  weighted_mean(("sari_turk", 1.0)))
setting("wiki",  weighted_mean(("sari_wiki", 1.0)))

task(gmean("asset", "turk", "wiki"))
