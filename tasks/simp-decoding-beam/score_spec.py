"""Score spec for simp-decoding-beam.

Corpus SARI (0-100, higher is better) of a FROZEN t5-base simplifier on THREE FIXED
simplification test settings (asset / turk / wiki); only the BEAM / REPETITION
decode config varies (length window fixed). Task score = geometric mean of the
per-setting SARI.

## SOTA=0.5 anchor convention (2026-07-05 fix)

Each per-setting logistic is anchored so the STRONGEST baseline (`beam_norep`)
scores EXACTLY 0.5: `ref=const(<beam_norep SARI>)`, `scale=(beam_norep - greedy) /
ln(9)` (`greedy` = the weakest baseline). This makes `greedy` score ~0.1 and
preserves the measured monotone ordering greedy < beam4 < beam_norep, with
headroom above 0.5 for any agent config that beats beam_norep.

Measured anchors (mrm8488/t5-base-finetuned-turk-text-simplification, seed 42,
300-sentence slice per setting; VALIDATED on k1 H20, 2026-07-03; corpus SARI matches
the official HuggingFace `evaluate` SARI within ~1-4 pts):
    setting   greedy   beam4   beam_norep(tuned=SOTA)   [SOTA ref: MUSS-S 44.5 / Control-T5 44.9 ASSET]
    asset     35.36    44.26      45.14
    turk      33.38    43.26      43.68
    wiki      34.19    43.86      43.32
The tuned beam_norep matches the literature SOTA-scale SARI and is the SOTA=0.5
anchor; greedy over-generates (lenratio ~1.5) and is the weakest-baseline floor.

Verified via score_record_details: beam_norep (strongest) -> ~0.5000 on all 3
settings; greedy (weakest) -> ~0.10 on all 3 settings.
"""
from mlsbench.scoring.dsl import *

# SOTA=0.5: ref = strongest baseline (beam_norep) SARI, scale = (beam_norep - greedy) / ln(9).
term("sari_asset", col("sari_asset").higher().id().sigmoid(ref=const(45.136026), scale=4.4475))
term("sari_turk",  col("sari_turk").higher().id().sigmoid(ref=const(43.676860), scale=4.6884))
term("sari_wiki",  col("sari_wiki").higher().id().sigmoid(ref=const(43.321471), scale=4.1556))

setting("asset", weighted_mean(("sari_asset", 1.0)))
setting("turk",  weighted_mean(("sari_turk", 1.0)))
setting("wiki",  weighted_mean(("sari_wiki", 1.0)))

task(gmean("asset", "turk", "wiki"))
