"""Score spec for simp-minlen-floor.

Corpus SARI (0-100, higher is better) of a FROZEN t5-base simplifier on THREE FIXED
simplification test settings (asset / turk / wiki), decoded under a FIXED beam
width, length-penalty and max_length (num_beams=5, no_repeat_ngram_size=3,
length_penalty=1.0, max_length=96 all FIXED); only the decoder-side min_length FLOOR
varies. This isolates the min-length floor from the length WINDOW's upper bound
(max_length) and from length_penalty. Task score = geometric mean of the
per-setting SARI.

## SOTA=0.5 anchor convention (2026-07-05 fix)

Each per-setting logistic is anchored so the STRONGEST baseline (`floor0`) scores
EXACTLY 0.5: `ref=const(<floor0 SARI>)`, `scale=(floor0 - floor60) / ln(9)`
(`floor60` = the weakest baseline). This makes `floor60` score ~0.1 and preserves
the measured ordering floor60 < floor20 < floor0, with headroom above 0.5 for any
agent config that beats floor0.

Measured anchors (mrm8488/t5-base-finetuned-turk-text-simplification, seeds 42 and
123 -- IDENTICAL, beam search is deterministic given a fixed decode config and does
not depend on the RNG seed; 300-sentence slice per setting; VALIDATED on k1 H20,
2026-07-05, image pytorch:2.5.1-cuda12.4-cudnn9-runtime pinned):
    setting   floor60(weak)   floor20(mid)   floor0(strong)=SOTA
    asset         40.92          44.54          45.14
    turk          41.05          43.31          43.68
    wiki          35.16          43.09          43.32
Monotone floor60 < floor20 < floor0 on all three settings, both seeds. A large
min_length floor forces the decoder to keep generating past the point where it
would naturally stop (natural EOS), padding the output with low-value continuation
tokens that rarely correspond to a correct ADD/KEEP edit; a zero floor lets the
model stop wherever beam search naturally finds the best EOS.

Verified via score_record_details: floor0 (strongest) -> ~0.5000 on all 3
settings; floor60 (weakest) -> ~0.10 on all 3 settings.
"""
from mlsbench.scoring.dsl import *

# SOTA=0.5: ref = strongest baseline (floor0) SARI, scale = (floor0 - floor60) / ln(9).
term("sari_asset", col("sari_asset").higher().id().sigmoid(ref=const(45.136026), scale=1.9200))
term("sari_turk",  col("sari_turk").higher().id().sigmoid(ref=const(43.676860), scale=1.1955))
term("sari_wiki",  col("sari_wiki").higher().id().sigmoid(ref=const(43.321471), scale=3.7126))

setting("asset", weighted_mean(("sari_asset", 1.0)))
setting("turk",  weighted_mean(("sari_turk", 1.0)))
setting("wiki",  weighted_mean(("sari_wiki", 1.0)))

task(gmean("asset", "turk", "wiki"))
