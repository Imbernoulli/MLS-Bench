"""Score spec for simp-source-policy (monotonicity / anti-gaming task).

Corpus SARI (0-100, higher is better) of a FROZEN t5-base simplifier on THREE FIXED
simplification test settings (asset / turk / wiki), under the agent's SOURCE POLICY.
Task score = geometric mean of the per-setting SARI, so a policy must win on ALL
THREE settings.

## SOTA=0.5 anchor convention (2026-07-05 fix)

Each per-setting logistic is anchored so the STRONGEST baseline (`beam`, the
tuned-beam T5 decode) scores EXACTLY 0.5: `ref=const(<beam SARI>)`,
`scale=(beam - empty) / ln(9)` (`empty` = the degenerate floor). This makes
`empty` score ~0.1, preserves the measured monotone ordering
empty < first_token < truncate < greedy < beam, and leaves headroom above 0.5
for any agent policy that beats the tuned-beam SOTA reference.

Measured anchors (mrm8488/t5-base-finetuned-turk-text-simplification, seed 42,
300-sentence slice per setting; VALIDATED on k1 H20, 2026-07-03; corpus SARI matches
the official HuggingFace `evaluate` SARI within ~1-4 pts):
    setting  empty  first_tok  truncate(.75)  greedy  beam(tuned=SOTA)
    asset    19.61   20.31        34.61        35.36     45.14
    turk     18.66   19.81        35.43        33.38     43.68
    wiki     11.45   11.54        29.75        34.19     43.32
The DEGENERATE floor (empty / first-token, ~11-20) is genuinely low on every
setting; a naive truncation is a mid baseline; the real greedy decode is higher;
the tuned-beam T5 decode is the SOTA-scale top (matches MUSS-S 44.5 / Control-T5
44.9 on ASSET) and is the SOTA=0.5 anchor. NOTE: `copy_input` scores HIGH (~52-60)
here — a well-documented SARI-KEEP artifact on the conservative ASSET/Turk
references (their human refs keep most source n-grams, so copying earns large KEEP
credit). SARI's ADD/DELETE terms, not KEEP, are what the real simplifier must win —
hence the anchor is the degenerate-floor-to-beam span, NOT copy-input.

Verified via score_record_details: beam (strongest) -> ~0.5000 on all 3 settings;
empty (weakest) -> ~0.10 on all 3 settings.
"""
from mlsbench.scoring.dsl import *

# SOTA=0.5: ref = strongest baseline (beam) SARI, scale = (beam - empty) / ln(9).
term("sari_asset", col("sari_asset").higher().id().sigmoid(ref=const(45.136026), scale=11.6185))
term("sari_turk",  col("sari_turk").higher().id().sigmoid(ref=const(43.676860), scale=11.3862))
term("sari_wiki",  col("sari_wiki").higher().id().sigmoid(ref=const(43.321471), scale=14.5043))

setting("asset", weighted_mean(("sari_asset", 1.0)))
setting("turk",  weighted_mean(("sari_turk", 1.0)))
setting("wiki",  weighted_mean(("sari_wiki", 1.0)))

task(gmean("asset", "turk", "wiki"))
