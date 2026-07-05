"""Score spec for simp-decoding-strategy.

Corpus SARI (0-100, higher is better) of a FROZEN t5-base simplifier on THREE FIXED
simplification test settings (asset / turk / wiki), under the agent's top-level
DECODING STRATEGY choice ("sample" / "topp" / "beam"). Task score = geometric mean
of the per-setting SARI.

## SOTA=0.5 anchor convention (2026-07-05 fix)

Each per-setting logistic is anchored so the STRONGEST baseline (`beam`) scores
EXACTLY 0.5: `ref=const(<beam SARI>)`, `scale=(beam - sample) / ln(9)` (`sample` =
the weakest baseline). This makes `sample` score ~0.1 and preserves the measured
ordering sample < topp < beam, with headroom above 0.5 for any agent config that
beats beam.

Measured anchors (mrm8488/t5-base-finetuned-turk-text-simplification, seed 42,
300-sentence slice per setting; VALIDATED on k1 H20, 2026-07-05, image
pytorch:2.5.1-cuda12.4-cudnn9-runtime pinned):
    setting   sample   topp    beam(tuned)=SOTA
    asset     33.23    35.29     45.63
    turk      31.84    34.39     44.15
    wiki      27.03    30.94     44.60
Monotone sample < topp < beam on all three settings.

Verified via score_record_details: beam (strongest) -> ~0.5000 on all 3 settings;
sample (weakest) -> ~0.10 on all 3 settings.
"""
from mlsbench.scoring.dsl import *

# SOTA=0.5: ref = strongest baseline (beam) SARI, scale = (beam - sample) / ln(9).
term("sari_asset", col("sari_asset").higher().id().sigmoid(ref=const(45.629270), scale=5.6437))
term("sari_turk",  col("sari_turk").higher().id().sigmoid(ref=const(44.153228), scale=5.6030))
term("sari_wiki",  col("sari_wiki").higher().id().sigmoid(ref=const(44.597375), scale=7.9935))

setting("asset", weighted_mean(("sari_asset", 1.0)))
setting("turk",  weighted_mean(("sari_turk", 1.0)))
setting("wiki",  weighted_mean(("sari_wiki", 1.0)))

task(gmean("asset", "turk", "wiki"))
