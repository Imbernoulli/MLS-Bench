"""Score spec for llm-qat-algorithm.

Scored on quantized WikiText-2 perplexity at three bit-widths (INT4, INT3,
INT2). Lower PPL is better; FP16 is the dense reference at ~13 PPL on
Pythia-1.4B. Task score is the gmean across the three bit-widths so an
agent has to do well at INT2 (the hardest case), not just at INT4.

Each perplexity term is scored on a log scale, i.e. on the evaluation
cross-entropy log(PPL) in nats, with the theoretical bound log(1) = 0.
Perplexity is exp(loss), so the raw scale is badly skewed: at INT2 the
worst baseline (no_qat, plain round-to-nearest) diverges to ~1e5 PPL, and on
the raw scale that floor puts the best baseline (lsq, ~19.5) at r_ref ~0.9998,
which drops the term into the sigmoid fallback where every PPL from ~1.5 to
~500 scores 0.498-0.500 (ste at 72.6 ties lsq at 19.5). On the loss scale the
same anchors give a smooth curve (r_ref ~0.74) and the INT4/INT3 terms keep
their discriminative, non-pathological calibration.

Note: ``degradation_*`` columns are NOT scored. They duplicate the signal
in ``wikitext2_ppl_*`` (since fp16_ppl is constant for a given model) and
can go negative when finetune_then_ptq drops PPL below FP16 — that
misbehaves with ``bounded_power(bound=0.0)``.
"""
from mlsbench.scoring.dsl import *

term("wikitext2_ppl_qat_1b_int4",
    col("wikitext2_ppl_qat-1b-int4").lower().log()
    .bounded_power(bound=1.0))

term("wikitext2_ppl_qat_1b_int3",
    col("wikitext2_ppl_qat-1b-int3").lower().log()
    .bounded_power(bound=1.0))

term("wikitext2_ppl_qat_1b_int2",
    col("wikitext2_ppl_qat-1b-int2").lower().log()
    .bounded_power(bound=1.0))

setting("qat-1b-int4", weighted_mean(("wikitext2_ppl_qat_1b_int4", 1.0)))
setting("qat-1b-int3", weighted_mean(("wikitext2_ppl_qat_1b_int3", 1.0)))
setting("qat-1b-int2", weighted_mean(("wikitext2_ppl_qat_1b_int2", 1.0)))

task(gmean("qat-1b-int4", "qat-1b-int3", "qat-1b-int2"))
