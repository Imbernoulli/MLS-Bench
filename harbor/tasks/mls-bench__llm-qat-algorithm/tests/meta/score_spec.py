"""Score spec for llm-qat-algorithm.

Scored on quantized WikiText-2 perplexity at three bit-widths (INT4, INT3,
INT2). Lower PPL is better; FP16 is the dense reference at ~13 PPL on
Pythia-1.4B. Task score is the gmean across the three bit-widths so an
agent has to do well at INT2 (the hardest case), not just at INT4.

Note: ``degradation_*`` columns are NOT scored. They duplicate the signal
in ``wikitext2_ppl_*`` (since fp16_ppl is constant for a given model) and
can go negative when finetune_then_ptq drops PPL below FP16 — that
misbehaves with ``bounded_power(bound=0.0)``.
"""
from mlsbench.scoring.dsl import *

term("wikitext2_ppl_qat_1b_int4",
    col("wikitext2_ppl_qat-1b-int4").lower().id()
    .bounded_power(bound=1.0))

term("wikitext2_ppl_qat_1b_int3",
    col("wikitext2_ppl_qat-1b-int3").lower().id()
    .bounded_power(bound=1.0))

term("wikitext2_ppl_qat_1b_int2",
    col("wikitext2_ppl_qat-1b-int2").lower().id()
    .bounded_power(bound=1.0))

setting("qat-1b-int4", weighted_mean(("wikitext2_ppl_qat_1b_int4", 1.0)))
setting("qat-1b-int3", weighted_mean(("wikitext2_ppl_qat_1b_int3", 1.0)))
setting("qat-1b-int2", weighted_mean(("wikitext2_ppl_qat_1b_int2", 1.0)))

task(gmean("qat-1b-int4", "qat-1b-int3", "qat-1b-int2"))
