"""Score spec for llm-pretrain-normalization.

Reference baseline: rmsnorm (seed=42)
  val_loss=2.295, wikitext2_ppl=44.75, lambada_ppl=68.29,
  arc_easy=54.97, hellaswag=33.25
"""
from mlsbench.scoring.dsl import *

term("val_loss",
    col("val_loss_gpt-345m").lower().id()
    .bounded_power(bound=0.0))

term("wikitext2_ppl",
    col("wikitext2_ppl_gpt-345m").lower().id()
    .bounded_power(bound=1.0))

term("lambada_ppl",
    col("lambada_ppl_gpt-345m").lower().id()
    .bounded_power(bound=1.0))

term("arc_easy",
    col("arc_easy_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("hellaswag",
    col("hellaswag_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("piqa",
    col("piqa_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("winogrande",
    col("winogrande_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

setting("gpt-345m", weighted_mean(
    ("val_loss", 2.0),
    ("wikitext2_ppl", 1.0),
    ("lambada_ppl", 1.0),
))

setting("lm-eval-345m", weighted_mean(
    ("arc_easy", 1.0),
    ("hellaswag", 1.0),
    ("piqa", 1.0),
    ("winogrande", 1.0),
))

task(gmean("gpt-345m", "lm-eval-345m"))
