"""Score spec for llm-kv-structural-reduction.

Primary evaluation is 345M pretraining (aligned with llm-pretrain-attention),
augmented with the KV-footprint metric specific to this structural
compression task.

Refs calibrated from the four D=21N baseline runs (seed=42, ~7.1B tokens):

  baseline   kv_B/tok  val_loss  heldout  arc_e  hella
  ----------------------------------------------------
  mha          4096    2.275     3.967    54.9   33.4
  gqa(4×)      1024    2.313     3.969    55.0   33.1
  mqa(16×)      256    2.338     3.999    53.5   32.5
  mla(r=0.25)   192    2.307     3.988    54.8   33.2

ref values are set near the baseline mean so the four anchors spread
roughly around 0.5; bound is the theoretical or practically attainable
limit of each metric.

Generation throughput is intentionally NOT scored — `kv_bytes_per_token`
already captures MLA's structural advantage, and a wall-clock t/s number
in pure-PyTorch eager mode reflects per-layer op count more than model
design (real MLA serving uses fused CUDA kernels we can't require here).
"""
from mlsbench.scoring.dsl import *

# --- 345M pretraining quality ---
term("val_loss_345m",
    col("val_loss_gpt-345m").lower().id()
    .bounded_power(bound=0.0))

# kv_bytes_per_token at 345M: LOWER is better. baseline spread 192-4096.
term("kv_bytes_per_token_345m",
    col("kv_bytes_per_token_gpt-345m").lower().id()
    .sigmoid())

# heldout_loss (avg over wikitext2/103/lambada): lower is better.
term("heldout_loss_345m",
    col("heldout_loss_gpt-345m").lower().id()
    .bounded_power(bound=0.0))

# --- lm-eval downstream tasks (0-shot) ---
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
    ("val_loss_345m", 2.0),
    ("heldout_loss_345m", 0.5),
    ("kv_bytes_per_token_345m", 1.5),
))

setting("lm-eval-345m", weighted_mean(
    ("arc_easy", 1.0),
    ("hellaswag", 1.0),
    ("piqa", 1.0),
    ("winogrande", 1.0),
))

task(gmean("gpt-345m", "lm-eval-345m"))
