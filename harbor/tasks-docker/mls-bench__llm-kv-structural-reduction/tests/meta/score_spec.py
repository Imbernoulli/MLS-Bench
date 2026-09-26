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

KV budget. The template starts as dense MHA, and before this constraint MHA
was the top-scoring baseline (0.361 vs MLA 0.330): its kv term was already 0
(it is the worst anchor), but the floor-anchored quality terms outweighed the
flat linear kv sigmoid, so leaving the attention untouched won a KV-reduction
task. The task now fixes a budget of at least a 4x reduction relative to the
dense MHA control, kv_bytes_per_token <= 1024 (GQA-4x sits exactly on it;
MQA and MLA are well inside). It is enforced multiplicatively in both
settings with exp(-0.003 * excess): 1100 B -> x0.80, 1536 B -> x0.22,
2048 B -> x0.05, 4096 B (MHA) -> x1e-4.

Inside the budget the kv term is scored on log bytes, so each halving of the
cache is worth the same (the linear sigmoid gave 192 -> 96 B only +0.01 on the
term, and 0 B scored 0.52). Anchors are unchanged: 4096 B scores 0, the best
baseline (MLA, 192 B) scores 0.5.

heldout_loss is reported but NOT scored. It is measured on only 64 random
windows (16 iters x batch 4), and a re-run of the same code in the current
image shifts it by ~0.10 for every baseline (mla 3.885 vs 3.988, mha 3.864 vs
3.967) while val_loss reproduces to ~0.001; that shift is ~3x the whole
baseline spread (3.967-3.999), so the term measured the environment rather
than the method. val_loss carries the same quality signal reproducibly.

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
# Log scale: KV memory savings are multiplicative.
term("kv_bytes_per_token_345m",
    col("kv_bytes_per_token_gpt-345m").lower().log()
    .sigmoid())

# KV budget: at least 4x below dense MHA (4096 B/token at 345M).
term("kv_budget_345m",
    penalty_upper(col("kv_bytes_per_token_gpt-345m").lower().id(),
                  target=1024.0, sharpness=0.003))

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
    ("kv_bytes_per_token_345m", 1.5),
), constraints=["kv_budget_345m"])

setting("lm-eval-345m", weighted_mean(
    ("arc_easy", 1.0),
    ("hellaswag", 1.0),
    ("piqa", 1.0),
    ("winogrande", 1.0),
), constraints=["kv_budget_345m"])

task(gmean("gpt-345m", "lm-eval-345m"))
