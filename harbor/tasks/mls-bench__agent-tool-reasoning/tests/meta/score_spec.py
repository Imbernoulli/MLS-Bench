"""Score spec for agent-tool-reasoning.

Three evaluation settings, each a different agent LLM backbone on the
StableToolBench I1-instruction subset (labels match config.json test_cmds):

  I1-instruction-deepseek : DeepSeek deepseek-chat    (DeepSeek official API)
  I1-instruction-qwen72b  : qwen2.5-72b-instruct      (Dashscope)
  I1-instruction-qwen7b   : qwen2.5-7b-instruct       (Dashscope)

Scoring uses only the two quality metrics that matter in the literature:

  - pass_rate: fraction of queries where the agent self-reports a valid
               final answer (reported by ToolLLM paper).
  - sopr:      Stable Pass Rate — judged by an independent LLM
               (meta-llama/llama-3.3-70b-instruct via OpenRouter),
               reported by StableToolBench paper as the primary metric.

avg_queries (efficiency) and give_up_rate (largely redundant with
1 - pass_rate) remain in the leaderboard as informational columns but
do NOT enter the task score. Count columns (sopr_n_scored_*) and
elapsed_* are also informational.

Normalization uses dynamic leaderboard anchors, so the strongest current
paper-standard baseline becomes the 50-point anchor for each scored metric.
"""
from mlsbench.scoring.dsl import *

# ── I1-instruction-deepseek ─
term("pass_rate_deepseek",
    col("pass_rate_deepseek").higher().id()
    .bounded_power(bound=1.0))

term("sopr_deepseek",
    col("sopr_deepseek").higher().id()
    .bounded_power(bound=1.0))

# ── I1-instruction-qwen72b ──
term("pass_rate_qwen72b",
    col("pass_rate_qwen72b").higher().id()
    .bounded_power(bound=1.0))

term("sopr_qwen72b",
    col("sopr_qwen72b").higher().id()
    .bounded_power(bound=1.0))

# ── I1-instruction-qwen7b ───
term("pass_rate_qwen7b",
    col("pass_rate_qwen7b").higher().id()
    .bounded_power(bound=1.0))

term("sopr_qwen7b",
    col("sopr_qwen7b").higher().id()
    .bounded_power(bound=1.0))

setting("I1-instruction-deepseek", weighted_mean(
    ("pass_rate_deepseek", 1.0),
    ("sopr_deepseek", 1.0),
))
setting("I1-instruction-qwen72b", weighted_mean(
    ("pass_rate_qwen72b", 1.0),
    ("sopr_qwen72b", 1.0),
))
setting("I1-instruction-qwen7b", weighted_mean(
    ("pass_rate_qwen7b", 1.0),
    ("sopr_qwen7b", 1.0),
))

task(gmean("I1-instruction-deepseek", "I1-instruction-qwen72b", "I1-instruction-qwen7b"))
