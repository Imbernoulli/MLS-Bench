"""Score spec for llm-dllm-demask-strategy.

Scored settings mirror the real config.json test_cmd labels:
llada-math, llada-humaneval, and hidden dream-text. Historical columns from
older step-sweep runs remain in the leaderboard but are not scored.
"""
from mlsbench.scoring.dsl import *


term("accuracy_llada_math",
    col("accuracy_llada-math").higher().id()
    .bounded_power(bound=1.0))

term("avg_steps_llada_math",
    col("avg_steps_llada-math").lower().id()
    .bounded_power(bound=0.0))

term("accuracy_llada_humaneval",
    col("accuracy_llada-humaneval").higher().id()
    .bounded_power(bound=1.0))

term("avg_steps_llada_humaneval",
    col("avg_steps_llada-humaneval").lower().id()
    .bounded_power(bound=0.0))

term("gen_ppl_dream_text",
    col("gen_ppl_dream-text").lower().id()
    .bounded_power(bound=1.0))

term("mauve_dream_text",
    col("mauve_dream-text").higher().id()
    .bounded_power(bound=1.0))

term("entropy_dream_text",
    col("entropy_dream-text").higher().id()
    .sigmoid())

term("rep2_dream_text",
    col("rep2_dream-text").lower().id()
    .bounded_power(bound=0.0))

term("avg_steps_dream_text",
    col("avg_steps_dream-text").lower().id()
    .bounded_power(bound=0.0))

setting("llada-math", weighted_mean(
    ("accuracy_llada_math", 1.0),
    ("avg_steps_llada_math", 0.5),
))

setting("llada-humaneval", weighted_mean(
    ("accuracy_llada_humaneval", 1.0),
    ("avg_steps_llada_humaneval", 0.5),
))

setting("dream-text", weighted_mean(
    ("gen_ppl_dream_text", 1.0),
    ("mauve_dream_text", 1.0),
    ("entropy_dream_text", 0.5),
    ("rep2_dream_text", 0.5),
    ("avg_steps_dream_text", 0.5),
))

task(gmean("llada-math", "llada-humaneval", "dream-text"))
