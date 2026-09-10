# Benchmark Data

This task does not ship checked-in benchmark JSON manifests. Package data
preparation materializes the public benchmark assets into the offline Hugging
Face cache before final runs:

- `THUDM/LongBench`: full LongBench-E `hotpotqa_e`,
  `passage_retrieval_en_e`, and `repobench-p_e`.
- `opencompass/NeedleBench`: English haystacks used for the shared
  RULER/NeedleBench-style NIAH probe at three document depths.
- `openai/gsm8k`: full `main` test split.

These sources, split names, prompt templates, generation limits, and scoring
semantics are intentionally shared with `llm-kv-selection-budgeting` wherever
the workload labels match. Final evaluation runs with `HF_HUB_OFFLINE=1` and
`HF_DATASETS_OFFLINE=1`; if the cache is missing, prepare it through the
`transformers-kv-lab` package data dependency before submitting baselines.

`ADAPTIVE_KV_MAX_EXAMPLES=0` is the final-run setting and evaluates the full
configured workload. Positive values are only for local smoke validation.
