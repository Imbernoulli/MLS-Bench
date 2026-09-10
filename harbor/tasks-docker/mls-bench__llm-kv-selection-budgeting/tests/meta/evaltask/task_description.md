# LLM KV Cache: Selection Budgeting

## Research Question

Design a KV token-retention controller inside a shared full-attention
Hugging Face decoding harness. The task asks how a token-scoring rule
should rank prefill KV entries so that the model can decode from a small
fixed-budget subset of those entries without losing long-context quality.

## Background

Long-context LLM inference stores key-value (KV) tensors for every
attention layer. KV selection methods reduce this cache by retaining a
subset of historical tokens while preserving enough context for generation
quality. The shared hook in this task exposes the union of behaviors
needed by three reference families:

- StreamingLLM (Xiao et al., ICLR 2024; arXiv:2309.17453) preserves a
  small set of attention sinks plus a recent window, with optional
  RoPE-aware key rerotation so the rolling window remains positionally
  consistent.
- Expected Attention (Devoto, Jeblick, Jegou, 2025; arXiv:2510.00636)
  estimates how future queries will attend to each prefill KV pair, in
  closed form from observed distributional statistics, and prunes
  low-expected-attention pairs.
- LagKV (Liang et al., 2025; arXiv:2504.04704) is an attention-free
  scoring rule that uses subsequent (lag) tokens to normalize and score
  earlier KV entries via a hardware-friendly token/channel statistic.

Reference defaults follow each paper's experimental section and are
mirrored in the NVIDIA `kvpress` library.

## Task

Modify only the `SelectionPolicy` class in
`transformers-kv-lab/custom_selection_eval.py`. Implement:

- `retention_plan(layer_id, request_meta, cache_meta)`
- `score_tokens(module, hidden_states, keys, values, kwargs, plan)`
- `select_cache(module, keys, values, scores, n_kept)`

The harness owns the model, datasets, prompt templates, cache budget,
decode loop, and scoring. The editable policy owns the retention metadata
and the per-token scoring rule used to rank prefill KV entries. The shared
hook exposes the union needed by the included source-backed baselines:

- no compression
- attention-sink/recent-window retention
- expected future-attention scoring
- LagKV lag-relative key/value scoring
- optional RoPE-aware key rerotation after pruning

The canonical compression setting is `SELECTION_KV_COMPRESSION_RATIO=0.8`,
meaning methods retain roughly 20% of prefill KV tokens unless the
`full_attention` anchor explicitly disables compression. Final scoring
applies a soft over-budget penalty once `mean_retained_fraction > 0.25`.

## Evaluation

The canonical model is aligned with `llm-kv-adaptive-quantization`:
`Qwen/Qwen2.5-3B-Instruct`. Workloads share the same public
text-benchmark protocol used by `llm-kv-adaptive-quantization`. Dataset
sources, split names, prompt templates, generation limits, and scoring
semantics stay aligned across those tasks whenever the same workload label
is used.

| Label | Source | Final score |
|---|---|---|
| `longbench_hotpotqa` | LongBench-E `hotpotqa_e` | LongBench QA F1 (0-100) |
| `longbench_passage_retrieval` | LongBench-E `passage_retrieval_en_e` | LongBench retrieval (0-100) |
| `longbench_repobench` | LongBench-E `repobench-p_e` | LongBench code similarity (0-100) |
| `longbench_v2` | LongBench v2 `train` split | multiple-choice exact accuracy (0-100), official head-tail truncation when over context |
| `gsm8k` | `openai/gsm8k` main test split | exact final-answer accuracy after numeric normalization (0-100) |

Canonical data sources are the public upstream datasets listed above:
`THUDM/LongBench`, `THUDM/LongBench-v2` / `zai-org/LongBench-v2`, and
`openai/gsm8k`. The runtime resolves model and datasets from the local
build-time or mounted Hugging Face cache only; missing assets are hard
failures rather than implicit network downloads.

`SELECTION_KV_MAX_EXAMPLES=0` means full available workload. Setting it to
a positive integer is allowed for local smoke validation but should not be
used for final leaderboard evidence.

## Baselines

Each baseline cites its paper and identifies the canonical reference
implementation it mirrors on the shared hook surface.

- `full_attention`: uncompressed full-cache reference anchor (HuggingFace
  `DynamicCache`).
- `streamingllm`: Xiao et al. (ICLR 2024; arXiv:2309.17453), "Efficient
  Streaming Language Models with Attention Sinks". Section 3.2 + Table 2
  defaults: `sink_tokens = 4`, recent-window inferred from the budget.
  RoPE-aware key rerotation following the paper's "Rolling KV Cache with
  Attention Sinks" formulation; reference re-rotation routine mirrors
  NVIDIA/kvpress `KeyRerotationPress` (audit commit 0.3.0). Original
  code: github.com/mit-han-lab/streaming-llm.
- `expected_attention`: Devoto, Jeblick, Jegou (NVIDIA), 2025
  (arXiv:2510.00636), "Expected Attention: KV Cache Compression by
  Estimating Attention from Future Queries Distribution". Equations 4-7
  for the score; defaults from the paper's experimental section
  (matched in NVIDIA/kvpress `ExpectedAttentionPress`, audit 0.3.0):
  `n_future_positions=512`, `n_sink=4`, covariance enabled, value-norm
  rescaling enabled.
- `lagkv`: Liang et al., 2025 (arXiv:2504.04704), "LagKV: Lag-Relative
  Information of the KV Cache Tells Which Tokens Are Important".
  Algorithm 1 (Section 3.2) defines the lag-relative score; defaults from
  paper Section 4.1 / Table 1 (mirrored in NVIDIA/kvpress `LagKVPress`,
  audit 0.3.0): `n_sink=4`, `lag_size=128`, `cross_scoring=False`.

### Harness enforcement vs. advisory metadata

The `retention_plan(...)` dict serves as the policy's internal
communication channel between `retention_plan` and `score_tokens`. The
harness enforces only what is observable from the post-`select_cache`
cache state:

| Field | Status | Notes |
|---|---|---|
| `compression_ratio` | enforced | Harness force-overrides to its own value at the call site (`PrefillSelectionCompressor.forward_hook`). Policies cannot lie about the budget. |
| `mean_retained_fraction` | measured, enforced | Computed from `select_cache`'s actual output `n_kept / keys.shape[2]` per layer, then averaged. Drives the soft budget penalty in `score_spec.py`. |
| `disable_compression` | enforced | If `True`, harness skips `score_tokens`/`select_cache` entirely and reports `retained = 1.0`. Used by the `full_attention` anchor. |
| `method` | logged only | Recorded for provenance; not used in scoring. |
| `sink_tokens`, `lag_size`, `n_future_positions`, `subspace_dim`, etc. | advisory | Used internally by the policy's own `score_tokens`. The harness does not verify that declared "sinks" are actually preserved by `select_cache`'s top-K output. Honesty here only matters for provenance and ablation reproducibility, not for scoring. |

Final scoring depends only on the end-to-end measured signals
(`final_score`, `mean_retained_fraction`, `runtime_seconds`).

## Metrics

The parser expects one `TEST_METRICS:` line per workload with:

- `final_score`: benchmark-native final task score on a 0-100 scale
- `mean_retained_fraction`: average retained prefill KV fraction after the
  policy runs
- `runtime_seconds`: workload wall-clock runtime in seconds

## Canonical Ranking

The leaderboard uses a single scalar computed from accuracy, runtime, and
cache reduction under the fixed retained-fraction constraint. Each workload
combines three normalized terms with weights `accuracy:time:reduction = 6:2:2`:

- `accuracy_score`: bounded 0-100 quality normalization calibrated against
  the visible baseline envelope
- `time_score`: soft lower-is-better sigmoid normalization of
  `runtime_seconds`, calibrated from the visible baseline runtime envelope
- `reduction_score`: bounded lower-is-better normalization of
  `mean_retained_fraction`

The per-workload score is the weighted mean of those three terms, and the
task score is the geometric mean across workloads. Rows whose
`mean_retained_fraction_*` exceeds the fixed budget tolerance receive a
soft upper-bound penalty, so the `full_attention` row remains a visible
reference anchor rather than a valid compressed-cache submission.
