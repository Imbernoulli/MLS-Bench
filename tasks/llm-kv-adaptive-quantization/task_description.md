# LLM KV Cache: Adaptive Quantization Policy

## Research Question

Design an adaptive low-bit KV-cache quantizer for decoder-only LLM
inference on top of a tensor-level Hugging Face `Transformers` replay
harness. The task asks whether an algorithm can preserve benchmark output
quality while reducing the effective KV footprint through bit allocation,
axis selection, residual windows, and optional prefill-time observation.

## Background

KV-cache quantization compresses the running key/value tensors used during
decode rather than the model weights. Three reference families inform the
design space exposed by this task:

- KIVI (Liu et al., ICML 2024; arXiv:2402.02750) is a tuning-free 2-bit KV
  quantizer that quantizes the key cache per-channel and the value cache
  per-token, with a residual tail of recent FP entries to preserve the
  freshly generated rows.
- KVTuner (Li et al., 2025; arXiv:2502.04420) treats KV quantization as a
  layer-wise mixed-precision search, picking different bit-widths per layer
  based on sensitivity. Its Qwen2.5-3B configurations are used directly as
  baseline presets for this task.
- SQuat (Zhang et al., 2025; arXiv:2503.24358) builds a query subspace via
  SVD during prefill and enforces that the (de)quantized key error is
  orthogonal to that subspace, reducing attention-output drift from key
  quantization.

The harness exposes the union of behaviors needed to implement these
methods on a common tensor-level interface, so the contribution is a
quantization policy rather than a backend choice.

## What You Can Modify

The editable region is the `AdaptiveKVQuantizer` class in
`custom_quant_eval.py`. The fixed harness supplies real KV-cache tensors and
calls the editable class:

- `reset_request(request_meta, budget_state)` before each example
- `needs_prefill_qkv_observer() -> bool`
- `query_observation_position() -> str`
- `observe_prefill_qkv(layer_id, query_states, key_states, value_states, attention_meta)`
- `quantize_key(layer_id, key_states, cache_meta) -> tensor | (tensor, layout)`
- `quantize_value(layer_id, value_states, cache_meta) -> tensor | (tensor, layout)`

`key_states` and `value_states` have shape
`[batch, heads, seq_len, head_dim]`. The class implements the actual tensor
algorithm: grouping, asymmetric ranges, zero-points, per-layer bit presets,
residual retention, and query-subspace transforms all belong inside this
class. The task does not expose a fixed algorithm enum or a backend selector.

KV memory is measured by the fixed harness, not self-reported. `layout` is
`{"group_ids": ..., "group_bits": ...}`: `group_ids` is an integer tensor
with the returned tensor's shape that maps every element to a quantization
group (`-1` = kept at FP16), and `group_bits` is a 1-D integer tensor of
per-group bit-widths in `[1, 16]`. Every used group must exactly fill an
axis-aligned box of the returned tensor (for example a run of tokens in one
channel, a run of channels in one token, or a tile), hold at least 32
elements, and hold at most `2**bits` distinct stored values, otherwise the
run aborts; an arbitrary element-to-group map would be uncharged side
information. Returning a bare tensor (or a `None` layout) charges every
element at FP16. Scale/zero-point metadata is not charged. Bits are counted
on the returned tensor in its own coordinates, the values attention reads: a
transform-domain quantizer (for example a rotation) whose inverse-transformed
values are not `2**bits` distinct values per group must declare more bits,
and a 32-element group of arbitrary values needs 5.

## What You Cannot Modify

- The model family and deterministic decode replay loop
- The benchmark workload definitions, prompts, generation limits
- The parser or final-score definitions
- The underlying `Transformers` model implementation

## Evaluation

The visible model is `Qwen/Qwen2.5-3B-Instruct`. Workloads share the same
public text-benchmark protocol with `llm-kv-selection-budgeting`; dataset
sources, split names, prompt templates, generation limits, and scoring
semantics stay aligned across those tasks whenever a workload label is
shared.

| Label | Source | Final score |
|---|---|---|
| `longbench_hotpotqa` | LongBench-E `hotpotqa_e` | LongBench QA F1 (0-100) |
| `longbench_passage_retrieval` | LongBench-E `passage_retrieval_en_e` | LongBench retrieval score (0-100) |
| `longbench_repobench` | LongBench-E `repobench-p_e` | LongBench code similarity (0-100) |
| `needlebench_niah` | RULER/NeedleBench-style needle in public essay text | exact phrase retrieval accuracy (0-100) |
| `gsm8k` | `openai/gsm8k` main test split | exact final-answer accuracy after numeric normalization (0-100) |

For NIAH the canonical needle is:
`The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.`

The parser expects one `TEST_METRICS:` line per workload with:

- `final_score`: benchmark-native quality on a 0-100 scale
- `effective_kv_bits`: effective KV bits per cached element, measured by the
  harness from the stored tensors and their layouts
- `kv_compression_ratio`: `16 / effective_kv_bits`, using FP16 KV as the
  reference footprint
- `runtime_seconds`: task-level wall-clock runtime for the workload command

`effective_kv_bits` is measured at a 4096-token reference KV span so the
efficiency term is hardware-independent: after each request the harness
quantizes that request's real KV cache, cropped or tiled to 4096 tokens,
with the submitted quantizer and measures the bits. The cache used for
generation is measured too: at one uniformly random decode step per
request (the quantizer cannot tell which), the harness measures the bits the
returned tensors store on their older tokens, all but the most recent 160.
The reference span's older tokens (its first 3936) are charged at the larger
of their own bits and these decode-time bits, so storing less at the
reference length than during decoding gains nothing; only the most recent
160 tokens are accounted at the reference span alone. The per-request bits
are averaged over requests.

## Baselines

Baselines are paper-linked configurations implemented on the same tensor
interface; only the algorithm is the contribution, not a paper repository
backend.

- `kivi_overlap_4bit`: KIVI-style K4/V4 (arXiv:2402.02750) with key
  per-channel, value per-token, group size `32`, key `block_modulo`
  residual blocks, value tail residual length `128`.
- `kvtuner4_pertoken_qwen25_3b`: KVTuner (arXiv:2502.04420)
  `Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml` preset, signed-asymmetric
  vanilla cache formula, axis `0/0`, `q_group_size=-1`, `residual_length=0`.
- `kvtuner4_kivi_qwen25_3b`: KVTuner
  `Qwen2.5-3B-Instruct_kivi_KVTuner4_0.yaml` preset, signed-asymmetric
  formula, axis `1/0`, `block_modulo` residual length `32`,
  `q_group_size=32`.
- `squat_subspace_4bit`: SQuat (arXiv:2503.24358) LongBench-style 4-bit
  configuration: query SVD during prefill, future-dimension key correction,
  subspace dimension `60`, `squat_lambda=0.001`, `quant_group_size=64`,
  `shared_svd=True`, K/V group size `32`, residual block length `32`.

## Canonical Ranking

The leaderboard uses the standard mature MLS-Bench text-task pattern:

- each workload emits a benchmark-native `final_score_*` quality column
- each workload also emits a hardware-independent `kv_compression_ratio_*`
  efficiency column
- quality uses the repository standard bounded-power normalization with the
  worst current baseline as the floor, `100` as the bound, and the best
  current baseline as the reference point
- efficiency uses bounded-power normalization on `kv_compression_ratio` with
  the worst current baseline as the floor, `4x` compression as the
  reference, and `8x` compression as the bound
- `runtime_seconds_*` is an emitted diagnostic column but does not enter
  the final score (this tensor-replay harness is not a runtime-native
  packed-cache speed benchmark)
- each workload score is a weighted mean with quality weight `6` and KV
  efficiency weight `4`, multiplied by a quality gate: below 90% of the
  worst current baseline's quality on that workload the score is scaled by
  `exp(-s * (target - quality))`, with `s` set so that quality 0 gives
  `x0.01`, so compression cannot score when output quality collapses
- the task score is the geometric mean across the LongBench-E workloads,
  NIAH, and GSM8K

## Notes

- The harness runs deterministic greedy generation over `Transformers`
  decode steps and scores generated answers.
- At each decode step after prefill, it snapshots real KV tensors,
  quantizes them with the current quantizer, restores the quantized cache,
  and advances generation.
