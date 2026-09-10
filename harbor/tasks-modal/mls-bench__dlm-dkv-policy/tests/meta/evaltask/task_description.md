# Diffusion LM KV Cache Policy

## Research Question

Design a cache policy for diffusion language-model inference. Given a fixed
LLaDA-8B-Instruct host model and public final-task benchmarks, can a method
preserve benchmark accuracy while reusing KV state during the bidirectional
denoising rollout?

## Background

LLaDA (Large Language Diffusion Models, Nie et al., 2025; arXiv:2502.09992)
predicts masked tokens with a Transformer that attends bidirectionally over
the entire sequence at every denoising step. Unlike autoregressive models,
the standard prefix-only KV cache is not directly reusable, because keys and
values for previously committed tokens can keep changing as more tokens are
unmasked. A growing line of work studies how to nonetheless reuse cached
features: dLLM-Cache (arXiv:2506.06295) refreshes prompt features on a long
interval and recomputes only low-similarity generated rows; d2Cache
(arXiv:2509.23094) uses a two-stage selection that combines an active query
mask with an attention rollout / certainty-density top-up to decide which
tokens to re-encode each step; and Elastic-Cache (Nguyen et al., ICLR 2026;
arXiv:2510.14973) uses an attention-aware drift test on the most-attended
token to decide when to refresh, plus a depth-aware schedule that recomputes
only deeper layers from a chosen layer onward.

The task isolates this design space onto a shared cache-control surface so a
policy can be evaluated end-to-end on real LLaDA generation rather than on a
proxy token-trajectory metric.

## Evaluation Setup

The harness runs real `LLaDA-8B-Instruct` inference end to end. For each
workload it:

1. Loads the public benchmark dataset.
2. Runs one fixed denoising rollout with a shared cache-plan interface.
3. Generates deterministic outputs using the submitted cache policy.
4. Scores the generated outputs with benchmark-native final-task metrics.
5. Emits the benchmark-native final score.

The task is not a backend-selection problem: paper baselines are implemented
through the same cache-control surface rather than called as black-box
generation backends. Some cache mechanisms require additional LLaDA forward
arguments such as active query rows or tracked-token positions; the harness
may load task-local compatibility model classes to expose those forward
hooks, but the outer rollout remains policy-driven and does not call paper
repository generation functions.

## Editable Surface

You may edit only the policy class in `dLLM-cache/custom_dlm_eval.py`. The
compatibility class name is `DLMRefreshPolicy`, but semantically it is a DLM
cache-plan policy.

The required hook families are:

| Method | Purpose |
|---|---|
| `block_schedule(request_meta)` | Controls generation length, block length, steps per block, and whether a block starts with a full warm forward. |
| `query_plan(step_meta, mask_state, cache_state)` | Selects token positions to forward or recompute: full sequence, current block, active query rows, tracked tokens, or a masked query window. |
| `cache_refresh_plan(layer_meta, step_meta, token_stats, cache_state)` | Decides per-layer recompute/reuse, prompt-vs-generation refresh, selected row refresh, KV overwrite, and layer reset. |
| `attention_probe_plan(layer_meta, step_meta)` | Requests attention weights or attention-similarity probes and supplies parameters such as rollout fraction, `current_k`, `gamma`, and `track_num`. |
| `token_transfer_plan(logits, mask_state, step_meta)` | Chooses which masked tokens are committed back to the global denoising state. |
| `after_step(step_meta, logits, attention_stats, transfer_state, cache_state)` | Updates state such as active query masks, attention rollout, tracked tokens, density scores, and layer reset boundaries. |

The full hook contract and baseline mapping are recorded in
`CACHE_HOOK_CONTRACT.md`.

## Fixed Components

Participants may not modify:

- the model weights or tokenizer
- benchmark loaders and scorers
- task scripts, parser, score spec, or leaderboard schema
- source-reference snapshots under `third_party/official_dlm_cache_baselines`
- any harness code outside the editable policy region

Each baseline uses one predeclared cache policy across all workloads to
avoid rewarding per-benchmark hyperparameter search.

## Workloads

| Label | Workload | Public source | Final metric |
|---|---|---|---|
| `math` | MATH-500 test split | exact final-answer accuracy |
| `humaneval` | OpenAI HumanEval | pass@1 execution accuracy |
| `lm-eval` | ARC-Challenge test split | exact answer-letter accuracy |

All examples in the selected public splits are evaluated by default.

## Metrics

Each script prints one `TEST_METRICS:` line. The parser records the benchmark
score and runtime diagnostics:

| Metric | Direction | Meaning |
|---|---|---|
| `final_score` | higher | benchmark-native final task score on a 0-100 scale |
| `reuse_ratio` | higher | diagnostic fraction of generated-token cache work reused by the hook plan |
| `refresh_ratio` | lower | diagnostic `1 - reuse_ratio` |
| `tokens_per_s` | higher | diagnostic decode throughput on the current hardware |
| `peak_memory_mb` | lower | diagnostic peak GPU memory allocated during the example loop |
| `n_examples` | fixed | number of examples evaluated |
| `elapsed` | lower | diagnostic wall-clock time recorded by the harness for the script |

`final_score` is the canonical quality metric. `reuse_ratio` and
`tokens_per_s` enter the scalar ranking because the task is a cache-policy
benchmark: methods should preserve final-task quality while reducing
redundant denoising work and improving decode throughput.

## Canonical Ranking

The score in `score_spec.py` follows the MLS-Bench efficiency-task pattern:

- each workload applies `final_score_*` as a near-lossless soft quality gate
- once the quality gate is satisfied, small benchmark-native score
  differences are not rewarded further
- each workload ranks cache reuse and decode throughput as efficiency terms
- throughput is normalized against the visible baseline envelope rather than
  a hard hardware-specific pass/fail range
- the task score is the geometric mean across the three workloads

## Baselines

| Baseline | Source |
|---|---|
| `vanilla_uncached` | no-cache LLaDA control: full denoising forward every step |
| `dllm_cache` | dLLM-Cache (arXiv:2506.06295), `maomaocun/dLLM-cache`: prompt/generation feature refresh and low-similarity generated-row update |
| `d2cache` | d2Cache (arXiv:2509.23094), `Kamichanw/d2Cache`: active query mask, eager attention rollout, and certainty-density top-up |
| `elastic_cache` | Elastic-Cache (arXiv:2510.14973), `VILA-Lab/Elastic-Cache`: tracked-token query window and attention-similarity layer reset |

Task-local source snapshots and commit hashes are documented in
`third_party/official_dlm_cache_baselines/NOTICE.md`.

Small compatibility shims may be used for model loading and source-oracle
checks when a hook requires extra forward arguments. They are capability
adapters, not participant-facing backend choices; canonical task behavior
must still be explained in terms of the shared hook contract.
