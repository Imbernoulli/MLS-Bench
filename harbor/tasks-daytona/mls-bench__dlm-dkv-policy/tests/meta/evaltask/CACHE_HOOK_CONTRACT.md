# DLM Cache Hook Contract

This note records the shared editable surface for `dlm-dkv-policy`. It replaces
the rejected backend-selection design. The task should expose one LLaDA
denoising rollout and let each policy control the cache mechanisms that the
audited paper implementations actually change.

## Design Rule

Participants should not select a named cache backend. They should implement a
semantic cache plan over a shared rollout. The harness owns model loading,
benchmark data, scoring, and the fixed denoising loop; the policy owns only the
mechanism decisions below.

Official repositories are retained as fidelity oracles and source references.
They are not the runtime API exposed to participants.

The fixed harness can use task-local model compatibility classes when the base
LLaDA class lacks a required hook argument. That adapter choice is derived from
semantic hook requirements such as `active_q_mask` or tracked-token
`positions`, and it must not call the paper repository's generation entry
point.

## Hook Union

The union of the audited implementations opens these hook points:

| Hook | Purpose | Required by |
|---|---|---|
| `block_schedule(request_meta)` | Generation length, block length, and per-block step count. | dLLM-Cache, d2Cache |
| `query_plan(step_meta, mask_state, cache_state)` | Which token positions are forwarded this step; supports full sequence, active query rows, tracked tokens, and sliding masked windows. | d2Cache, Elastic-Cache |
| `cache_refresh_plan(layer_meta, step_meta, token_stats, cache_state)` | Per-layer/per-segment recompute vs reuse; supports prompt/gen interval refresh, selected-row refresh, cached-row update, and layer reset. | dLLM-Cache, d2Cache, Elastic-Cache |
| `attention_probe_plan(layer_meta, step_meta)` | Whether attention weights or similarity probes are needed, and the parameters used by those probes. | d2Cache, Elastic-Cache |
| `token_transfer_plan(logits, mask_state, step_meta)` | Which masked tokens are committed back to the global sequence. | d2Cache, Elastic-Cache |
| `after_step(step_meta, logits, attention_stats, transfer_state, cache_state)` | Update rollout state such as active query masks, attention rollout, tracked tokens, density scores, and layer reset boundaries. | d2Cache, Elastic-Cache |

The compatibility class may keep the historical name `DLMRefreshPolicy`, but its
semantics must be a cache-plan policy, not backend dispatch.

## Baseline Mapping

| Baseline | Official mechanism to reproduce on the shared hooks |
|---|---|
| `vanilla_uncached` | Full prompt+generation sequence forward every denoising step, no cache reuse, standard low-confidence token transfer. |
| `dllm_cache` | Keep the full top-level LLaDA rollout, split hidden states into prompt and generated segments, refresh prompt/gen segments by interval, reuse cached attention/MLP features, and refresh the generated rows with lowest value-feature cosine similarity. |
| `d2cache` | Preserve the frame/delta outer loop, narrow later forwards to `active_q_mask`, reuse cached K/V for inactive rows, request eager attention weights, accumulate attention rollout/global importance, and top up active positions using certainty density. |
| `elastic_cache` | Forward the union of tracked tokens, newly decoded tokens, and a masked query window; maintain block-local `x/q/k/v` caches; update tracked tokens from masked attention top-k; reset deeper layer caches when tracked-token attention similarity falls below `gamma`. |

## Fidelity Oracles

Each paper-backed baseline needs a source-level oracle before leaderboard rows
are trusted:

- `dllm_cache`: cached output should match the same dLLM-Cache LLaDA path with
  feature cache disabled for fixed prompts and seeds.
- `d2cache`: compare trace state rather than no-cache equality: `active_q_mask`,
  attention rollout, transfer index, generated tokens, and final decoded text.
- `elastic_cache`: compare `track_position`, per-step `start_reset`, decoded
  positions, final tokens, and the official compute-ratio heuristic.

If a baseline cannot satisfy its oracle on a fixed smoke set, it should not be
advertised as a faithful paper reproduction.
