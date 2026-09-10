# Adaptive Quantization Hook Contract

## Contract

The participant edits only `AdaptiveKVQuantizer` in `custom_quant_eval.py`.
The fixed harness owns data loading, greedy generation, DynamicCache snapshot
and restore, scoring, and result emission. The editable class owns the KV-cache
quantization algorithm itself.

The harness calls these methods:

- `reset_request(request_meta, budget_state)` before each example
- `needs_prefill_qkv_observer() -> bool`
- `query_observation_position() -> str`
- `observe_prefill_qkv(layer_id, query_states, key_states, value_states, attention_meta)`
- `quantize_key(layer_id, key_states, cache_meta) -> tensor | (tensor, avg_bits)`
- `quantize_value(layer_id, value_states, cache_meta) -> tensor | (tensor, avg_bits)`
- `estimate_bits(layer_id, kv_kind, seq_len, head_dim, cache_meta) -> float`

`key_states` and `value_states` are real tensors from the model cache with shape
`[batch, heads, seq_len, head_dim]`. A quantizer may implement grouping,
residual windows, per-layer bit presets, asymmetric zero-points, query-subspace
observation, reordering, or other tensor transforms inside the editable class.
The returned tensor must preserve the input shape.

## Source-Fidelity Mapping

- KIVI source: `github.com/jy-yuan/KIVI` at audit commit `876b4d2`.
  `kivi_overlap_4bit` implements K/V 4-bit min/max fake quantization directly
  in `AdaptiveKVQuantizer`, with key per-channel grouping, value per-token
  grouping, group size 32, key block-modulo residual blocks, and value tail
  residual length 128.
- KVTuner source: `github.com/cmd2001/KVTuner` at audit commit `96dd05e`.
  `kvtuner4_pertoken_qwen25_3b` implements a KVTuner-style
  `Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml` layer preset with the
  FlexibleVanillaQuantizedCache signed-asymmetric formula, axis 0/0,
  `q_group_size=-1`, and no residual window.
- KVTuner source: `github.com/cmd2001/KVTuner` at audit commit `96dd05e`.
  `kvtuner4_kivi_qwen25_3b` implements a KVTuner-style
  `Qwen2.5-3B-Instruct_kivi_KVTuner4_0.yaml` layer preset with the same
  signed-asymmetric formula, key axis 1, value axis 0, `q_group_size=32`, and
  block-modulo residual length 32.
- SQuat source: `github.com/Red-Hat-AI-Innovation-Team/SQuat` at audit commit
  `1ba3495`. `squat_subspace_4bit` implements the 4-bit
  query-subspace quantizer inside `AdaptiveKVQuantizer`: query SVD during
  prefill, subspace dimension 60, `squat_lambda=0.001`, future-dimension key
  correction, `quant_group_size=64`, shared SVD, grouped value quantization, and
  residual block length 32.
