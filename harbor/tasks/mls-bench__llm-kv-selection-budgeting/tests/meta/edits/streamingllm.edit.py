"""Baseline: StreamingLLM attention-sink retention.

Algorithm: Xiao et al., "Efficient Streaming Language Models with Attention
Sinks", ICLR 2024 (arXiv:2309.17453). Section 3.2 introduces the attention-sink
mechanism: keep the first `n_sink` tokens (default 4) plus the most recent
window, dropping the middle. The original reference implementation lives at
github.com/mit-han-lab/streaming-llm.

Hyperparameters (Section 3.2 / Table 2): `sink_tokens = 4`. Recent-window size
is implied by the budget — here `compression_ratio = 0.8` retains roughly the
last 20% of tokens after the 4 sinks.

RoPE re-rotation: when streaming-style retention drops middle tokens, the kept
keys must have their rotary embedding re-applied at their NEW positions for the
attention scores to remain coherent. Section 3.2 of the StreamingLLM paper
("Rolling KV Cache with Attention Sinks") formalizes this; the canonical
implementation in NVIDIA/kvpress (`KeyRerotationPress`, audit commit 0.3.0)
provides a reference re-rotation routine that this baseline mirrors.
"""

_FILE = "transformers-kv-lab/custom_selection_eval.py"

_POLICY = """\
class SelectionPolicy:
    \"\"\"StreamingLLM: keep attention sinks and the most recent tokens.\"\"\"

    method_name = "streamingllm"
    rerotate_selected_keys = True

    def retention_plan(self, layer_id, request_meta, cache_meta):
        return {
            "method": self.method_name,
            "sink_tokens": 4,
            "compression_ratio": cache_meta["compression_ratio"],
        }

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        k_len = int(keys.shape[2])
        n_sink = int(plan.get("sink_tokens", 4))
        ratio = float(plan["compression_ratio"])
        assert k_len > n_sink, f"Input should contain more tokens than sink_tokens={n_sink}"
        n_pruned = k_len - int(k_len * (1.0 - ratio))
        scores = torch.ones_like(keys[..., 0])
        scores[:, :, n_sink : n_sink + n_pruned] = 0
        return scores

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def rerotate_cache_keys(self, module, indices, keys):
        bsz, num_key_value_heads, n_kept = indices.shape
        device = indices.device
        device_type = keys.device.type
        dtype = keys.dtype
        inv_freq = module.rotary_emb.inv_freq[None, None, :, None].float().expand(
            bsz, num_key_value_heads, -1, 1
        )
        new_positions = torch.arange(0, n_kept, device=device).unsqueeze(0)[:, None, :].float()
        new_positions = new_positions.expand(bsz, num_key_value_heads, n_kept)
        delta_pos = (new_positions - indices.float()).unsqueeze(2)
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (delta_pos.float() * inv_freq.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().contiguous()
            sin = emb.sin().contiguous()
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        gathered = keys.gather(2, gather_idx).contiguous()
        return (gathered * cos) + (self.rotate_half(gathered) * sin)

    def select_cache(self, module, keys, values, scores, n_kept):
        indices = scores.topk(n_kept, dim=-1).indices
        indices = torch.sort(indices, dim=2).values
        selected_keys = self.rerotate_cache_keys(module, indices, keys)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
        selected_values = values.gather(2, gather_idx).contiguous()
        return selected_keys, selected_values
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 40,
        "end_line": 101,
        "content": _POLICY,
    },
]
