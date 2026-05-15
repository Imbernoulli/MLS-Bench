"""Baseline: Expected Attention KV selection.

Algorithm: Devoto, Jeblick, Jegou (NVIDIA), "Expected Attention: KV Cache
Compression by Estimating Attention from Future Queries Distribution",
arXiv:2510.00636 (2025). Equations 4-7 derive the expected-attention score by
modelling future query states as a Gaussian centred on the mean of the prefill
queries (post-RoPE-averaged) with a per-layer covariance, then computing the
closed-form expected dot-product score against each cached key.

The paper's authors are NVIDIA-affiliated and released the canonical
reference implementation as `ExpectedAttentionPress` in NVIDIA/kvpress
(github.com/NVIDIA/kvpress, audit commit 0.3.0); no separate "official" repo
exists. The hyperparameter defaults below are from that reference impl, which
matches the experimental settings reported in the paper:

- `n_future_positions = 512`   (Section 4.1, "future window")
- `n_sink = 4`                  (carry over StreamingLLM-style sinks)
- `use_covariance = True`       (Equation 7 covariance correction)
- `use_value_norm = True`       (value-norm rescaling, Section 3.3)
- `epsilon = 0.0`
"""

_FILE = "transformers-kv-lab/custom_selection_eval.py"

_POLICY = """\
class SelectionPolicy:
    \"\"\"Expected Attention: estimate future-query attention before pruning.\"\"\"

    method_name = "expected_attention"
    rerotate_selected_keys = False

    def repeat_kv(self, hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states
        bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            bsz, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(bsz, num_key_value_heads * n_rep, slen, head_dim)

    def get_prerope_query_states(self, module, hidden_states):
        bsz, q_len, _ = hidden_states.shape
        num_heads = int(module.config.num_attention_heads)
        head_dim = int(module.head_dim)
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states)
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states)
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"Query projection not implemented for {module.__class__}.")
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        if hasattr(module, "q_norm"):
            query_states = module.q_norm(query_states)
        return query_states

    def avg_rope(self, module, mu, cov, q_len, n_future_positions):
        position_ids = torch.arange(q_len, q_len + n_future_positions, device=mu.device).unsqueeze(0)
        head_dim = int(module.head_dim)
        cos, sin = module.rotary_emb(mu, position_ids)
        cos, sin = cos[0], sin[0]
        identity = torch.eye(head_dim, device=cos.device, dtype=cos.dtype)
        perm = torch.zeros((head_dim, head_dim), device=cos.device, dtype=cos.dtype)
        half = head_dim // 2
        perm[half:, :half] = torch.eye(half, device=cos.device, dtype=cos.dtype)
        perm[:half, half:] = -torch.eye(half, device=cos.device, dtype=cos.dtype)
        rotation = (cos.unsqueeze(1) * identity + sin.unsqueeze(1) * perm).mean(dim=0).to(mu.device)
        mu = torch.matmul(mu, rotation.T)
        if cov is not None:
            cov = torch.matmul(rotation, torch.matmul(cov, rotation.T))
        return mu, cov

    def retention_plan(self, layer_id, request_meta, cache_meta):
        return {
            "method": self.method_name,
            "sink_tokens": 4,
            "n_future_positions": 512,
            "use_covariance": True,
            "use_value_norm": True,
            "epsilon": 0.0,
            "compression_ratio": cache_meta["compression_ratio"],
        }

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        n_sink = int(plan.get("sink_tokens", 4))
        n_future = int(plan.get("n_future_positions", 512))
        use_covariance = bool(plan.get("use_covariance", True))
        use_vnorm = bool(plan.get("use_value_norm", True))
        epsilon = float(plan.get("epsilon", 0.0))
        assert keys.size(2) > n_sink, f"Input should contain more tokens than sink_tokens={n_sink}"
        keys_body = keys[:, :, n_sink:]
        values_body = values[:, :, n_sink:]
        h = hidden_states[:, n_sink:]
        query_states = self.get_prerope_query_states(module, h)
        mean_query = query_states.mean(dim=2, keepdim=True)
        cov_query = None
        if use_covariance:
            centered_states = query_states - mean_query
            cov_query = torch.einsum("bnsi,bnsj->bnij", centered_states, centered_states) / max(h.shape[1], 1)
        mean_query = mean_query.squeeze(2)
        mean_query, cov_query = self.avg_rope(module, mean_query, cov_query, hidden_states.shape[1], n_future)
        bsz, num_key_value_heads, q_len, dim = keys_body.shape
        num_key_value_groups = int(module.config.num_attention_heads) // num_key_value_heads
        repeated_keys = self.repeat_kv(keys_body, num_key_value_groups).transpose(2, 3)
        scores = torch.matmul(mean_query.unsqueeze(2), repeated_keys).squeeze(2) / math.sqrt(dim)
        if use_covariance:
            scores += torch.einsum("bhin,bhij,bhjn->bhn", repeated_keys, cov_query, repeated_keys) / dim / 2
        scores = F.softmax(scores, dim=-1)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len).mean(dim=2)
        if use_vnorm:
            scores = (scores + epsilon) * values_body.norm(dim=-1)
        return F.pad(scores, (n_sink, 0), value=scores.max().item())

    def select_cache(self, module, keys, values, scores, n_kept):
        indices = scores.topk(n_kept, dim=-1).indices
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        selected_keys = keys.gather(2, gather_idx).contiguous()
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
