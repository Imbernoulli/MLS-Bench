"""Baseline: LagKV lag-relative KV selection.

Algorithm: LagKV (Liang, Zhang, Li, Li; arXiv:2504.04704, 2025). Section 3.2
(Algorithm 1) defines the
lag-relative score: partition the prefill cache into contiguous lag windows of
size `lag_size = 128`, then for each token in window `i` compute its key/value
range standard deviation NORMALIZED against the min/max envelope from the NEXT
window (the "lag"), softmax across the window, and average the per-key and
per-value scores. Argsort-rank the resulting score (without cross-window
scoring) to obtain the per-token importance rank.

LagKV's canonical reference implementation lives in NVIDIA/kvpress as
`LagKVPress` (github.com/NVIDIA/kvpress, audit commit 0.3.0); the paper does
not ship a separate official repo. Defaults below match both the kvpress
reference impl and the paper's reported settings (Section 4.1, Table 1):

- `n_sink = 4`           (carry-over attention sink)
- `lag_size = 128`       (Section 4.1 "lag partition size")
- `cross_scoring = False` (use within-window argsort ranks)
"""

_FILE = "transformers-kv-lab/custom_selection_eval.py"

_POLICY = """\
class SelectionPolicy:
    \"\"\"LagKV: score tokens by lag-relative key/value variation.\"\"\"

    method_name = "lagkv"
    rerotate_selected_keys = False

    def retention_plan(self, layer_id, request_meta, cache_meta):
        return {
            "method": self.method_name,
            "sink_tokens": 4,
            "lag_size": 128,
            "cross_scoring": False,
            "compression_ratio": cache_meta["compression_ratio"],
        }

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        bsz, num_key_value_heads, q_len, dim = keys.shape
        n_sink = int(plan.get("sink_tokens", 4))
        lag_size = int(plan.get("lag_size", 128))
        if q_len < n_sink + 2 * lag_size:
            scores = torch.ones((bsz, num_key_value_heads, q_len), dtype=keys.dtype, device=keys.device)
            if q_len > n_sink:
                scores[:, :, n_sink:] = (
                    torch.arange(q_len - n_sink, device=keys.device) / (q_len - n_sink)
                ).to(keys.dtype)
            return scores
        end_idx = n_sink + ((q_len - n_sink) // lag_size) * lag_size
        tail_len = lag_size + q_len - end_idx

        def state_score(target):
            ref = target[:, :, 1:, :, :]
            value = target[:, :, :-1, :, :]
            min_ref = ref.min(dim=-2).values.unsqueeze(-2).expand_as(value)
            max_ref = ref.max(dim=-2).values.unsqueeze(-2).expand_as(value)
            return ((value - min_ref) / (max_ref - min_ref)).std(dim=-1).softmax(dim=-1)

        key_score = state_score(keys[:, :, n_sink:end_idx].view(bsz, num_key_value_heads, -1, lag_size, dim))
        value_score = state_score(values[:, :, n_sink:end_idx].view(bsz, num_key_value_heads, -1, lag_size, dim))
        scores = (key_score + value_score) / 2
        if not bool(plan.get("cross_scoring", False)):
            scores = scores.argsort(dim=-1).argsort(dim=-1) / lag_size
            scores = scores.to(keys.dtype)
        sink_scores = torch.ones((bsz, num_key_value_heads, n_sink), dtype=scores.dtype, device=scores.device)
        tail_scores = torch.ones((bsz, num_key_value_heads, tail_len), dtype=scores.dtype, device=scores.device)
        return torch.cat((sink_scores, scores.reshape(bsz, num_key_value_heads, -1), tail_scores), dim=-1)

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
