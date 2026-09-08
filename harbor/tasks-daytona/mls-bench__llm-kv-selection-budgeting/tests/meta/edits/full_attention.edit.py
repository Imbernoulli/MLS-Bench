"""Baseline: naive full-attention KV cache.

Reference:
- Hugging Face Transformers DynamicCache without KV eviction.
Fidelity:
- exact no-compression anchor
"""

_FILE = "transformers-kv-lab/custom_selection_eval.py"

_POLICY = """\
class SelectionPolicy:
    \"\"\"Naive full-attention anchor: keep every prefill KV token.\"\"\"

    method_name = "full_attention"
    rerotate_selected_keys = False

    def retention_plan(self, layer_id, request_meta, cache_meta):
        return {
            "method": self.method_name,
            "disable_compression": True,
        }

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        return None

    def select_cache(self, module, keys, values, scores, n_kept):
        return keys, values
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
