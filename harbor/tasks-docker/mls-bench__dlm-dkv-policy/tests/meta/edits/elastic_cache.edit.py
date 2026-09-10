"""Elastic-Cache policy on the shared DLM hook surface."""

import importlib.util
from pathlib import Path

_FILE = "dLLM-cache/custom_dlm_eval.py"

_SPEC = importlib.util.spec_from_file_location(
    "_policy_span", Path(__file__).with_name("_policy_span.py")
)
_POLICY_SPAN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_POLICY_SPAN)
_START_LINE, _END_LINE = _POLICY_SPAN.policy_span()

_POLICY = """\
class DLMRefreshPolicy:
    \"\"\"Elastic-Cache: tracked-token windows with attention-similarity reset.\"\"\"

    policy_name = "elastic_cache"

    def block_schedule(self, request_meta):
        wl = WORKLOAD_CONFIGS[request_meta["workload"]]
        return {
            "gen_length": wl["gen_length"],
            "block_length": wl["block_length"],
            "window_length": 16,
            "num_steps": wl["num_steps"],
            "warmup_forward": False,
        }

    def query_plan(self, step_meta, mask_state, cache_state):
        return {
            "query_scope": "full_sequence" if step_meta["step"] == 0 else "tracked_window",
            "query_positions": None,
            "track_positions": cache_state.get("track_positions", []),
            "masked_window": (mask_state["block_start"], mask_state["block_end"]),
        }

    def cache_refresh_plan(self, layer_meta, step_meta, token_stats, cache_state):
        return {
            "use_feature_cache": False,
            "prompt_refresh_interval": 1,
            "gen_refresh_interval": 1,
            "transfer_ratio": 0.0,
            "row_selector": "tracked_tokens_and_masked_window",
            "kv_update": "tracked_window_layer_reset",
            "layer_reset": "attention_similarity",
        }

    def attention_probe_plan(self, layer_meta, step_meta):
        return {
            "need_attention_weights": True,
            "rollout_p": 0.0,
            "current_k": 0,
            "gamma": 0.9,
            "track_num": 1,
        }

    def token_transfer_plan(self, logits, mask_state, step_meta):
        return {
            "mode": "confidence_threshold",
            "scope": "masked_window",
            "num_transfer_tokens": 1,
            "threshold": 0.9,
            "force_one": True,
        }

    def after_step(self, step_meta, logits, attention_stats, transfer_state, cache_state):
        return cache_state
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": _START_LINE,
        "end_line": _END_LINE,
        "content": _POLICY,
    },
]
