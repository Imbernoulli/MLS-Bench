"""d2Cache policy on the shared DLM hook surface."""

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
    \"\"\"d2Cache: active query rows plus attention-rollout top-up.\"\"\"

    policy_name = "d2cache"

    def block_schedule(self, request_meta):
        wl = WORKLOAD_CONFIGS[request_meta["workload"]]
        return {
            "gen_length": wl["gen_length"],
            "block_length": wl["gen_length"],
            "num_steps": wl["gen_length"],
            "warmup_forward": False,
        }

    def query_plan(self, step_meta, mask_state, cache_state):
        return {
            "query_scope": "full_sequence" if step_meta["step_in_block"] == 0 else "active_query_rows",
            "query_positions": cache_state.get("active_q_mask"),
            "track_positions": [],
            "masked_window": (mask_state["block_start"], mask_state["block_end"]),
        }

    def cache_refresh_plan(self, layer_meta, step_meta, token_stats, cache_state):
        return {
            "use_feature_cache": False,
            "prompt_refresh_interval": 1,
            "gen_refresh_interval": 1,
            "transfer_ratio": 0.0,
            "row_selector": "certainty_density_attention_rollout",
            "kv_update": "active_q_mask",
            "layer_reset": None,
        }

    def attention_probe_plan(self, layer_meta, step_meta):
        return {
            "need_attention_weights": True,
            "rollout_p": 0.1,
            "current_k": 32,
            "gamma": None,
            "track_num": 0,
            "sigma": 10.0,
            "inflate_w": 0,
        }

    def token_transfer_plan(self, logits, mask_state, step_meta):
        return {
            "mode": "low_confidence",
            "scope": "current_block",
            "num_transfer_tokens": step_meta["default_num_transfer_tokens"],
            "threshold": None,
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
