"""Vanilla no-cache control on the shared DLM cache-hook surface."""

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
    \"\"\"No-cache control: full LLaDA denoising forward every step.\"\"\"

    policy_name = "vanilla_uncached"

    def block_schedule(self, request_meta):
        wl = WORKLOAD_CONFIGS[request_meta["workload"]]
        return {
            "gen_length": wl["gen_length"],
            "block_length": wl["block_length"],
            "num_steps": wl["num_steps"],
            "warmup_forward": False,
        }

    def query_plan(self, step_meta, mask_state, cache_state):
        return {
            "query_scope": "full_sequence",
            "query_positions": None,
            "track_positions": [],
            "masked_window": None,
        }

    def cache_refresh_plan(self, layer_meta, step_meta, token_stats, cache_state):
        return {
            "use_feature_cache": False,
            "prompt_refresh_interval": 1,
            "gen_refresh_interval": 1,
            "transfer_ratio": 0.0,
            "row_selector": "none",
            "kv_update": "full_refresh",
            "layer_reset": None,
        }

    def attention_probe_plan(self, layer_meta, step_meta):
        return {
            "need_attention_weights": False,
            "rollout_p": 0.0,
            "current_k": 0,
            "gamma": None,
            "track_num": 0,
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
