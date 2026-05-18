"""Real LLaDA rollout harness for dlm-dkv-policy."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import torch

# dLLM-cache path resolution
_HERE = Path(__file__).resolve().parent

def _find_dllm_cache_root() -> Path:
    env = os.environ.get('DLLM_CACHE_DIR', '')
    if env and (Path(env) / 'dllm_cache').is_dir():
        return Path(env)
    for p in [Path.cwd(), _HERE, _HERE.parent]:
        if (p / 'dllm_cache').is_dir():
            return p
    raise RuntimeError('dLLM-cache not found. Set DLLM_CACHE_DIR env var.')

_DLLM_ROOT = _find_dllm_cache_root()
if str(_DLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DLLM_ROOT))
_MODEL_DIR = os.environ.get('LLADA_MODEL_DIR', 'LLaDA-8B-Instruct')
_TASK_SLUG = 'dlm-dkv-policy'

REGIMES = {"final": {}}

WORKLOAD_CONFIGS = {
    "math":      {"source_family": "MATH-500",      "num_steps": 256, "gen_length": 256, "block_length": 32},
    "humaneval": {"source_family": "HumanEval",     "num_steps": 512, "gen_length": 512, "block_length": 32},
    "lm_eval":   {"source_family": "ARC-Challenge", "num_steps": 64,  "gen_length": 64,  "block_length": 32},
}

# Editable region: DLMRefreshPolicy. config.json must match this marker-delimited
# span; baseline edit files fail fast if the configured window drifts.
MASK_ID = 126336  # LLaDA mask token id




class DLMRefreshPolicy:
    """Default shared-hook policy: uncached LLaDA denoising rollout.

    The participant-facing surface is a cache-plan interface over one fixed
    rollout, not a selector for paper-specific backend modules.
    """

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




# end of editable region – evaluation code starts below.




# ---------------------------------------------------------------------------
# Evaluation infrastructure (do not edit below this line)
# ---------------------------------------------------------------------------


def _resolve_model_dir(configured: str) -> Path:
    candidates = [Path(configured)] if configured else []
    seen = set()
    for candidate in candidates:
        if not str(candidate):
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        has_config = (resolved / 'config.json').exists()
        has_weights = (resolved / 'model.safetensors').exists() or any(resolved.glob('model-*.safetensors'))
        if has_config and has_weights:
            return resolved
    return Path(configured)


def _task_dir() -> Path | None:
    raw = os.environ.get("MLSBENCH_TASK_DIR", "")
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def _policy_initial_plan(policy: "DLMRefreshPolicy", workload_name: str = "math") -> tuple[dict, dict]:
    request_meta = {"workload": workload_name, "step_budget": "final"}
    schedule = dict(policy.block_schedule(request_meta))
    step_meta = {
        "step": 0,
        "step_in_block": 0,
        "block": 0,
        "workload": workload_name,
        "regime": "final",
        "total_steps": _clamp_int(schedule.get("num_steps"), 1, WORKLOAD_CONFIGS[workload_name]["num_steps"]),
        "prompt_len": 0,
        "gen_length": _clamp_int(schedule.get("gen_length"), 1, WORKLOAD_CONFIGS[workload_name]["gen_length"]),
        "block_length": _clamp_int(schedule.get("block_length"), 1, WORKLOAD_CONFIGS[workload_name]["block_length"]),
        "default_num_transfer_tokens": 1,
    }
    refresh_plan = dict(policy.cache_refresh_plan(
        {"layer_id": -1, "segment": "all"}, step_meta, [], {},
    ))
    return schedule, refresh_plan


def _needs_active_query_model(policy: "DLMRefreshPolicy") -> bool:
    _, refresh_plan = _policy_initial_plan(policy)
    return refresh_plan.get("kv_update") == "active_q_mask"


def _needs_tracked_window_model(policy: "DLMRefreshPolicy") -> bool:
    _, refresh_plan = _policy_initial_plan(policy)
    return refresh_plan.get("kv_update") == "tracked_window_layer_reset" or refresh_plan.get("layer_reset") == "attention_similarity"


def _load_llada_class(package_name: str, package_path: Path):
    runtime_pkg = sys.modules.get(package_name)
    if runtime_pkg is None:
        runtime_pkg = types.ModuleType(package_name)
        runtime_pkg.__path__ = [str(package_path)]
        runtime_pkg.__file__ = str(package_path / "__init__.py")
        sys.modules[package_name] = runtime_pkg
    importlib.invalidate_caches()
    modeling_llada = importlib.import_module(f"{package_name}.modeling_llada")
    llada_cls = modeling_llada.LLaDAModelLM
    llada_cls._tied_weights_keys = {
        "model.transformer.ff_out.weight": "model.transformer.wte.weight"
    }
    return llada_cls


def _install_d2_cache_import_stub() -> None:
    """Avoid importing d2Cache's CLI/frame stack when only model hooks are needed."""

    if "src.cache" in sys.modules:
        return

    class _NullD2Cache:
        def __init__(self, model_config):
            self.model_config = model_config

        def model_forward(self, x):
            return _D2ContextManager(_D2ModelForwardContext(x))

        def attention(self, layer_idx, x, attn_norm, q_proj, k_proj, v_proj, attention_mask=None, position_ids=None):
            normed = attn_norm(x)
            return _D2ContextManager(_D2AttentionContext(
                q=q_proj(normed),
                k=k_proj(normed),
                v=v_proj(normed),
                residual=x,
                attention_mask=attention_mask,
                q_position_ids=position_ids,
                kv_position_ids=position_ids,
            ))

        def ffn(self, layer_idx, x):
            return _D2ContextManager(_D2FFNContext(x))

    cache_module = types.ModuleType("src.cache")
    cache_module.dCache = _NullD2Cache
    cache_module.d2Cache = _NullD2Cache
    sys.modules["src.cache"] = cache_module


def _install_minimal_einops() -> None:
    """Provide the two rearrange patterns used by the Elastic LLaDA adapter."""

    if "einops" in sys.modules or importlib.util.find_spec("einops") is not None:
        return

    einops_module = types.ModuleType("einops")

    def rearrange(x, pattern: str, **axes):
        normalized = " ".join(pattern.split())
        if normalized == "b s three h d -> b h three s d":
            return x.permute(0, 3, 2, 1, 4).contiguous()
        if normalized == "b h s d -> b s (h d)":
            bsz, heads, seq, dim = x.shape
            return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq, heads * dim)
        raise NotImplementedError(f"Unsupported fallback einops pattern: {pattern}")

    einops_module.rearrange = rearrange
    sys.modules["einops"] = einops_module


def _load_d2_llada_class(task_dir: Path):
    source_root = task_dir / "third_party" / "official_dlm_cache_baselines" / "d2cache"
    if not (source_root / "src" / "models" / "llada" / "modeling_llada.py").exists():
        raise FileNotFoundError(f"Active-query LLaDA model class not found under {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    _install_d2_cache_import_stub()
    importlib.invalidate_caches()
    modeling_llada = importlib.import_module("src.models.llada.modeling_llada")
    modeling_llada.d2Cache = _SharedD2Cache
    llada_cls = modeling_llada.LLaDAModelLM
    if not getattr(llada_cls, "_mlsbench_d2_hf_compat", False):
        orig_tie_weights = llada_cls.tie_weights

        def compat_tie_weights(self, *args, **kwargs):
            return orig_tie_weights(self)

        llada_cls.tie_weights = compat_tie_weights
        llada_cls._mlsbench_d2_hf_compat = True
    return llada_cls


def _load_elastic_llada_class(task_dir: Path):
    class_dir = task_dir / "third_party" / "official_dlm_cache_baselines" / "elastic_cache" / "llada" / "model"
    if not (class_dir / "modeling_llada.py").exists():
        raise FileNotFoundError(f"Tracked-window LLaDA model class not found under {class_dir}")
    _install_minimal_einops()
    return _load_llada_class("_mlsbench_llada_tracked_window", class_dir)


def _patch_hf_compat(llada_cls) -> None:
    if getattr(llada_cls, "_mlsbench_hf_compat", False):
        return
    orig_init = llada_cls.__init__
    orig_tie_weights = llada_cls.tie_weights

    def compat_init(self, config, model=None, init_params=False):
        if not hasattr(config, "use_cache"):
            config.use_cache = False
        if not hasattr(config, "return_dict"):
            config.return_dict = True
        orig_init(self, config, model=model, init_params=init_params)
        if hasattr(self, "get_expanded_tied_weights_keys"):
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
        else:
            self.all_tied_weights_keys = dict(getattr(self, "_tied_weights_keys", {}))

    def compat_tie_weights(self, missing_keys=None, recompute_mapping=True):
        result = orig_tie_weights(self)
        if hasattr(self, "get_expanded_tied_weights_keys"):
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
        else:
            self.all_tied_weights_keys = dict(getattr(self, "_tied_weights_keys", {}))
        return result

    llada_cls.__init__ = compat_init
    llada_cls.tie_weights = compat_tie_weights
    llada_cls._mlsbench_hf_compat = True


def _load_model_and_tokenizer(policy: "DLMRefreshPolicy" | None = None):
    model_dir = _resolve_model_dir(_MODEL_DIR)
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from transformers import AutoTokenizer
    if not model_dir.exists():
        raise FileNotFoundError(
            f'LLaDA model not found: {_MODEL_DIR}. '
            f'Set LLADA_MODEL_DIR to a prepared GSAI-ML/LLaDA-8B-Instruct directory.'
        )
    if policy is not None and _needs_active_query_model(policy):
        task_dir = _task_dir()
        if task_dir is None:
            raise FileNotFoundError("MLSBENCH_TASK_DIR is required for active-query model compatibility files.")
        llada_cls = _load_d2_llada_class(task_dir)
    elif policy is not None and _needs_tracked_window_model(policy):
        task_dir = _task_dir()
        if task_dir is None:
            raise FileNotFoundError("MLSBENCH_TASK_DIR is required for tracked-window model compatibility files.")
        llada_cls = _load_elastic_llada_class(task_dir)
    else:
        llada_cls = _load_llada_class("_mlsbench_llada_runtime", model_dir)
    if not _needs_active_query_model(policy) if policy is not None else True:
        _patch_hf_compat(llada_cls)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = llada_cls.from_pretrained(
        str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if not hasattr(model.config, "return_dict"):
        model.config.return_dict = True
    return model, tokenizer, device


class _FeatureCacheRuntime:
    """Shared runtime adapter for feature-level cache reuse hooks."""

    def __init__(self, model):
        from dataclasses import asdict
        from dllm_cache.cache import dLLMCache, dLLMCacheConfig
        from dllm_cache.hooks import register_cache_LLaDA, logout_cache_LLaDA

        self.model = model
        self._cache_cls = dLLMCache
        self._config_cls = dLLMCacheConfig
        self._asdict = asdict
        self._register = register_cache_LLaDA
        self._logout = logout_cache_LLaDA
        self.cache = None

    def disable(self) -> None:
        self._logout(self.model, 'model.transformer.blocks')
        self.cache = None

    def enable(self, gen_interval: int, prompt_interval: int, transfer: float, prompt_len: int):
        self.disable()
        self._cache_cls.new_instance(**self._asdict(self._config_cls(
            prompt_interval_steps=max(1, prompt_interval),
            gen_interval_steps=max(1, gen_interval),
            transfer_ratio=float(transfer),
        )))
        self._register(self.model, 'model.transformer.blocks')
        self.cache = self._cache_cls()
        self.cache.reset_cache(prompt_len)
        return self.cache

    def update(self, gen_interval: int, prompt_interval: int, transfer: float) -> None:
        if self.cache is None:
            return
        self.cache.gen_interval_steps = max(1, gen_interval)
        self.cache.prompt_interval_steps = max(1, prompt_interval)
        self.cache.transfer_ratio = float(transfer)


class _D2ModelForwardContext:
    def __init__(self, x: torch.Tensor):
        self.x = x
        self.logits = None


class _D2AttentionContext:
    def __init__(self, q, k, v, residual, attention_mask=None, q_position_ids=None, kv_position_ids=None):
        self.q = q
        self.k = k
        self.v = v
        self.residual = residual
        self.attention_mask = attention_mask
        self.q_position_ids = q_position_ids
        self.kv_position_ids = kv_position_ids
        self.o = None
        self.attn_weight = None


class _D2FFNContext:
    def __init__(self, x: torch.Tensor):
        self.x = x
        self.residual = x
        self.ffn_out = None


class _D2ContextManager:
    def __init__(self, context, after=None):
        self.context = context
        self.after = after

    def __enter__(self):
        return self.context

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None and self.after is not None:
            self.after(self.context)
        return False


def _d2_convert_attention_mask(attention_mask, dtype, query_length=None, key_value_length=None):
    if attention_mask is None:
        return None
    if attention_mask.dim() == 2:
        attention_mask = attention_mask[:, None, None, :].expand(
            attention_mask.size(0),
            1,
            query_length or attention_mask.size(1),
            key_value_length or attention_mask.size(1),
        )
    elif attention_mask.dim() != 4:
        raise ValueError(f"Expected attention_mask rank 2 or 4, got {attention_mask.dim()}.")
    return (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min


def _d2_select_position_ids(position_ids=None, q_mask=None, kv_mask=None):
    q_position_ids, kv_position_ids = position_ids, position_ids
    if position_ids is not None:
        if q_mask is not None:
            q_position_ids = position_ids[q_mask].view(q_mask.size(0), -1)
        if kv_mask is not None:
            kv_position_ids = position_ids[kv_mask].view(kv_mask.size(0), -1)
    return q_position_ids, kv_position_ids


def _d2_certainty_density(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    assert sigma > 0
    batch, length = mask.shape
    device = mask.device
    float_mask = mask.float()
    padded_mask = torch.nn.functional.pad(float_mask, (length, length), "constant", 1.0)
    padded_mask[mask[:, -1] == False, 2 * length:] = 0.0
    extended_len = 3 * length
    padded_len = 2 * extended_len
    dist = torch.cat((
        torch.arange(extended_len, device=device),
        torch.arange(-extended_len, 0, device=device),
    ))
    kernel_fft = torch.fft.fft(torch.exp(-(dist**2) / (2 * sigma**2)), n=padded_len)
    weighted_sum_ext = torch.fft.ifft(
        torch.fft.fft(torch.nn.functional.pad(padded_mask, (0, extended_len)), n=padded_len) * kernel_fft,
        n=padded_len,
    ).real
    kernel_sum_ext = torch.fft.ifft(
        torch.fft.fft(torch.ones(batch, extended_len * 2, device=device), n=padded_len) * kernel_fft,
        n=padded_len,
    ).real
    return weighted_sum_ext[..., length:2 * length] / kernel_sum_ext[..., length:2 * length].clamp_min(1e-8)


def _d2_nucleus_select(scores: torch.Tensor, top_p: float, min_k: int = 1, mask: torch.Tensor | None = None):
    scores = torch.where(mask, scores, 0.0) if mask is not None else scores
    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus_mask = cumulative_probs <= top_p
    top_k_mask = torch.arange(nucleus_mask.shape[-1], device=nucleus_mask.device) < min(min_k, scores.shape[-1])
    combined_mask = nucleus_mask | top_k_mask
    if mask is not None:
        combined_mask &= torch.gather(mask, 1, sorted_indices)
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, sorted_indices, combined_mask)


def _d2_top_up_mask_(mask: torch.Tensor, target_count: int, scores: torch.Tensor):
    num_selected = mask.sum(dim=-1)
    num_to_pad = (target_count - num_selected).clamp(min=0)
    if num_to_pad.sum() == 0:
        return mask
    max_pad = int(num_to_pad.max())
    ranked_scores = torch.where(mask, -torch.inf, scores)
    _, indices = torch.topk(ranked_scores, k=max_pad, dim=-1)
    pad_indices = indices.masked_select(
        torch.arange(max_pad, device=mask.device).expand(mask.shape[0], -1) < num_to_pad.unsqueeze(-1)
    )
    row_indices = torch.repeat_interleave(torch.arange(mask.shape[0], device=mask.device), num_to_pad.long())
    mask[row_indices, pad_indices] = True
    return mask


class _SharedD2Cache:
    """Active-query cache object implementing the d2Cache hook semantics."""

    def __init__(self, model_config, rollout_p=0.1, current_k=32, sigma=10.0, inflate_w=4):
        self.model_config = model_config
        self.active_q_mask = None
        self._active_seq_mask = None
        self.key_cache = []
        self.value_cache = []
        self._conf_cache = None
        self._full_q_mask = None
        self._density_score = None
        self._global_importance = None
        self.rollout_p = float(rollout_p)
        self.current_k = int(current_k)
        self.sigma = float(sigma)
        self.inflate_w = int(inflate_w)

    @property
    def active_seq_mask(self):
        if self._active_seq_mask is None:
            raise RuntimeError("active_seq_mask is not set")
        return self._active_seq_mask

    @active_seq_mask.setter
    def active_seq_mask(self, mask):
        self._active_seq_mask = mask

    def model_forward(self, x: torch.Tensor):
        batch, seq_len, channels = x.shape
        ctx = _D2ModelForwardContext(x)

        def after(context):
            if context.logits is None:
                raise RuntimeError("D2 model_forward did not receive logits")
            if self._full_q_mask is not None:
                if self.active_q_mask is None:
                    raise RuntimeError("D2 active_q_mask missing during logits restore")
                restored = torch.zeros(
                    (batch, seq_len, context.logits.size(-1)),
                    dtype=context.logits.dtype,
                    device=context.logits.device,
                )
                context.logits = restored.masked_scatter_(self.active_q_mask.unsqueeze(-1), context.logits)

        if self._full_q_mask is not None:
            self.active_q_mask = self.top_up_mask(self._full_q_mask[self.active_seq_mask])
            ctx.x = x[self.active_q_mask].view(batch, -1, channels)
        return _D2ContextManager(ctx, after)

    def attention(self, layer_idx, x, attn_norm, q_proj, k_proj, v_proj, attention_mask=None, position_ids=None):
        residual = x
        normed = attn_norm(x)
        if normed.numel() > 0:
            q, k, v = q_proj(normed), k_proj(normed), v_proj(normed)
        else:
            q = k = v = normed[:, 0:0]
        ctx = _D2AttentionContext(q=q, k=k, v=v, residual=residual)

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(ctx.k.detach().clone())
            self.value_cache.append(ctx.v.detach().clone())
        else:
            if self.active_q_mask is None:
                raise RuntimeError("D2 active_q_mask missing for cache row update")
            if layer_idx == 0:
                active_seq_idx = torch.where(self.active_seq_mask)[0]
                m_nonzero = self.active_q_mask.nonzero(as_tuple=False)
                self._active_q_indices = (
                    active_seq_idx[m_nonzero[:, 0]],
                    m_nonzero[:, 1],
                )
            self.key_cache[layer_idx][self._active_q_indices] = ctx.k.flatten(0, 1)
            self.value_cache[layer_idx][self._active_q_indices] = ctx.v.flatten(0, 1)
            ctx.k = self.key_cache[layer_idx][self.active_seq_mask]
            ctx.v = self.value_cache[layer_idx][self.active_seq_mask]

        if layer_idx == 0:
            self._q_position_ids, self._kv_position_ids = _d2_select_position_ids(
                position_ids, self.active_q_mask
            )
            self._attention_mask = _d2_convert_attention_mask(
                attention_mask,
                dtype=ctx.k.dtype,
                query_length=ctx.q.shape[1],
                key_value_length=self.value_cache[layer_idx].shape[1],
            )

        ctx.q_position_ids = self._q_position_ids
        ctx.kv_position_ids = self._kv_position_ids
        ctx.attention_mask = self._attention_mask

        def after(context):
            if context.o is None:
                raise RuntimeError("D2 attention did not receive an output tensor")
            if context.attn_weight is None:
                raise RuntimeError("D2 requires eager attention weights for rollout")
            if layer_idx == 0:
                self._attn_rollout = torch.eye(
                    self.key_cache[layer_idx].size(1),
                    device=x.device,
                    dtype=x.dtype,
                ).expand(x.size(0), -1, -1)
            self.accumulate_attn_rollout(context.attn_weight)

        return _D2ContextManager(ctx, after)

    def ffn(self, layer_idx, x):
        ctx = _D2FFNContext(x)

        def after(context):
            if context.ffn_out is None:
                raise RuntimeError("D2 FFN did not receive an output tensor")
            if context.residual.shape != context.ffn_out.shape:
                raise RuntimeError("D2 FFN output shape mismatch")

        return _D2ContextManager(ctx, after)

    def top_up_mask(self, q_mask):
        q_mask = q_mask.clone()
        selected_per_seq = q_mask.sum(dim=-1)
        if self._density_score is None or self._global_importance is None:
            return q_mask
        _, gen_length = self._density_score.shape
        if torch.any(selected_per_seq != selected_per_seq.max()):
            combined_scores = torch.where(
                q_mask, -torch.inf, self._global_importance[self.active_seq_mask]
            )
            finite = combined_scores[torch.isfinite(combined_scores)]
            base = finite.max() if finite.numel() else torch.tensor(0.0, device=q_mask.device)
            combined_scores[:, -gen_length:] += base + self._density_score[self.active_seq_mask]
            _d2_top_up_mask_(q_mask, int(selected_per_seq.max()), combined_scores)
        return q_mask

    def accumulate_attn_rollout(self, attn_scores):
        batch, _, _, seq_len = attn_scores.shape
        device, dtype = attn_scores.device, attn_scores.dtype
        if self.active_q_mask is None:
            effective_attn = attn_scores.mean(dim=1)
        else:
            effective_attn = torch.eye(seq_len, device=device, dtype=dtype).repeat(batch, 1, 1)
            effective_attn[self.active_q_mask] = attn_scores.mean(dim=1).reshape(-1, seq_len)
        residual_attn = effective_attn + torch.eye(seq_len, device=device, dtype=dtype)
        residual_attn = residual_attn / residual_attn.sum(dim=-1, keepdim=True)
        self._attn_rollout = residual_attn @ self._attn_rollout

    def on_step_end(self, prompt_len, block_start, block_end, generated_tokens_before, generated_tokens_after, confidence, transfer_positions):
        batch, gen_length = generated_tokens_before.shape
        total_len = prompt_len + gen_length
        device = confidence.device
        active_conf = confidence
        if self._conf_cache is None:
            self._conf_cache = active_conf.detach().clone()

        remaining_mask = generated_tokens_after == MASK_ID
        if self.active_q_mask is not None:
            valid_mask = self.active_q_mask[:, prompt_len:] & (generated_tokens_before == MASK_ID)
            self._conf_cache[self.active_seq_mask][valid_mask] = active_conf[valid_mask]

        block_mask = torch.zeros((batch, gen_length), dtype=torch.bool, device=device)
        block_mask[:, block_start:block_end] = True

        block_size = block_mask.sum(dim=1, keepdim=True).clamp_min(1)
        meets_target = torch.cumsum(remaining_mask.int(), dim=1) >= self.current_k
        min_search_end = torch.argmax(meets_target.int(), dim=1, keepdim=True)
        min_search_end[~meets_target.any(dim=1, keepdim=True)] = gen_length - 1
        search_end = (((min_search_end // block_size) + 1) * block_size) - 1
        block_start_indices = torch.argmax(block_mask.int(), dim=1, keepdim=True)
        col_indices = torch.arange(gen_length, device=device)
        search_mask = (col_indices >= block_start_indices) & (col_indices <= search_end)

        scores = self._conf_cache[self.active_seq_mask] * _d2_certainty_density(~remaining_mask, self.sigma)
        finite = scores[torch.isfinite(scores)]
        bias = finite.max() if finite.numel() else torch.tensor(0.0, device=device)
        scores = scores.clone()
        scores[block_mask] += bias
        _, indices = torch.topk(
            torch.where(search_mask & remaining_mask, scores, -torch.inf),
            k=min(self.current_k, gen_length),
            dim=-1,
        )
        selected_mask = (
            torch.zeros_like(remaining_mask, dtype=torch.bool).scatter_(1, indices, True)
            & remaining_mask
        )
        response_mask = selected_mask.clone()
        if transfer_positions.numel():
            row_indices = torch.zeros_like(transfer_positions)
            response_mask[row_indices, transfer_positions] = True

        q_mask = torch.nn.functional.pad(response_mask, (prompt_len, 0), value=False)
        global_importance = self._attn_rollout.sum(dim=1)
        q_mask |= _d2_nucleus_select(global_importance, self.rollout_p, mask=~q_mask)

        if self.inflate_w > 0:
            arange_t = torch.arange(total_len, device=device).expand(batch, -1)
            masked_next = torch.where(q_mask, arange_t, total_len)
            next_selected = torch.cummin(torch.flip(masked_next, dims=[-1]), dim=-1).values
            next_selected = torch.flip(next_selected, dims=[-1])
            dist_next = next_selected - arange_t
            masked_prev = torch.where(q_mask, arange_t, -1)
            prev_selected = torch.cummax(masked_prev, dim=-1).values
            dist_prev = arange_t - prev_selected
            gap_len = dist_next + dist_prev
            q_mask |= (
                (gap_len <= self.inflate_w)
                & (prev_selected >= 0)
                & (next_selected < total_len)
            )

        if self._full_q_mask is None:
            self._full_q_mask = q_mask
            self._global_importance = global_importance
            self._density_score = scores
        else:
            self._full_q_mask[self.active_seq_mask] = q_mask
            self._global_importance[self.active_seq_mask] = global_importance
            self._density_score[self.active_seq_mask] = scores


# ---------------------------------------------------------------------------
# Benchmarks: final-task datasets and scorers
# ---------------------------------------------------------------------------

def _example_limit() -> int:
    raw = os.environ.get('DLM_MAX_EXAMPLES') or os.environ.get('DLM_MAX_PROMPTS') or '0'
    return int(raw or '0')


def _limit_examples(examples: list[dict]) -> list[dict]:
    limit = _example_limit()
    return examples[:limit] if limit > 0 else examples


def _load_benchmark_examples(workload_name: str) -> list[dict]:
    from datasets import load_dataset

    if workload_name == 'math':
        ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
        examples = []
        for i, row in enumerate(ds):
            problem = str(row.get('problem') or row.get('question') or '')
            solution = str(row.get('solution') or row.get('answer') or '')
            prompt = (
                'Solve the following competition math problem. Show concise reasoning, '
                'then put the final answer in \\\\boxed{}.\n\n'
                f'Problem: {problem}\n\nSolution:'
            )
            examples.append({'id': f'math-{i}', 'prompt': prompt, 'answer': solution})
        return _limit_examples(examples)

    if workload_name == 'humaneval':
        ds = load_dataset('openai_humaneval', split='test', trust_remote_code=True)
        examples = []
        for row in ds:
            prompt = str(row['prompt'])
            examples.append({
                'id': str(row['task_id']),
                'prompt': prompt,
                'code_prompt': prompt,
                'test': str(row['test']),
                'entry_point': str(row['entry_point']),
            })
        return _limit_examples(examples)

    if workload_name == 'lm_eval':
        ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
        examples = []
        for i, row in enumerate(ds):
            choices = row['choices']
            labels = [str(x) for x in choices['label']]
            texts = [str(x) for x in choices['text']]
            option_lines = '\n'.join(f'{label}. {text}' for label, text in zip(labels, texts))
            prompt = (
                'Answer the following multiple-choice science question. '
                'End with only the answer letter.\n\n'
                f'Question: {row["question"]}\n'
                f'Options:\n{option_lines}\n\nAnswer:'
            )
            examples.append({'id': f'arc-{i}', 'prompt': prompt, 'answer': str(row['answerKey'])})
        return _limit_examples(examples)

    raise ValueError(f'Unsupported workload: {workload_name}')


def _encode_prompt(tokenizer, prompt: str):
    if os.environ.get("DLM_APPLY_CHAT_TEMPLATE", "1") != "0" and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(prompt, return_tensors='pt')['input_ids']


def _decode_output(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def _extract_boxed(text: str) -> str:
    matches = re.findall(r'\\boxed\{([^{}]+)\}', text)
    return matches[-1].strip() if matches else ''


def _extract_math_answer(text: str) -> str:
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    matches = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?', text.replace(',', ''))
    return matches[-1] if matches else text.strip()


def _normalize_answer(text: str) -> str:
    text = _extract_math_answer(str(text))
    text = text.replace('$', '').replace('\\left', '').replace('\\right', '')
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\s+', '', text).rstrip('.;,').replace(',', '')
    return text.lower()


def _score_math(prediction: str, example: dict) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(example['answer']) else 0.0


def _extract_answer_letter(prediction: str) -> str:
    matches = re.findall(r'\b([A-J])\b', prediction.upper())
    return matches[-1] if matches else ''


def _score_lm_eval(prediction: str, example: dict) -> float:
    return 1.0 if _extract_answer_letter(prediction) == str(example['answer']).upper() else 0.0


def _refine_code_text(text: str) -> str:
    return text.replace("\t", "    ").replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        return False


def _extract_longest_valid_code(text: str) -> str:
    lines = text.splitlines()[:100]
    best_lines = 0
    best = ""
    for start in range(len(lines)):
        for end in range(start, len(lines)):
            snippet = "\n".join(lines[start:end + 1])
            if _syntax_ok(snippet):
                non_empty = sum(1 for line in lines[start:end + 1] if line.strip())
                if non_empty > best_lines:
                    best_lines = non_empty
                    best = snippet
    return best


def _definition_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _has_return(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Return) for child in ast.walk(node))


def _node_deps(node: ast.AST) -> set[str]:
    deps: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for child in ast.iter_child_nodes(current):
            if isinstance(child, ast.Name):
                deps.add(child.id)
            elif isinstance(child, ast.Attribute):
                deps.add(child.attr)
            else:
                stack.append(child)
    return deps


def _reachable_defs(entrypoint: str, deps: dict[str, set[str]]) -> set[str]:
    visited: set[str] = set()
    queue = [entrypoint]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        queue.extend(sorted(deps.get(current, set()) - visited))
    return visited


def _sanitize_humaneval(text: str, entrypoint: str) -> str:
    text = text.split("```python\n", 1)[-1].split("```")[0]
    code = _extract_longest_valid_code(_refine_code_text(text))
    if not code:
        return ""
    tree = ast.parse(code)
    imports: list[ast.AST] = []
    definitions: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.ClassDef):
            definitions[node.name] = node
        elif isinstance(node, ast.FunctionDef):
            if _has_return(node):
                definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            name = _definition_name(node)
            if name:
                definitions[name] = node
    reachable = _reachable_defs(entrypoint, {name: _node_deps(node) for name, node in definitions.items()})
    output = [ast.unparse(node) for node in imports]
    output.extend(ast.unparse(node) for name, node in definitions.items() if name in reachable)
    return "\n".join(output)


def _score_humaneval(prediction: str, example: dict) -> float:
    try:
        candidate = _sanitize_humaneval(example['code_prompt'] + "\n" + prediction, example['entry_point'])
    except (SyntaxError, ValueError, MemoryError):
        return 0.0
    program = candidate + '\n' + example['test'] + f'\ncheck({example["entry_point"]})\n'
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'candidate.py'
        path.write_text(program)
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmp,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(os.environ.get('DLM_HUMANEVAL_TIMEOUT', '5')),
            )
        except subprocess.TimeoutExpired:
            return 0.0
    return 1.0 if result.returncode == 0 else 0.0


def _score_prediction(workload_name: str, prediction: str, example: dict) -> float:
    if workload_name == 'math':
        return _score_math(prediction, example)
    if workload_name == 'humaneval':
        return _score_humaneval(prediction, example)
    if workload_name == 'lm_eval':
        return _score_lm_eval(prediction, example)
    raise ValueError(f'Unsupported workload: {workload_name}')


# ---------------------------------------------------------------------------
# Policy-driven generation (one denoising block at a time)
# ---------------------------------------------------------------------------

def _clamp_int(value, minimum: int, default: int) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _select_transfer_positions(mask_in_block, block_conf, plan: dict, default_n: int):
    if len(mask_in_block) == 0:
        return mask_in_block

    threshold = plan.get("threshold")
    if threshold is not None:
        selected = mask_in_block[block_conf[mask_in_block] >= float(threshold)]
        if len(selected) or not plan.get("force_one", True):
            return selected

    n_transfer = _clamp_int(plan.get("num_transfer_tokens"), 0, default_n)
    n_transfer = min(n_transfer, len(mask_in_block))
    if n_transfer <= 0 and plan.get("force_one", True):
        n_transfer = 1
    if n_transfer <= 0:
        return mask_in_block[:0]

    top_rel = block_conf[mask_in_block].topk(n_transfer).indices
    return mask_in_block[top_rel]


def _token_stats_from_conf(conf: torch.Tensor, prev_conf: torch.Tensor | None, step_id: int, total_steps: int):
    difficulty_tok = (1.0 - conf).clamp(0.0, 1.0)
    if prev_conf is not None:
        similarity_tok = 1.0 - (conf - prev_conf).abs().clamp(0.0, 1.0)
    else:
        similarity_tok = torch.ones(conf.shape[0], device=conf.device)
    staleness_tok = step_id / max(total_steps - 1, 1)
    return [
        {
            "importance": float(conf[i]),
            "staleness": float(staleness_tok),
            "difficulty": float(difficulty_tok[i]),
            "similarity": float(similarity_tok[i]),
        }
        for i in range(conf.shape[0])
    ]


def _pred_ids_and_conf(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    top_logits, pred_ids = logits.max(dim=-1)
    conf = torch.exp(top_logits - torch.logsumexp(logits, dim=-1))
    return pred_ids, conf


def _generate_with_active_query_hooks(
    model,
    tokenizer,
    device: str,
    prompt: str,
    policy: DLMRefreshPolicy,
    workload_name: str,
    regime_name: str,
) -> tuple[list[int], dict]:
    """Run d2Cache-style active-query and attention-rollout hooks."""

    request_meta = {"workload": workload_name, "step_budget": regime_name}
    schedule = dict(policy.block_schedule(request_meta))
    gen_length = _clamp_int(schedule.get("gen_length"), 1, WORKLOAD_CONFIGS[workload_name]["gen_length"])
    block_length = _clamp_int(schedule.get("block_length"), 1, WORKLOAD_CONFIGS[workload_name]["block_length"])
    # d2Cache's official LLaDA path transfers one token per step until the
    # current block has no mask tokens left; num_steps is not used as a quota.
    if gen_length % block_length != 0:
        raise ValueError("Active-query policies require gen_length divisible by block_length.")

    input_ids = _encode_prompt(tokenizer, prompt).to(device)
    prompt_len = int(input_ids.shape[1])
    generated = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
    x = torch.cat([input_ids, generated], dim=1)

    num_blocks = max(gen_length // block_length, 1)
    total_steps = gen_length

    probe_plan = dict(policy.attention_probe_plan({"layer_id": -1, "segment": "all"}, {"step": 0}))
    cache = _SharedD2Cache(
        model.config,
        rollout_p=float(probe_plan.get("rollout_p", 0.1)),
        current_k=_clamp_int(probe_plan.get("current_k"), 1, block_length),
        sigma=float(probe_plan.get("sigma", 10.0)),
        inflate_w=_clamp_int(probe_plan.get("inflate_w"), 0, 4),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    all_step_confs: list[float] = []
    refreshed_units = 0.0
    possible_units = 0.0
    policy_cache_state = {"active_q_mask": None}

    for block_idx in range(num_blocks):
        block_start = block_idx * block_length
        block_end = block_start + block_length

        step_i = 0
        while True:
            if (generated[:, block_start:block_end] == MASK_ID).sum() == 0:
                break

            global_step = block_idx * block_length + step_i
            default_n = 1
            step_meta = {
                "step": global_step,
                "step_in_block": step_i,
                "block": block_idx,
                "workload": workload_name,
                "regime": regime_name,
                "total_steps": total_steps,
                "prompt_len": prompt_len,
                "gen_length": gen_length,
                "block_length": block_length,
                "default_num_transfer_tokens": default_n,
            }
            mask_state = {
                "block_start": prompt_len + block_start,
                "block_end": prompt_len + block_end,
                "masked_positions": (generated[0] == MASK_ID).nonzero(as_tuple=True)[0],
                "block_masked_positions": (generated[0, block_start:block_end] == MASK_ID).nonzero(as_tuple=True)[0],
            }
            query_plan = dict(policy.query_plan(step_meta, mask_state, policy_cache_state))
            if query_plan.get("query_scope") not in {"full_sequence", "active_query_rows"}:
                raise NotImplementedError("Active-query runtime supports full_sequence and active_query_rows scopes.")
            planned_active_q = query_plan.get("query_positions")
            if planned_active_q is not None and query_plan.get("query_scope") == "active_query_rows":
                cache._full_q_mask = planned_active_q

            cache.active_seq_mask = torch.ones(1, dtype=torch.bool, device=device)
            refresh_plan = dict(policy.cache_refresh_plan(
                {"layer_id": -1, "segment": "active_query"}, step_meta, [], policy_cache_state,
            ))
            if refresh_plan.get("kv_update") != "active_q_mask":
                raise ValueError("Active-query runtime requires cache_refresh_plan(...)[kv_update] == 'active_q_mask'.")

            generated_before = generated.clone()
            x = torch.cat([input_ids, generated], dim=1)
            with torch.no_grad():
                outputs = model(
                    x,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                )
            logits0 = outputs.logits[0, prompt_len:, :].float()
            pred_ids, conf = _pred_ids_and_conf(logits0)

            transfer_mask = generated[0] == MASK_ID
            if cache.active_q_mask is not None:
                transfer_mask &= cache.active_q_mask[0, prompt_len:]
            block_mask = torch.zeros_like(transfer_mask)
            block_mask[block_start:block_end] = True
            transfer_mask &= block_mask
            mask_in_block = transfer_mask[block_start:block_end].nonzero(as_tuple=True)[0]
            if len(mask_in_block):
                all_step_confs.append(float(conf[block_start:block_end][mask_in_block].mean()))

            transfer_plan = dict(policy.token_transfer_plan(logits0, mask_state, step_meta))
            positions_rel = _select_transfer_positions(
                mask_in_block,
                conf[block_start:block_end],
                transfer_plan,
                default_n,
            )
            positions = block_start + positions_rel
            if len(positions):
                generated[0, positions] = pred_ids[positions]

            cache.on_step_end(
                prompt_len,
                block_start,
                block_end,
                generated_before,
                generated,
                conf.unsqueeze(0),
                positions,
            )
            refreshed_units += int(cache.active_q_mask.sum().item()) if cache.active_q_mask is not None else gen_length
            possible_units += prompt_len + gen_length

            policy_cache_state = policy.after_step(
                step_meta,
                logits0,
                {"active_q_count": int(cache.active_q_mask.sum().item()) if cache.active_q_mask is not None else prompt_len + gen_length},
                {"positions": positions.detach().cpu().tolist()},
                {"active_q_mask": cache._full_q_mask},
            ) or {"active_q_mask": cache._full_q_mask}
            if policy_cache_state.get("active_q_mask") is not None:
                cache._full_q_mask = policy_cache_state["active_q_mask"]
            step_i += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_mem_mb = 0.0
    elapsed_s = time.time() - t0
    refresh_ratio = min(refreshed_units / max(possible_units, 1.0), 1.0)
    return generated[0].tolist(), {
        "elapsed_s": elapsed_s,
        "reuse_ratio": max(0.0, 1.0 - refresh_ratio),
        "refresh_ratio": refresh_ratio,
        "peak_memory_mb": peak_mem_mb,
        "gen_tokens": gen_length,
        "mean_transfer_conf": sum(all_step_confs) / max(len(all_step_confs), 1),
    }


def _generate_with_tracked_window_hooks(
    model,
    tokenizer,
    device: str,
    prompt: str,
    policy: DLMRefreshPolicy,
    workload_name: str,
    regime_name: str,
) -> tuple[list[int], dict]:
    """Run Elastic-Cache-style tracked-token masked-window hooks."""

    request_meta = {"workload": workload_name, "step_budget": regime_name}
    schedule = dict(policy.block_schedule(request_meta))
    gen_length = _clamp_int(schedule.get("gen_length"), 1, WORKLOAD_CONFIGS[workload_name]["gen_length"])
    window_length = _clamp_int(schedule.get("window_length"), 1, 16)
    window_length = min(window_length, gen_length)

    input_ids = _encode_prompt(tokenizer, prompt).to(device)
    prompt_len = int(input_ids.shape[1])
    x = torch.full((1, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    for block in model.model.transformer.blocks:
        block.x_cache = None
        block.q_cache = None
        block.k_cache = None
        block.v_cache = None
        block.track_token = None

    probe_plan = dict(policy.attention_probe_plan({"layer_id": -1, "segment": "all"}, {"step": 0}))
    gamma = float(probe_plan.get("gamma", 0.9))
    track_num = _clamp_int(probe_plan.get("track_num"), 1, 1)
    eos_id = int(getattr(tokenizer, "eos_token_id", 126081) or 126081)

    query_position = torch.arange(prompt_len + gen_length, device=device)
    track_position = query_position[:0].clone()
    new_decoded_position = query_position[:prompt_len].clone()
    masked_position = query_position[prompt_len:].clone()
    decoded_eos = False
    nfe = 0
    computed_layers = 0.0
    possible_layers = 0.0
    num_layers = len(model.model.transformer.blocks)
    policy_cache_state = {"track_positions": track_position}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    while masked_position.numel() > 0:
        query_masked_position = masked_position[:window_length]
        step_meta = {
            "step": nfe,
            "step_in_block": nfe,
            "block": 0,
            "workload": workload_name,
            "regime": regime_name,
            "total_steps": gen_length,
            "prompt_len": prompt_len,
            "gen_length": gen_length,
            "block_length": window_length,
            "default_num_transfer_tokens": 1,
        }
        mask_state = {
            "block_start": int(query_masked_position[0].item()) if query_masked_position.numel() else prompt_len,
            "block_end": int(query_masked_position[-1].item()) + 1 if query_masked_position.numel() else prompt_len,
            "masked_positions": masked_position - prompt_len,
            "block_masked_positions": query_masked_position - prompt_len,
        }
        query_plan = dict(policy.query_plan(step_meta, mask_state, policy_cache_state))
        if query_plan.get("query_scope") not in {"full_sequence", "tracked_window"}:
            raise NotImplementedError("Tracked-window runtime supports full_sequence and tracked_window scopes.")
        planned_track = query_plan.get("track_positions")
        if planned_track is not None:
            track_position = torch.as_tensor(planned_track, dtype=torch.long, device=device)
        planned_window = query_plan.get("masked_window")
        if planned_window is not None and query_plan.get("query_scope") == "tracked_window":
            window_start, window_end = int(planned_window[0]), int(planned_window[1])
            in_window = (masked_position >= window_start) & (masked_position < window_end)
            planned_masked = masked_position[in_window]
            if planned_masked.numel():
                query_masked_position = planned_masked[:window_length]

        if nfe == 0:
            x_query = x
            start_reset = -1
            query_position = torch.arange(prompt_len + gen_length, device=device)
        else:
            query_position = torch.cat([track_position, new_decoded_position, query_masked_position], dim=0)
            x_query = x[:, query_position]
            start_reset = num_layers

        refresh_plan = dict(policy.cache_refresh_plan(
            {"layer_id": -1, "segment": "tracked_window"}, step_meta, [], policy_cache_state,
        ))
        if refresh_plan.get("kv_update") != "tracked_window_layer_reset":
            raise ValueError("Tracked-window runtime requires kv_update='tracked_window_layer_reset'.")

        positions = [query_position, track_position, query_masked_position, masked_position]
        lengths = [x.shape[1], start_reset, gamma, track_num]
        with torch.no_grad():
            output = model(x_query, use_cache=True, lengths=lengths, positions=positions)
        logits = output.logits
        if logits.shape[1] == x.shape[1]:
            logits = logits[:, query_masked_position, :]
        else:
            logits = logits[:, -query_masked_position.shape[0]:, :]

        tracked = [block.track_token for block in model.model.transformer.blocks if block.track_token is not None]
        if tracked:
            track_position = torch.cat(tracked, dim=0).unique(sorted=False)
        policy_cache_state["track_positions"] = track_position

        pred_ids, conf = _pred_ids_and_conf(logits[0])
        transfer_plan = dict(policy.token_transfer_plan(logits[0].float(), mask_state, step_meta))
        threshold = transfer_plan.get("threshold")
        if threshold is not None:
            keep = conf >= min(float(threshold), float(conf.max()))
            if not keep.any() and transfer_plan.get("force_one", True):
                keep[conf.argmax()] = True
        else:
            n_transfer = _clamp_int(transfer_plan.get("num_transfer_tokens"), 1, 1)
            keep = torch.zeros_like(conf, dtype=torch.bool)
            keep[conf.topk(min(n_transfer, conf.numel())).indices] = True

        new_decoded_position = query_masked_position[keep]
        if new_decoded_position.numel():
            x[:, new_decoded_position] = pred_ids[keep]
        masked_position = masked_position[~torch.isin(masked_position, new_decoded_position)]

        if not decoded_eos and new_decoded_position.numel():
            eos_mask = pred_ids[keep].eq(eos_id)
            if eos_mask.any():
                eos_pos = int(new_decoded_position[eos_mask].min().item())
                decoded_eos = True
                masked_position = masked_position[masked_position <= eos_pos]

        nfe += 1
        computed_layers += num_layers - lengths[1]
        possible_layers += num_layers
        policy_cache_state = policy.after_step(
            step_meta,
            logits[0].float(),
            {"track_positions": track_position.detach().cpu().tolist(), "start_reset": lengths[1]},
            {"positions": (new_decoded_position - prompt_len).detach().cpu().tolist()},
            policy_cache_state,
        ) or policy_cache_state
        if policy_cache_state.get("track_positions") is not None:
            track_position = torch.as_tensor(policy_cache_state["track_positions"], dtype=torch.long, device=device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_mem_mb = 0.0
    elapsed_s = time.time() - t0
    refresh_ratio = min(computed_layers / max(possible_layers, 1.0), 1.0)
    return x[0, prompt_len:].tolist(), {
        "elapsed_s": elapsed_s,
        "reuse_ratio": max(0.0, 1.0 - refresh_ratio),
        "refresh_ratio": refresh_ratio,
        "peak_memory_mb": peak_mem_mb,
        "gen_tokens": gen_length,
        "nfe": nfe,
    }


def _generate_with_shared_hooks(
    model,
    tokenizer,
    device: str,
    prompt: str,
    policy: DLMRefreshPolicy,
    workload_name: str,
    regime_name: str,
) -> tuple[list[int], dict]:
    request_meta = {"workload": workload_name, "step_budget": regime_name}
    schedule = dict(policy.block_schedule(request_meta))
    gen_length = _clamp_int(schedule.get("gen_length"), 1, WORKLOAD_CONFIGS[workload_name]["gen_length"])
    block_length = _clamp_int(schedule.get("block_length"), 1, WORKLOAD_CONFIGS[workload_name]["block_length"])
    num_steps = _clamp_int(schedule.get("num_steps"), 1, WORKLOAD_CONFIGS[workload_name]["num_steps"])

    input_ids = _encode_prompt(tokenizer, prompt).to(device)
    prompt_len = int(input_ids.shape[1])
    x = torch.full((1, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    num_blocks = max(gen_length // block_length, 1)
    steps_per_block = max(num_steps // num_blocks, 1)
    total_steps = num_blocks * steps_per_block

    init_step_meta = {
        "step": 0,
        "block": 0,
        "workload": workload_name,
        "regime": regime_name,
        "total_steps": total_steps,
        "prompt_len": prompt_len,
        "gen_length": gen_length,
        "block_length": block_length,
        "default_num_transfer_tokens": 1,
    }
    init_plan = dict(policy.cache_refresh_plan(
        {"layer_id": -1, "segment": "all"}, init_step_meta, [], {},
    ))
    if init_plan.get("kv_update") == "active_q_mask":
        return _generate_with_active_query_hooks(
            model, tokenizer, device, prompt, policy, workload_name, regime_name,
        )
    if init_plan.get("kv_update") == "tracked_window_layer_reset":
        return _generate_with_tracked_window_hooks(
            model, tokenizer, device, prompt, policy, workload_name, regime_name,
        )
    use_feature_cache = bool(init_plan.get("use_feature_cache", False))
    feature_runtime = _FeatureCacheRuntime(model)
    if use_feature_cache:
        init_transfer_ratio = float(init_plan.get("transfer_ratio", 0.0))
        if init_plan.get("row_selector") != "lowest_value_feature_similarity":
            init_transfer_ratio = 0.0
        cache = feature_runtime.enable(
            _clamp_int(init_plan.get("gen_refresh_interval"), 1, 1),
            _clamp_int(init_plan.get("prompt_refresh_interval"), 1, 1),
            init_transfer_ratio,
            prompt_len,
        )
    else:
        feature_runtime.disable()
        cache = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    all_step_confs: list[float] = []
    prev_conf: torch.Tensor | None = None
    refreshed_units = 0.0
    possible_units = 0.0
    cache_state = {"use_feature_cache": use_feature_cache}

    for block_idx in range(num_blocks):
        start = prompt_len + block_idx * block_length
        end = prompt_len + (block_idx + 1) * block_length
        block_mask_pos = (x[0, start:end] == MASK_ID).nonzero(as_tuple=True)[0]
        n_mask = len(block_mask_pos)
        base_n = n_mask // steps_per_block
        rem = n_mask % steps_per_block

        for step_i in range(steps_per_block):
            global_step = block_idx * steps_per_block + step_i
            step_meta = {
                "step": global_step,
                "step_in_block": step_i,
                "block": block_idx,
                "workload": workload_name,
                "regime": regime_name,
                "total_steps": total_steps,
                "prompt_len": prompt_len,
                "gen_length": gen_length,
                "block_length": block_length,
                "default_num_transfer_tokens": base_n + (1 if step_i < rem else 0),
            }
            mask_state = {
                "block_start": start,
                "block_end": end,
                "masked_positions": (x[0, prompt_len:] == MASK_ID).nonzero(as_tuple=True)[0],
                "block_masked_positions": (x[0, start:end] == MASK_ID).nonzero(as_tuple=True)[0],
            }
            query_plan = dict(policy.query_plan(step_meta, mask_state, cache_state))
            if query_plan.get("query_scope", "full_sequence") != "full_sequence":
                raise NotImplementedError(
                    "This shared-hook stage supports full-sequence query plans only. "
                    "Add query slicing before enabling this baseline."
                )

            with torch.no_grad():
                logits = model(x).logits[:, prompt_len:, :]
            logits0 = logits[0].float()
            pred_ids, conf = _pred_ids_and_conf(logits0)

            token_stats = _token_stats_from_conf(conf, prev_conf, global_step, total_steps)
            prev_conf = conf.detach()

            refresh_plan = dict(policy.cache_refresh_plan(
                {"layer_id": -1, "segment": "all"}, step_meta, token_stats, cache_state,
            ))
            if bool(refresh_plan.get("use_feature_cache", False)) != use_feature_cache:
                raise ValueError("A policy may not toggle feature-cache support mid-example.")
            gen_interval = _clamp_int(refresh_plan.get("gen_refresh_interval"), 1, 1)
            prompt_interval = _clamp_int(refresh_plan.get("prompt_refresh_interval"), 1, 1)
            transfer_ratio = float(refresh_plan.get("transfer_ratio", 0.0))
            if refresh_plan.get("row_selector") != "lowest_value_feature_similarity":
                transfer_ratio = 0.0
            feature_runtime.update(gen_interval, prompt_interval, transfer_ratio)

            mask_in_block = (x[0, start:end] == MASK_ID).nonzero(as_tuple=True)[0]
            if len(mask_in_block):
                all_step_confs.append(float(conf[start - prompt_len : end - prompt_len][mask_in_block].mean()))

            n_unmask = base_n + (1 if step_i < rem else 0)
            transfer_plan = dict(policy.token_transfer_plan(logits0, mask_state, step_meta))
            if n_unmask > 0 and len(mask_in_block):
                block_conf = conf[start - prompt_len : end - prompt_len]
                positions = _select_transfer_positions(mask_in_block, block_conf, transfer_plan, n_unmask)
                x[0, start + positions] = pred_ids[start - prompt_len + positions]
            else:
                positions = mask_in_block[:0]

            if cache is None:
                refreshed_units += gen_length
            else:
                refreshed_units += gen_length if (step_i % gen_interval) == 0 else max(1.0, gen_length * transfer_ratio)
            possible_units += gen_length
            cache_state = policy.after_step(
                step_meta,
                logits0,
                {},
                {"positions": positions.detach().cpu().tolist()},
                cache_state,
            ) or cache_state

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_mem_mb = 0.0
    elapsed_s = time.time() - t0
    reuse_ratio = 1.0 - min(refreshed_units / max(possible_units, 1.0), 1.0)
    return x[0, prompt_len:].tolist(), {
        "elapsed_s": elapsed_s,
        "reuse_ratio": max(0.0, reuse_ratio),
        "refresh_ratio": min(refreshed_units / max(possible_units, 1.0), 1.0),
        "peak_memory_mb": peak_mem_mb,
        "gen_tokens": gen_length,
    }


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation(workload_name: str, regime_name: str):
    policy = DLMRefreshPolicy()
    print(f'[INFO] Loading LLaDA model from {_resolve_model_dir(_MODEL_DIR)}')
    print(f'[INFO] Using shared DLM cache hook policy={getattr(policy, "policy_name", "custom")}')
    model, tokenizer, device = _load_model_and_tokenizer(policy)

    print(f'[INFO] Loading final-score benchmark examples for workload={workload_name}')
    examples = _load_benchmark_examples(workload_name)
    print(f'[INFO] {len(examples)} examples loaded')

    final_scores: list[float] = []
    reuse_values: list[float] = []
    elapsed_values: list[float] = []
    peak_mem_mb = 0.0

    print(f'[INFO] Evaluating policy on {len(examples)} examples...')
    for i, example in enumerate(examples):
        out, stats = _generate_with_shared_hooks(
            model, tokenizer, device, example['prompt'], policy, workload_name, regime_name,
        )
        prediction = _decode_output(tokenizer, out)
        score = _score_prediction(workload_name, prediction, example)
        final_scores.append(score)
        reuse_values.append(float(stats["reuse_ratio"]))
        elapsed_values.append(float(stats["elapsed_s"]))
        peak_mem_mb = max(peak_mem_mb, float(stats["peak_memory_mb"]))
        print(
            f'  [{i+1}/{len(examples)}] id={example["id"]} '
            f'final_score={score:.4f} reuse={stats["reuse_ratio"]:.4f}'
        )

    final_score = 100.0 * (sum(final_scores) / max(len(final_scores), 1))
    reuse_ratio = sum(reuse_values) / max(len(reuse_values), 1)
    refresh_ratio = 1.0 - reuse_ratio
    total_s = sum(elapsed_values)
    tokens_per_s = len(examples) * WORKLOAD_CONFIGS[workload_name]["gen_length"] / max(total_s, 1e-6)
    print(
        f'TEST_METRICS: '
        f'final_score={final_score:.4f} '
        f'reuse_ratio={reuse_ratio:.4f} '
        f'refresh_ratio={refresh_ratio:.4f} '
        f'tokens_per_s={tokens_per_s:.2f} '
        f'peak_memory_mb={peak_mem_mb:.1f} '
        f'n_examples={len(examples)} '
        f'eval_mode=real_rollout '
        f'policy={getattr(policy, "policy_name", "custom")} '
        f'workload={workload_name} '
        f'regime={regime_name}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', choices=sorted(WORKLOAD_CONFIGS), required=True)
    parser.add_argument('--regime', choices=sorted(REGIMES), required=True)
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '42')))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    run_evaluation(args.workload, args.regime)
