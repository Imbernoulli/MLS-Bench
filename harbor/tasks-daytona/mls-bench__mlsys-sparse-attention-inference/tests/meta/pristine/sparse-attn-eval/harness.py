"""Shared harness for the LLM-only long-context inference task.

Functions:
  - patch_qwen(model, sparse_factory): replace forward() of every
      Qwen2 attention layer (Qwen2Attention / Qwen2SdpaAttention /
      Qwen2FlashAttention2) with a wrapper that routes Q/K/V (after
      RoPE + GQA replication) through SparseAttention.
  - density tracking: every wrapped module reports ``last_density``;
      ``enforce_budget()`` aggregates and aborts if the mean exceeds the
      budget (with a small slack), except for the dense oracle.

Qwen2.5-1.5B-Instruct has native 32K context length, so NO RoPE rescaling
is needed for the 8K target — ``apply_ntk_rope_scaling`` is kept as a
backwards-compatible no-op so any caller / test referring to the old name
still resolves.
"""

import contextlib
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


_DENSITY_RECORDS = []


def _cuda_device_summary():
    if not torch.cuda.is_available():
        return "cpu"
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return f"{props.name} cc={props.major}.{props.minor} cuda={torch.version.cuda}"


def reset_density():
    _DENSITY_RECORDS.clear()


def get_density_stats():
    if not _DENSITY_RECORDS:
        return {"mean": 0.0, "max": 0.0, "count": 0}
    arr = torch.tensor(_DENSITY_RECORDS)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "count": len(_DENSITY_RECORDS),
    }


def _record_density(d):
    if d is None:
        raise RuntimeError("SparseAttention.last_density was not set")
    try:
        d = float(d)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"SparseAttention.last_density must be a finite float, got {d!r}"
        ) from exc
    if not math.isfinite(d) or not (0.0 <= d <= 1.0):
        raise RuntimeError(
            f"SparseAttention.last_density must be in [0, 1], got {d!r}"
        )
    _DENSITY_RECORDS.append(d)


@contextlib.contextmanager
def density_window():
    reset_density()
    try:
        yield
    finally:
        pass


# ── HF transformers Qwen2 attention wrapper ───────────────────────────────────

def patch_qwen(model, sparse_module):
    """Replace forward() in every Qwen2-family attention layer.

    Qwen2.5-1.5B-Instruct uses Grouped-Query Attention with 12 query heads
    and 2 KV heads — so K/V must be replicated 6× before being fed to
    SparseAttention (which expects Q and K/V to share the head dimension).

    The replacement forward:
      1. Projects hidden_states -> q/k/v via the layer's own q_proj, k_proj,
         v_proj (separate, not fused — unlike GPTNeoX's query_key_value).
      2. Reshapes to (B, n_q_heads, N, D) / (B, n_kv_heads, N, D).
      3. Applies RoPE using the externally-supplied position_embeddings or
         the layer's own rotary_emb (transformers >=4.46 prefers the former).
      4. Replicates K/V to match Q via ``repeat_kv``.
      5. Routes (q, k, v) through ``self._sparse_attn`` with is_causal=True
         and scale = 1/sqrt(head_dim).
      6. Reshapes (B, n_q_heads, N, D) -> (B, N, hidden) and applies o_proj.
      7. Returns ``(attn_output, None, past_key_value)`` matching Qwen2's
         expected return tuple.

    We do not support KV cache / generate-with-cache — every forward processes
    the full prefix in one shot. The runner sets ``use_cache=False`` for both
    the model load and ``model.generate(...)``.
    """
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        apply_rotary_pos_emb,
        repeat_kv,
    )

    def is_patchable_qwen_attention(module):
        class_name = module.__class__.__name__
        if isinstance(module, Qwen2Attention):
            return True
        if class_name in {"Qwen2Attention", "Qwen2SdpaAttention", "Qwen2FlashAttention2"}:
            return all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
        return (
            class_name.startswith("Qwen2")
            and class_name.endswith("Attention")
            and all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
        )

    def make_forward(sa):
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()
            if q_len <= 0 or int(getattr(self, "head_dim", 0)) <= 0:
                print(
                    "ATTN_DIAGNOSTIC "
                    f"phase=invalid layer={getattr(self, 'layer_idx', 'unknown')} "
                    f"q_len={q_len} head_dim={getattr(self, 'head_dim', None)}",
                    flush=True,
                )
                raise RuntimeError("Invalid Qwen attention shape before sparse forward")
            if int(getattr(self, "layer_idx", -1)) == 0:
                print(
                    "ATTN_DIAGNOSTIC "
                    f"phase=forward_start q_len={q_len} dtype={hidden_states.dtype} "
                    f"device={hidden_states.device} cuda={_cuda_device_summary()}",
                    flush=True,
                )

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            # RoPE
            if position_embeddings is None:
                # transformers <=4.46 fallback path
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

            # GQA: replicate K/V so they share head dim with Q
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            target_dtype = value_states.dtype
            if query_states.dtype != target_dtype:
                query_states = query_states.to(target_dtype)
            if key_states.dtype != target_dtype:
                key_states = key_states.to(target_dtype)

            scale = 1.0 / math.sqrt(float(self.head_dim))
            attn_output = sa(
                query_states, key_states, value_states,
                is_causal=True, scale=scale,
            )
            _record_density(getattr(sa, "last_density", None))

            # (B, H, N, D) -> (B, N, H*D)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value
        return forward

    n_patched = 0
    for module in model.modules():
        if is_patchable_qwen_attention(module):
            sa = sparse_module(
                head_dim=module.head_dim,
                num_heads=module.num_heads,
            )
            param = next(module.parameters())
            sa = sa.to(param.device, dtype=param.dtype)
            module.forward = make_forward(sa).__get__(module, type(module))
            module._sparse_attn = sa  # keep alive
            n_patched += 1
    return n_patched


def apply_ntk_rope_scaling(model, scale_factor):
    """No-op for Qwen2.5 (native 32K context — 8K is well within range).

    Kept as a backwards-compatible stub so existing callers / tests don't
    need to be rewritten. Returns 0 to indicate no rotary modules patched.
    """
    if scale_factor <= 1.0:
        return 0
    print(f"[harness] apply_ntk_rope_scaling x{scale_factor:.2f}: no-op "
          f"(Qwen2.5 native context >= target)", flush=True)
    return 0


# ── Top-level monkey-patch dispatcher ─────────────────────────────────────────

def patch_model(model, modality, sparse_factory):
    if modality == "llm":
        n = patch_qwen(model, sparse_factory)
    else:
        raise ValueError(f"unknown modality: {modality}")
    print(f"[harness] patched {n} attention layers ({modality})", flush=True)
    if n <= 0:
        raise RuntimeError("no Qwen2 attention layers were patched; refusing to run native attention")
    return n


def enforce_budget(modality_label, budget, allow_dense=False):
    stats = get_density_stats()
    print(f"DENSITY_STATS modality={modality_label} mean={stats['mean']:.4f} "
          f"max={stats['max']:.4f} count={stats['count']}", flush=True)
    if stats["count"] == 0:
        raise RuntimeError(
            "density budget could not be enforced: no SparseAttention "
            "density records were produced"
        )
    if allow_dense:
        return stats
    slack = 0.02
    if stats["mean"] > budget + slack:
        raise RuntimeError(
            f"density budget violated: mean={stats['mean']:.4f} > "
            f"budget={budget:.4f} (+{slack} slack)"
        )
    return stats
