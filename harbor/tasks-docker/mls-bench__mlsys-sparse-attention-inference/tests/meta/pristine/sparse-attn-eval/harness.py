"""Shared harness for the LLM-only long-context inference task.

Functions:
  - patch_qwen(model, sparse_factory): replace forward() of every
      Qwen2 attention layer (Qwen2Attention / Qwen2SdpaAttention /
      Qwen2FlashAttention2) with a wrapper that routes Q/K/V (after
      RoPE + GQA replication) through SparseAttention.
  - density tracking: after every forward the harness reads the attention
      mask the module reports in ``last_mask`` and computes the density
      itself (causal-adjusted); ``None`` means dense. The module's own
      ``last_density`` is not trusted. On random query rows (always
      including the last one) the harness recomputes the masked attention
      from its own copies of q/k/v and aborts if the module's output does
      not match, so a module cannot report a sparser mask than it used.
      ``enforce_budget()`` aggregates and aborts if the mean exceeds the
      budget (with a small slack), except for the dense oracle.

Qwen2.5-1.5B-Instruct has native 32K context length, so NO RoPE rescaling
is needed for the 8K target — ``apply_ntk_rope_scaling`` is kept as a
backwards-compatible no-op so any caller / test referring to the old name
still resolves.
"""

import contextlib
import math
import secrets
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
    _VERIFY_STATS["max_err"] = 0.0
    _VERIFY_STATS["max_leak"] = 0.0
    _VERIFY_STATS["calls"] = 0
    _VERIFY_STATS["rows"] = 0
    _VERIFY_STATS["skipped"] = 0


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
    d = float(d)
    if not math.isfinite(d) or not (0.0 <= d <= 1.0):
        raise RuntimeError(f"computed attention density must be in [0, 1], got {d!r}")
    _DENSITY_RECORDS.append(d)


# ── trusted density + output verification ─────────────────────────────────────
#
# The density is derived here, from the mask the module reports in
# ``last_mask``, and the module's output is checked on randomly sampled query
# rows (plus the last row, which drives the next-token logits). The rows are
# drawn before the module runs, from a generator seeded from the OS, and the
# check uses the harness's own copies of q/k/v.
#
# On each sampled row the harness computes, in float64 from its own copies of
# q/k/v, the attention restricted to the mask, ref_m, and the full causal
# attention, ref_d, and checks the module's output o against them (norms are
# relative to the head's RMS output norm):
#   1. |o - ref_m| <= _VERIFY_ERR_TOL: the output is attention over the mask.
#      Without this bound only the component of o - ref_m along ref_d - ref_m
#      was checked, head by head, so a module could report a diagonal mask and
#      return dense attention plus a correction that cancels that component
#      but lies in o_proj's near-null space (error ~22, dense quality at
#      density 0.0005). Clamping such a correction to an error of 1 leaves
#      the model at quality 0.
#   2. No leak toward full attention, looking for attention outside the mask
#      rather than for bit-exactness:
#          leak = <o - ref_m, ref_d - ref_m> / |ref_d - ref_m|
#      A row fails when the mask matters for it (|ref_d - ref_m| >
#      _VERIFY_TOL), leak > _VERIFY_TOL, and o covers more than _VERIFY_FRAC
#      of the way. Dense compute behind a 0.2-density mask fails on ~40% of
#      sampled rows.
# Rows with a causal logit above _VERIFY_LOGIT_MAX in magnitude are not
# checked. There fp16 spaces logits >= 1 apart, and TF32 (used by SDPA's fp32
# math under the global TF32 setting) rounds q and k as coarsely, so honest
# low-precision attention can move the output by up to 0.5 in any direction,
# including toward ref_d. In Qwen2.5-1.5B only layer 0 has such logits (up to
# ~2e4; about a fifth of its rows, < 1% of all checked rows). Measured at 4K
# context on the checked rows: fp16 SDPA max error 0.04 (leak 0.03), fp32
# baselines 0.0025. Logits from a plain fp16 matmul reach a leak of 0.09 in
# layer 0 (|logit| up to 1024), so hand-written attention should compute the
# logits in fp32.

_VERIFY_ROWS = 32          # random query rows checked per attention call
_VERIFY_TOL = 0.1          # leak toward full attention, relative to head RMS norm
_VERIFY_FRAC = 0.25        # ... and as a fraction of |ref_d - ref_m|
_VERIFY_ERR_TOL = 1.0      # total error vs attention restricted to the mask
_VERIFY_LOGIT_MAX = 1024.0 # rows with a larger |logit| are not checked
_MASK_ROW_CHUNK = 1024     # query rows per chunk when counting mask entries
_VERIFY_GEN = torch.Generator(device="cpu")
_VERIFY_GEN.manual_seed(secrets.randbits(63))
_VERIFY_STATS = {"max_err": 0.0, "max_leak": 0.0, "calls": 0, "rows": 0, "skipped": 0}


def _sample_verify_rows(n, device):
    k = min(_VERIFY_ROWS, n)
    rows = torch.randperm(n, generator=_VERIFY_GEN)[:k]
    rows = torch.unique(torch.cat([rows, torch.tensor([n - 1])]))
    return rows.to(device)


def _normalize_mask(mask, b, h, n, device):
    """Validate ``last_mask`` and view it as (b|1, h|1, n, n) bool."""
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise RuntimeError(
            "SparseAttention.last_mask must be a torch.bool tensor (True = "
            f"attended) or None for dense, got {type(mask).__name__}"
            + (f" dtype={mask.dtype}" if isinstance(mask, torch.Tensor) else "")
        )
    if mask.dim() < 2 or mask.dim() > 4 or tuple(mask.shape[-2:]) != (n, n):
        raise RuntimeError(
            f"SparseAttention.last_mask must have shape broadcastable to "
            f"(B, H, N, N) = ({b}, {h}, {n}, {n}), got {tuple(mask.shape)}"
        )
    m = mask.reshape((1,) * (4 - mask.dim()) + tuple(mask.shape))
    if m.shape[0] not in (1, b) or m.shape[1] not in (1, h):
        raise RuntimeError(
            f"SparseAttention.last_mask must have shape broadcastable to "
            f"(B, H, N, N) = ({b}, {h}, {n}, {n}), got {tuple(mask.shape)}"
        )
    return m.to(device)


def _mask_density(m, b, h, n, device):
    """Fraction of causal (q, k) pairs the mask attends, over all (B, H)."""
    if m is None:
        return 1.0
    idx = torch.arange(n, device=device)
    count = torch.zeros((), dtype=torch.int64, device=device)
    for r0 in range(0, n, _MASK_ROW_CHUNK):
        r1 = min(n, r0 + _MASK_ROW_CHUNK)
        tri = idx[None, :] <= idx[r0:r1, None]
        count += (m[:, :, r0:r1, :] & tri).sum()
    factor = (b // m.shape[0]) * (h // m.shape[1])
    denom = b * h * n * (n + 1) / 2.0
    return float(count.item()) * factor / max(denom, 1.0)


@torch.no_grad()
def _verify_output(out, m, q_rows, k, v, rows, scale, shape):
    b, h, n, d = shape
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != (b, h, n, d):
        raise RuntimeError(
            f"SparseAttention.forward must return a (B, H, N, D) = "
            f"({b}, {h}, {n}, {d}) tensor, got "
            f"{tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__}"
        )
    o = out[:, :, rows, :].double()
    if not torch.isfinite(o).all():
        raise RuntimeError("SparseAttention output has NaN/inf on verified rows")
    # float64: exact regardless of the global TF32 matmul setting (Qwen's
    # layer-0 logits reach ~1e4, where TF32 rounding alone is visible).
    s = torch.matmul(q_rows.double(), k.double().transpose(-2, -1)) * scale
    cols = torch.arange(n, device=s.device)
    causal = (cols[None, :] <= rows[:, None]).view(1, 1, rows.numel(), n)

    def _ref(allowed):
        p = torch.softmax(s.masked_fill(~allowed, float("-inf")), dim=-1)
        return torch.matmul(torch.nan_to_num(p, nan=0.0), v.double())

    ref_d = _ref(causal)
    ref_m = ref_d if m is None else _ref(causal & m[:, :, rows, :])
    head_rms = ref_m.norm(dim=-1).pow(2).mean(dim=-1, keepdim=True).sqrt()
    head_rms = head_rms.clamp_min(1e-6)
    # rows whose logits fp16 cannot resolve are not checked (see above)
    checked = s.abs().masked_fill(~causal, 0.0).amax(dim=-1) <= _VERIFY_LOGIT_MAX
    err_rows = ((o - ref_m).norm(dim=-1) / head_rms).masked_fill(~checked, 0.0)
    err = float(err_rows.max().item())
    _VERIFY_STATS["calls"] += 1
    _VERIFY_STATS["rows"] += checked.numel()
    _VERIFY_STATS["skipped"] += int((~checked).sum().item())
    _VERIFY_STATS["max_err"] = max(_VERIFY_STATS["max_err"], err)
    if err > _VERIFY_ERR_TOL:
        raise RuntimeError(
            f"SparseAttention output is not attention restricted to the "
            f"reported last_mask: on a sampled query row it differs from it "
            f"by {err:.4f} (head-RMS units, limit {_VERIFY_ERR_TOL}). The "
            f"output must be attention over the (q, k) pairs in last_mask."
        )
    if m is None:
        return  # reported dense: density 1.0, nothing outside the mask
    diff = ref_d - ref_m
    gap = diff.norm(dim=-1)                                        # (B, H, R)
    leak_abs = (o - ref_m).mul(diff).sum(dim=-1) / gap.clamp_min(1e-12)
    leak = leak_abs / head_rms                                     # (B, H, R)
    frac = leak_abs / gap.clamp_min(1e-12)
    matters = checked & (gap / head_rms > _VERIFY_TOL)
    bad = matters & (leak > _VERIFY_TOL) & (frac > _VERIFY_FRAC)
    worst = float(leak.masked_fill(~matters, 0.0).max().item())
    _VERIFY_STATS["max_leak"] = max(_VERIFY_STATS["max_leak"], worst)
    if bool(bad.any().item()):
        i = int(leak.masked_fill(~bad, float("-inf")).flatten().argmax())
        raise RuntimeError(
            f"SparseAttention attended (q, k) pairs outside the reported "
            f"last_mask: on a sampled query row the output moved "
            f"{float(leak.flatten()[i]):.4f} (head-RMS units, limit {_VERIFY_TOL}) "
            f"= {100 * float(frac.flatten()[i]):.0f}% of the way from attention "
            f"restricted to last_mask toward full causal attention (limit "
            f"{100 * _VERIFY_FRAC:.0f}%). The density is computed from "
            f"last_mask, so the mask must include every pair the module attends. "
            f"(Logits from a plain fp16/bf16 matmul are too coarse for this "
            f"check; compute them in fp32, as fused SDPA does.)"
        )


def _verified_sparse_attention(sa, q, k, v, scale):
    """Run the module, check its output against its mask, record density."""
    b, h, n, d = q.shape
    rows = _sample_verify_rows(n, q.device)
    q_rows = q[:, :, rows, :].clone()
    k_ref = k.clone()
    v_ref = v.clone()
    sa.last_mask = None
    out = sa(q, k, v, is_causal=True, scale=scale)
    m = _normalize_mask(getattr(sa, "last_mask", None), b, h, n, q.device)
    sa.last_mask = None
    _verify_output(out, m, q_rows, k_ref, v_ref, rows, scale, (b, h, n, d))
    _record_density(_mask_density(m, b, h, n, q.device))
    return out


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
            attn_output = _verified_sparse_attention(
                sa, query_states, key_states, value_states, scale,
            )

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
          f"max={stats['max']:.4f} count={stats['count']} "
          f"verify_max_leak={_VERIFY_STATS['max_leak']:.4f} "
          f"verify_max_err={_VERIFY_STATS['max_err']:.4f} "
          f"verify_calls={_VERIFY_STATS['calls']} "
          f"verify_skipped_rows={_VERIFY_STATS['skipped']}/{_VERIFY_STATS['rows']}",
          flush=True)
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
