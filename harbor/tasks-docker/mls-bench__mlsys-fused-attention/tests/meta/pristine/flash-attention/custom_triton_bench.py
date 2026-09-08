"""Fused Attention Kernel Benchmark — H100 GPU.

Benchmark harness for evaluating custom Triton attention kernels.
Aligned with Flash Attention 3 (Shah et al., NeurIPS 2024) benchmarks.
FLOP formula (FA2/FA3): 4 * batch * seqlen^2 * nheads * headdim  (halved for causal).
Total tokens fixed at 16384 (batch = 16384 / seqlen).
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ================================================================
# EDITABLE — Custom Triton Attention Kernel (lines 29 to 119)
# Implement an efficient fused self-attention forward pass.
# Interface: custom_attention_forward(q, k, v, causal, sm_scale) -> output
# Shapes: (batch, nheads, seqlen, headdim), FP16/BF16, contiguous.
# You may import additional modules, define helper kernels, and tune
# block sizes. The only requirement is that custom_attention_forward
# returns correct output matching the reference (max abs diff < 1e-2).
# ================================================================

@triton.jit
def _custom_attn_fwd(
    Q, K, V, Out,
    sm_scale,
    stride_qh, stride_qm, stride_qk,
    stride_kh, stride_kn, stride_kk,
    stride_vh, stride_vn, stride_vk,
    stride_oh, stride_om, stride_ok,
    seqlen,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Fused self-attention forward kernel (default: basic flash attention)."""
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    q_offset = off_hz * stride_qh
    k_offset = off_hz * stride_kh
    v_offset = off_hz * stride_vh
    o_offset = off_hz * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q tile [BLOCK_M, BLOCK_DMODEL]
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)

    # Online softmax accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Loop over K/V tiles
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else seqlen
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # Load K tile [BLOCK_N, BLOCK_DMODEL]
        k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        # S = Q @ K^T * scale  [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2((m_i - m_new) * 1.44269504)
        p = tl.math.exp2((qk - m_new[:, None]) * 1.44269504)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        # Load V tile and accumulate [BLOCK_N, BLOCK_DMODEL]
        v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Normalize and store
    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)


def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    """Python wrapper for the custom Triton attention kernel."""
    batch, nheads, seqlen, headdim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(headdim)
    o = torch.empty_like(q)
    # Conservative block sizes (FA1-style). Opportunities to optimize:
    # - Tune BLOCK_M/BLOCK_N per headdim (larger blocks for smaller heads)
    # - Use @triton.autotune to search configurations
    # - Adjust num_warps and num_stages
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
    _custom_attn_fwd[grid](
        q, k, v, o, sm_scale,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        o.stride(1), o.stride(2), o.stride(3),
        seqlen,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
    )
    return o

# ================================================================
# FIXED — Benchmark Harness (do not modify below this line)
# ================================================================


def reference_attention(q, k, v, causal=True, sm_scale=None):
    """PyTorch SDPA reference (dispatches to cuDNN/FlashAttention internally)."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=True, enable_mem_efficient=True
    ):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=sm_scale
        )


def compute_flops(batch, nheads, seqlen, headdim, causal):
    """FLOPs for attention forward (FA2/FA3 convention)."""
    flops = 4 * batch * seqlen * seqlen * nheads * headdim
    if causal:
        flops //= 2
    return flops


def benchmark_fn(fn, q, k, v, causal, sm_scale, warmup=25, rep=100):
    """Benchmark and return median latency in ms."""
    # Warmup
    for _ in range(warmup):
        fn(q, k, v, causal=causal, sm_scale=sm_scale)
    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        fn(q, k, v, causal=causal, sm_scale=sm_scale)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2]  # median ms


def main():
    parser = argparse.ArgumentParser(description="Fused Attention Kernel Benchmark")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--nheads", type=int, required=True)
    parser.add_argument("--headdim", type=int, required=True)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = "cuda"

    q = torch.randn(args.batch, args.nheads, args.seqlen, args.headdim,
                     dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sm_scale = 1.0 / math.sqrt(args.headdim)

    flops = compute_flops(args.batch, args.nheads, args.seqlen, args.headdim,
                          args.causal)

    print(f"Config: batch={args.batch} seqlen={args.seqlen} nheads={args.nheads} "
          f"headdim={args.headdim} causal={args.causal} dtype={args.dtype}")
    print(f"FLOPs: {flops / 1e12:.3f} TFLOPs per forward pass")

    # --- Correctness check ---
    ref_out = reference_attention(q, k, v, causal=args.causal, sm_scale=sm_scale)
    try:
        custom_out = custom_attention_forward(q, k, v, causal=args.causal,
                                              sm_scale=sm_scale)
    except Exception as e:
        print(f"ERROR: custom kernel failed: {e}")
        print(f"TEST_METRICS: speedup_vs_sdpa=0.0 tflops=0.0 latency_ms=999999.0 "
              f"sdpa_latency_ms=0.0 max_diff=1.0 correct=0")
        return

    max_diff = (custom_out.float() - ref_out.float()).abs().max().item()
    mean_diff = (custom_out.float() - ref_out.float()).abs().mean().item()
    print(f"TRAIN_METRICS: max_diff={max_diff:.6e} mean_diff={mean_diff:.6e}")

    CORRECTNESS_THRESHOLD = 1e-2
    if max_diff > CORRECTNESS_THRESHOLD:
        print(f"FAIL: max_diff {max_diff:.6e} > threshold {CORRECTNESS_THRESHOLD}")
        print(f"TEST_METRICS: speedup_vs_sdpa=0.0 tflops=0.0 latency_ms=999999.0 "
              f"sdpa_latency_ms=0.0 max_diff={max_diff:.6e} correct=0")
        return

    # --- Throughput benchmark ---
    # Benchmark BOTH custom kernel and PyTorch SDPA reference. The primary
    # cross-GPU-comparable metric is `speedup_vs_sdpa`: SDPA dispatches to
    # the best fused kernel available on the current GPU (cuDNN/FA2 on A100,
    # cuDNN/FA3 on H100/H200), so the ratio measures algorithmic merit
    # independent of the card's absolute throughput.
    latency_ms = benchmark_fn(custom_attention_forward, q, k, v,
                              args.causal, sm_scale,
                              warmup=args.warmup, rep=args.rep)
    sdpa_latency_ms = benchmark_fn(reference_attention, q, k, v,
                                   args.causal, sm_scale,
                                   warmup=args.warmup, rep=args.rep)
    tflops = flops / (latency_ms * 1e-3) / 1e12
    sdpa_tflops = flops / (sdpa_latency_ms * 1e-3) / 1e12
    speedup_vs_sdpa = sdpa_latency_ms / latency_ms

    print(f"TRAIN_METRICS: latency_ms={latency_ms:.3f} tflops={tflops:.1f} "
          f"sdpa_latency_ms={sdpa_latency_ms:.3f} sdpa_tflops={sdpa_tflops:.1f} "
          f"speedup_vs_sdpa={speedup_vs_sdpa:.3f}")
    print(f"TEST_METRICS: speedup_vs_sdpa={speedup_vs_sdpa:.4f} "
          f"tflops={tflops:.4f} latency_ms={latency_ms:.4f} "
          f"sdpa_latency_ms={sdpa_latency_ms:.4f} "
          f"max_diff={max_diff:.6e} correct=1")


if __name__ == "__main__":
    main()
