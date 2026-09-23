# MLS-Bench: mlsys-fused-attention

# Fused Attention Kernel Design for H100 GPUs

## Research Question

Design an efficient fused self-attention forward pass kernel using OpenAI
Triton that maximizes throughput (TFLOPs/s) on H100 GPUs while maintaining
numerical correctness.

## Background

Self-attention is the computational bottleneck of Transformer models. The
standard implementation materializes the full N x N attention score matrix,
requiring O(N^2) memory and O(N^2 d) FLOPs. FlashAttention (Dao et al.,
NeurIPS 2022; arXiv:2205.14135) introduced a tiled, IO-aware algorithm
using online softmax that avoids materializing the full matrix, reducing
HBM accesses from O(Nd + N^2) to O(N^2 d^2 / M) where M is SRAM size.

Subsequent versions improved throughput through better parallelism and
hardware-specific optimizations:

- FlashAttention-2 (Dao, 2023; arXiv:2307.08691): reduced non-matmul
  FLOPs, parallelized over sequence length, better warp-level work
  partitioning.
- FlashAttention-3 (Shah et al., NeurIPS 2024; arXiv:2407.08608):
  exploits H100 Hopper features — warp specialization (producer/consumer
  warps overlapping TMA loads with GMMA compute), GEMM-softmax
  interleaving, and FP8 support. The paper reports ~740 TFLOPs/s on H100
  in FP16 (~75% utilization).

## Task

Modify the `custom_attention_forward` function and the associated Triton
kernel `_custom_attn_fwd` to implement an efficient fused attention
forward pass. You may:

- Redesign the tiling strategy (block sizes, tile shapes)
- Optimize the online softmax computation (e.g., use exp2 instead of exp,
  delay rescaling)
- Improve memory access patterns (coalescing, prefetching)
- Split the causal/non-causal iteration into separate passes to avoid
  per-block masking overhead
- Use Triton autotuning (`@triton.autotune`) to search configurations
- Define multiple helper kernels if needed

## Interface

```python
def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    """
    Args:
        q, k, v: (batch, nheads, seqlen, headdim), contiguous, FP16
        causal: if True, apply causal mask (key_pos <= query_pos)
        sm_scale: softmax scale factor (default: 1/sqrt(headdim))
    Returns:
        output: (batch, nheads, seqlen, headdim), same dtype as input
    """
```

Correctness constraint: max absolute difference from reference (PyTorch
SDPA) must be `< 1e-2`.

The kernel must be your own: `custom_attention_forward` may not call the
reference (`F.scaled_dot_product_attention` or any of its backend ops) or
a packaged attention kernel (flash-attn, xformers, cuDNN, ...); such a
call fails the run. Every timed call receives freshly drawn inputs and its
output may be checked against the reference too, so each call must
compute the attention it is given.

## Hints

- The default template provides a basic flash attention kernel. Key
  optimization opportunities:
  1. Two-pass causal: split the K/V loop into non-causal blocks (no mask
     check) and causal boundary blocks, reducing branch overhead
  2. Block size tuning: different `(BLOCK_M, BLOCK_N)` for different
     headdims — larger blocks amortize loop overhead but increase
     register pressure
  3. Triton autotuning: use `@triton.autotune` with `configs=[...]` to
     search block sizes at compile time
  4. Reduced rescaling: in the online softmax, the rescaling
     `acc *= alpha` can be deferred or batched to reduce non-matmul
     operations
  5. Memory coalescing: ensure K/V loads are coalesced along the headdim
     dimension
- The Triton tutorial fused attention demonstrates the two-pass approach
- FA3 achieves its speedup through Hopper-specific CUDA features (warp
  specialization, TMA, GMMA) that are not directly accessible from
  Triton — closing the gap requires algorithmic cleverness in the Triton
  DSL
- Available imports: `torch`, `triton`, `triton.language as tl`, `math`,
  `torch.nn.functional as F`


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/flash-attention/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `flash-attention/custom_triton_bench.py`
- editable lines **29–119**




## Readable Context


### `flash-attention/custom_triton_bench.py`  [EDITABLE — lines 29–119 only]

```python
     1: """Fused Attention Kernel Benchmark — H100 GPU.
     2: 
     3: Benchmark harness for evaluating custom Triton attention kernels.
     4: Aligned with Flash Attention 3 (Shah et al., NeurIPS 2024) benchmarks.
     5: FLOP formula (FA2/FA3): 4 * batch * seqlen^2 * nheads * headdim  (halved for causal).
     6: Total tokens fixed at 16384 (batch = 16384 / seqlen).
     7: """
     8: 
     9: import argparse
    10: import math
    11: import os
    12: import time
    13: 
    14: import torch
    15: import torch.nn.functional as F
    16: import triton
    17: import triton.language as tl
    18: 
    19: # ================================================================
    20: # EDITABLE — Custom Triton Attention Kernel (lines 29 to 119)
    21: # Implement an efficient fused self-attention forward pass.
    22: # Interface: custom_attention_forward(q, k, v, causal, sm_scale) -> output
    23: # Shapes: (batch, nheads, seqlen, headdim), FP16/BF16, contiguous.
    24: # You may import additional modules, define helper kernels, and tune
    25: # block sizes. The only requirement is that custom_attention_forward
    26: # returns correct output matching the reference (max abs diff < 1e-2).
    27: # ================================================================
    28: 
    29: @triton.jit
    30: def _custom_attn_fwd(
    31:     Q, K, V, Out,
    32:     sm_scale,
    33:     stride_qh, stride_qm, stride_qk,
    34:     stride_kh, stride_kn, stride_kk,
    35:     stride_vh, stride_vn, stride_vk,
    36:     stride_oh, stride_om, stride_ok,
    37:     seqlen,
    38:     BLOCK_M: tl.constexpr,
    39:     BLOCK_N: tl.constexpr,
    40:     BLOCK_DMODEL: tl.constexpr,
    41:     IS_CAUSAL: tl.constexpr,
    42: ):
    43:     """Fused self-attention forward kernel (default: basic flash attention)."""
    44:     start_m = tl.program_id(0)
    45:     off_hz = tl.program_id(1)
    46: 
    47:     q_offset = off_hz * stride_qh
    48:     k_offset = off_hz * stride_kh
    49:     v_offset = off_hz * stride_vh
    50:     o_offset = off_hz * stride_oh
    51: 
    52:     offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    53:     offs_n = tl.arange(0, BLOCK_N)
    54:     offs_d = tl.arange(0, BLOCK_DMODEL)
    55: 
    56:     # Load Q tile [BLOCK_M, BLOCK_DMODEL]
    57:     q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    58:     q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    59: 
    60:     # Online softmax accumulators
    61:     m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    62:     l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    63:     acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    64: 
    65:     # Loop over K/V tiles
    66:     hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else seqlen
    67:     for start_n in range(0, hi, BLOCK_N):
    68:         start_n = tl.multiple_of(start_n, BLOCK_N)
    69:         # Load K tile [BLOCK_N, BLOCK_DMODEL]
    70:         k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
    71:         k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
    72:         # S = Q @ K^T * scale  [BLOCK_M, BLOCK_N]
    73:         qk = tl.dot(q, tl.trans(k)) * sm_scale
    74:         if IS_CAUSAL:
    75:             qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
    76:         # Online softmax
    77:         m_ij = tl.max(qk, axis=1)
    78:         m_new = tl.maximum(m_i, m_ij)
    79:         alpha = tl.math.exp2((m_i - m_new) * 1.44269504)
    80:         p = tl.math.exp2((qk - m_new[:, None]) * 1.44269504)
    81:         l_i = l_i * alpha + tl.sum(p, axis=1)
    82:         acc = acc * alpha[:, None]
    83:         # Load V tile and accumulate [BLOCK_N, BLOCK_DMODEL]
    84:         v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
    85:         v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
    86:         acc += tl.dot(p.to(v.dtype), v)
    87:         m_i = m_new
    88: 
    89:     # Normalize and store
    90:     acc = acc / l_i[:, None]
    91:     o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    92:     tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)
    93: 
    94: 
    95: def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    96:     """Python wrapper for the custom Triton attention kernel."""
    97:     batch, nheads, seqlen, headdim = q.shape
    98:     q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    99:     if sm_scale is None:
   100:         sm_scale = 1.0 / math.sqrt(headdim)
   101:     o = torch.empty_like(q)
   102:     # Conservative block sizes (FA1-style). Opportunities to optimize:
   103:     # - Tune BLOCK_M/BLOCK_N per headdim (larger blocks for smaller heads)
   104:     # - Use @triton.autotune to search configurations
   105:     # - Adjust num_warps and num_stages
   106:     BLOCK_M, BLOCK_N = 64, 64
   107:     grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
   108:     _custom_attn_fwd[grid](
   109:         q, k, v, o, sm_scale,
   110:         q.stride(1), q.stride(2), q.stride(3),
   111:         k.stride(1), k.stride(2), k.stride(3),
   112:         v.stride(1), v.stride(2), v.stride(3),
   113:         o.stride(1), o.stride(2), o.stride(3),
   114:         seqlen,
   115:         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
   116:         BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
   117:     )
   118:     return o
   119: 
   120: # ================================================================
   121: # FIXED — Benchmark Harness (do not modify below this line)
   122: # ================================================================
   123: 
   124: import secrets
   125: import sys
   126: 
   127: from torch.utils._python_dispatch import TorchDispatchMode
   128: 
   129: 
   130: def reference_attention(q, k, v, causal=True, sm_scale=None):
   131:     """PyTorch SDPA reference (dispatches to cuDNN/FlashAttention internally)."""
   132:     if sm_scale is None:
   133:         sm_scale = 1.0 / math.sqrt(q.shape[-1])
   134:     with torch.backends.cuda.sdp_kernel(
   135:         enable_flash=True, enable_math=True, enable_mem_efficient=True
   136:     ):
   137:         return F.scaled_dot_product_attention(
   138:             q, k, v, is_causal=causal, scale=sm_scale
   139:         )
   140: 
   141: 
   142: # The task is to write the attention kernel, so the custom forward may not
   143: # hand the work to the reference (SDPA and every backend it dispatches to) or
   144: # to a packaged attention kernel. Every call of custom_attention_forward runs
   145: # under _NoLibraryAttention, which raises on these ops.
   146: _FORBIDDEN_OP_PREFIXES = (
   147:     "aten::scaled_dot_product",
   148:     "aten::_scaled_dot_product",
   149:     "aten::_fused_sdp_choice",
   150:     "aten::_flash_attention",
   151:     "aten::_efficient_attention",
   152:     "aten::_cudnn_attention",
   153:     "aten::_native_multi_head_attention",
   154:     "aten::_transformer_encoder_layer_fwd",
   155:     "aten::_triton_scaled_dot_attention",
   156:     "aten::_triton_multi_head_attention",
   157:     "flash_attn",
   158:     "xformers",
   159:     "transformer_engine",
   160: )
   161: _FORBIDDEN_MODULES = (
   162:     "flash_attn", "flash_attn_2_cuda", "flash_attn_interface", "flash_attn_3",
   163:     "flashattn_hopper_cuda", "flash_attn_3_cuda", "hopper", "xformers",
   164:     "cudnn", "transformer_engine", "flashinfer", "sageattention",
   165: )
   166: 
   167: 
   168: class _NoLibraryAttention(TorchDispatchMode):
   169:     def __torch_dispatch__(self, func, types, args=(), kwargs=None):
   170:         name = func._schema.name
   171:         if name.startswith(_FORBIDDEN_OP_PREFIXES):
   172:             raise RuntimeError(
   173:                 f"custom_attention_forward called the library attention op "
   174:                 f"{name}; the kernel must be your own")
   175:         return func(*args, **(kwargs or {}))
   176: 
   177: 
   178: def _forbidden_modules_loaded():
   179:     return sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN_MODULES)
   180: 
   181: 
   182: def _custom_call(q, k, v, causal=True, sm_scale=None):
   183:     with _NoLibraryAttention():
   184:         out = custom_attention_forward(q, k, v, causal=causal, sm_scale=sm_scale)
   185:     if type(out) is not torch.Tensor or out.shape != q.shape or out.dtype != q.dtype:
   186:         raise RuntimeError(
   187:             f"custom_attention_forward must return a plain tensor of shape "
   188:             f"{tuple(q.shape)} and dtype {q.dtype}")
   189:     return out
   190: 
   191: 
   192: def compute_flops(batch, nheads, seqlen, headdim, causal):
   193:     """FLOPs for attention forward (FA2/FA3 convention)."""
   194:     flops = 4 * batch * seqlen * seqlen * nheads * headdim
   195:     if causal:
   196:         flops //= 2
   197:     return flops
   198: 
   199: 
   200: def benchmark_fn(fn, shape, dtype, causal, sm_scale, warmup=25, rep=100,
   201:                  verify=0):
   202:     """Benchmark and return (median latency in ms, max abs diff).
   203: 
   204:     Every call gets freshly drawn q/k/v from an unpredictable seed, so no
   205:     output can be reused across calls or precomputed. `verify` timed calls,
   206:     picked at random and unknowable to `fn`, are checked against the
   207:     reference outside the timed span; a kernel that is wrong on enough calls
   208:     to move the median is caught with near certainty.
   209:     """
   210:     gen = torch.Generator(device="cuda")
   211:     gen.manual_seed(secrets.randbits(62))
   212:     checked = set(secrets.SystemRandom().sample(range(rep), min(verify, rep)))
   213: 
   214:     def draw():
   215:         return tuple(torch.randn(shape, dtype=dtype, device="cuda", generator=gen)
   216:                      for _ in range(3))
   217: 
   218:     # Warmup
   219:     for _ in range(warmup):
   220:         q, k, v = draw()
   221:         fn(q, k, v, causal=causal, sm_scale=sm_scale)
   222:     torch.cuda.synchronize()
   223: 
   224:     # Timed runs
   225:     start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
   226:     end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
   227:     max_diff = torch.zeros((), dtype=torch.float32, device="cuda")
   228:     for i in range(rep):
   229:         q, k, v = draw()
   230:         start_events[i].record()
   231:         out = fn(q, k, v, causal=causal, sm_scale=sm_scale)
   232:         end_events[i].record()
   233:         if i in checked:
   234:             ref = reference_attention(q, k, v, causal=causal, sm_scale=sm_scale)
   235:             max_diff = torch.maximum(
   236:                 max_diff, (out.float() - ref.float()).abs().max())
   237:         del out
   238:     torch.cuda.synchronize()
   239: 
   240:     times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
   241:     times.sort()
   242:     return times[len(times) // 2], max_diff.item()  # median ms
   243: 
   244: 
   245: def main():
   246:     parser = argparse.ArgumentParser(description="Fused Attention Kernel Benchmark")
   247:     parser.add_argument("--batch", type=int, required=True)
   248:     parser.add_argument("--seqlen", type=int, required=True)
   249:     parser.add_argument("--nheads", type=int, required=True)
   250:     parser.add_argument("--headdim", type=int, required=True)
   251:     parser.add_argument("--causal", action="store_true")
   252:     parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
   253:     parser.add_argument("--warmup", type=int, default=25)
   254:     parser.add_argument("--rep", type=int, default=100)
   255:     parser.add_argument("--output-dir", type=str, default=".")
   256:     parser.add_argument("--seed", type=int, default=42)
   257:     args = parser.parse_args()
   258: 
   259:     torch.manual_seed(args.seed)
   260:     torch.cuda.manual_seed_all(args.seed)
   261: 
   262:     dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
   263:     device = "cuda"
   264: 
   265:     q = torch.randn(args.batch, args.nheads, args.seqlen, args.headdim,
   266:                      dtype=dtype, device=device)
   267:     k = torch.randn_like(q)
   268:     v = torch.randn_like(q)
   269:     sm_scale = 1.0 / math.sqrt(args.headdim)
   270: 
   271:     flops = compute_flops(args.batch, args.nheads, args.seqlen, args.headdim,
   272:                           args.causal)
   273: 
   274:     print(f"Config: batch={args.batch} seqlen={args.seqlen} nheads={args.nheads} "
   275:           f"headdim={args.headdim} causal={args.causal} dtype={args.dtype}")
   276:     print(f"FLOPs: {flops / 1e12:.3f} TFLOPs per forward pass")
   277: 
   278:     CORRECTNESS_THRESHOLD = 1e-2
   279: 
   280:     def fail(reason, max_diff=1.0):
   281:         print(f"ERROR: {reason}")
   282:         print(f"TEST_METRICS: speedup_vs_sdpa=0.0 tflops=0.0 latency_ms=999999.0 "
   283:               f"sdpa_latency_ms=0.0 max_diff={max_diff:.6e} correct=0")
   284: 
   285:     # --- Correctness check ---
   286:     ref_out = reference_attention(q, k, v, causal=args.causal, sm_scale=sm_scale)
   287:     try:
   288:         custom_out = _custom_call(q, k, v, causal=args.causal, sm_scale=sm_scale)
   289:     except Exception as e:
   290:         fail(f"custom kernel failed: {e}")
   291:         return
   292: 
   293:     max_diff = (custom_out.float() - ref_out.float()).abs().max().item()
   294:     mean_diff = (custom_out.float() - ref_out.float()).abs().mean().item()
   295:     del custom_out, ref_out
   296:     print(f"TRAIN_METRICS: max_diff={max_diff:.6e} mean_diff={mean_diff:.6e}")
   297: 
   298:     if not max_diff <= CORRECTNESS_THRESHOLD:
   299:         print(f"FAIL: max_diff {max_diff:.6e} > threshold {CORRECTNESS_THRESHOLD}")
   300:         print(f"TEST_METRICS: speedup_vs_sdpa=0.0 tflops=0.0 latency_ms=999999.0 "
   301:               f"sdpa_latency_ms=0.0 max_diff={max_diff:.6e} correct=0")
   302:         return
   303: 
   304:     # --- Throughput benchmark ---
   305:     # Benchmark BOTH custom kernel and PyTorch SDPA reference. The primary
   306:     # cross-GPU-comparable metric is `speedup_vs_sdpa`: SDPA dispatches to
   307:     # the best fused kernel available on the current GPU (cuDNN/FA2 on A100,
   308:     # cuDNN/FA3 on H100/H200), so the ratio measures algorithmic merit
   309:     # independent of the card's absolute throughput. Every timed call of the
   310:     # custom kernel sees fresh inputs, and 16 of them are checked too.
   311:     shape = q.shape
   312:     del q, k, v
   313:     try:
   314:         latency_ms, timed_max_diff = benchmark_fn(
   315:             _custom_call, shape, dtype, args.causal, sm_scale,
   316:             warmup=args.warmup, rep=args.rep, verify=16)
   317:     except Exception as e:
   318:         fail(f"custom kernel failed during the timed runs: {e}")
   319:         return
   320:     print(f"TRAIN_METRICS: timed_max_diff={timed_max_diff:.6e}")
   321:     if not timed_max_diff <= CORRECTNESS_THRESHOLD:
   322:         fail(f"timed-run max_diff {timed_max_diff:.6e} > threshold "
   323:              f"{CORRECTNESS_THRESHOLD}", max(max_diff, timed_max_diff))
   324:         return
   325:     loaded = _forbidden_modules_loaded()
   326:     if loaded:
   327:         fail(f"library attention modules were imported: {', '.join(loaded)}; "
   328:              f"the kernel must be your own")
   329:         return
   330:     sdpa_latency_ms, _ = benchmark_fn(reference_attention, shape, dtype,
   331:                                       args.causal, sm_scale,
   332:                                       warmup=args.warmup, rep=args.rep)
   333:     tflops = flops / (latency_ms * 1e-3) / 1e12
   334:     sdpa_tflops = flops / (sdpa_latency_ms * 1e-3) / 1e12
   335:     speedup_vs_sdpa = sdpa_latency_ms / latency_ms
   336: 
   337:     print(f"TRAIN_METRICS: latency_ms={latency_ms:.3f} tflops={tflops:.1f} "
   338:           f"sdpa_latency_ms={sdpa_latency_ms:.3f} sdpa_tflops={sdpa_tflops:.1f} "
   339:           f"speedup_vs_sdpa={speedup_vs_sdpa:.3f}")
   340:     print(f"TEST_METRICS: speedup_vs_sdpa={speedup_vs_sdpa:.4f} "
   341:           f"tflops={tflops:.4f} latency_ms={latency_ms:.4f} "
   342:           f"sdpa_latency_ms={sdpa_latency_ms:.4f} "
   343:           f"max_diff={max_diff:.6e} correct=1")
   344: 
   345: 
   346: if __name__ == "__main__":
   347:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `flash_v1` baseline — editable region  [READ-ONLY — reference implementation]

In `flash-attention/custom_triton_bench.py`:

```python
Lines 29–106:
    26: # returns correct output matching the reference (max abs diff < 1e-2).
    27: # ================================================================
    28: 
    29: @triton.jit
    30: def _custom_attn_fwd(
    31:     Q, K, V, Out,
    32:     sm_scale,
    33:     stride_qh, stride_qm, stride_qk,
    34:     stride_kh, stride_kn, stride_kk,
    35:     stride_vh, stride_vn, stride_vk,
    36:     stride_oh, stride_om, stride_ok,
    37:     seqlen,
    38:     BLOCK_M: tl.constexpr,
    39:     BLOCK_N: tl.constexpr,
    40:     BLOCK_DMODEL: tl.constexpr,
    41:     IS_CAUSAL: tl.constexpr,
    42: ):
    43:     """FA1-style: single-pass tiling + online softmax, causal mask every block."""
    44:     start_m = tl.program_id(0)
    45:     off_hz = tl.program_id(1)
    46: 
    47:     q_offset = off_hz * stride_qh
    48:     k_offset = off_hz * stride_kh
    49:     v_offset = off_hz * stride_vh
    50:     o_offset = off_hz * stride_oh
    51: 
    52:     offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    53:     offs_n = tl.arange(0, BLOCK_N)
    54:     offs_d = tl.arange(0, BLOCK_DMODEL)
    55: 
    56:     q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    57:     q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    58: 
    59:     m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    60:     l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    61:     acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    62: 
    63:     hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else seqlen
    64:     for start_n in range(0, hi, BLOCK_N):
    65:         start_n = tl.multiple_of(start_n, BLOCK_N)
    66:         k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
    67:         k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
    68:         qk = tl.dot(q, tl.trans(k)) * sm_scale
    69:         if IS_CAUSAL:
    70:             qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
    71:         m_ij = tl.max(qk, axis=1)
    72:         m_new = tl.maximum(m_i, m_ij)
    73:         alpha = tl.math.exp2((m_i - m_new) * 1.44269504)
    74:         p = tl.math.exp2((qk - m_new[:, None]) * 1.44269504)
    75:         l_i = l_i * alpha + tl.sum(p, axis=1)
    76:         acc = acc * alpha[:, None]
    77:         v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
    78:         v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
    79:         acc += tl.dot(p.to(v.dtype), v)
    80:         m_i = m_new
    81: 
    82:     acc = acc / l_i[:, None]
    83:     o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    84:     tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)
    85: 
    86: 
    87: def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
    88:     """FA1-style wrapper with uniform block sizes."""
    89:     batch, nheads, seqlen, headdim = q.shape
    90:     q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    91:     if sm_scale is None:
    92:         sm_scale = 1.0 / math.sqrt(headdim)
    93:     o = torch.empty_like(q)
    94:     BLOCK_M, BLOCK_N = 64, 64
    95:     grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
    96:     _custom_attn_fwd[grid](
    97:         q, k, v, o, sm_scale,
    98:         q.stride(1), q.stride(2), q.stride(3),
    99:         k.stride(1), k.stride(2), k.stride(3),
   100:         v.stride(1), v.stride(2), v.stride(3),
   101:         o.stride(1), o.stride(2), o.stride(3),
   102:         seqlen,
   103:         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
   104:         BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
   105:     )
   106:     return o
   107: # ================================================================
   108: # FIXED — Benchmark Harness (do not modify below this line)
   109: # ================================================================
```

### `flash_v2` baseline — editable region  [READ-ONLY — reference implementation]

In `flash-attention/custom_triton_bench.py`:

```python
Lines 29–142:
    26: # returns correct output matching the reference (max abs diff < 1e-2).
    27: # ================================================================
    28: 
    29: @triton.jit
    30: def _flash_v2_fwd(
    31:     Q, K, V, Out,
    32:     stride_qh, stride_qm, stride_qk,
    33:     stride_kh, stride_kn, stride_kk,
    34:     stride_vh, stride_vn, stride_vk,
    35:     stride_oh, stride_om, stride_ok,
    36:     seqlen,
    37:     BLOCK_M: tl.constexpr,
    38:     BLOCK_N: tl.constexpr,
    39:     BLOCK_DMODEL: tl.constexpr,
    40:     IS_CAUSAL: tl.constexpr,
    41: ):
    42:     """FA2-style: two-pass causal, scale fused into Q, tuned block sizes."""
    43:     start_m = tl.program_id(0)
    44:     off_hz = tl.program_id(1)
    45: 
    46:     q_offset = off_hz * stride_qh
    47:     k_offset = off_hz * stride_kh
    48:     v_offset = off_hz * stride_vh
    49:     o_offset = off_hz * stride_oh
    50: 
    51:     offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    52:     offs_n = tl.arange(0, BLOCK_N)
    53:     offs_d = tl.arange(0, BLOCK_DMODEL)
    54: 
    55:     # Load Q and pre-multiply scale (FA2: fuse scale into Q)
    56:     q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    57:     q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    58: 
    59:     m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    60:     l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    61:     acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    62: 
    63:     # --- Pass 1: non-causal blocks (all positions valid, skip masking) ---
    64:     if IS_CAUSAL:
    65:         causal_boundary = start_m * BLOCK_M
    66:         non_causal_end = (causal_boundary // BLOCK_N) * BLOCK_N
    67:     else:
    68:         non_causal_end = seqlen
    69:         causal_boundary = seqlen
    70: 
    71:     for start_n in range(0, non_causal_end, BLOCK_N):
    72:         start_n = tl.multiple_of(start_n, BLOCK_N)
    73:         k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
    74:         k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
    75:         qk = tl.dot(q, tl.trans(k))
    76:         m_ij = tl.max(qk, axis=1)
    77:         m_new = tl.maximum(m_i, m_ij)
    78:         alpha = tl.math.exp2(m_i - m_new)
    79:         p = tl.math.exp2(qk - m_new[:, None])
    80:         l_i = l_i * alpha + tl.sum(p, axis=1)
    81:         acc = acc * alpha[:, None]
    82:         v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
    83:         v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
    84:         acc += tl.dot(p.to(v.dtype), v)
    85:         m_i = m_new
    86: 
    87:     # --- Pass 2: causal boundary blocks (need masking) ---
    88:     if IS_CAUSAL:
    89:         hi = (start_m + 1) * BLOCK_M
    90:     else:
    91:         hi = non_causal_end
    92: 
    93:     for start_n in range(non_causal_end, hi, BLOCK_N):
    94:         start_n = tl.multiple_of(start_n, BLOCK_N)
    95:         k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
    96:         k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
    97:         qk = tl.dot(q, tl.trans(k))
    98:         qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
    99:         m_ij = tl.max(qk, axis=1)
   100:         m_new = tl.maximum(m_i, m_ij)
   101:         alpha = tl.math.exp2(m_i - m_new)
   102:         p = tl.math.exp2(qk - m_new[:, None])
   103:         l_i = l_i * alpha + tl.sum(p, axis=1)
   104:         acc = acc * alpha[:, None]
   105:         v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
   106:         v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
   107:         acc += tl.dot(p.to(v.dtype), v)
   108:         m_i = m_new
   109: 
   110:     acc = acc / l_i[:, None]
   111:     o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
   112:     tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)
   113: 
   114: 
   115: def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
   116:     """FA2-style wrapper with per-headdim block sizes and fused scale."""
   117:     batch, nheads, seqlen, headdim = q.shape
   118:     q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
   119:     if sm_scale is None:
   120:         sm_scale = 1.0 / math.sqrt(headdim)
   121:     # FA2 optimization: fuse sm_scale into Q (saves one mul per element in inner loop)
   122:     q = (q * (sm_scale * 1.44269504)).contiguous()
   123:     o = torch.empty_like(q)
   124:     # FA2: per-headdim block sizes for better tensor core utilization
   125:     if headdim <= 64:
   126:         BLOCK_M, BLOCK_N = 128, 64
   127:     elif headdim <= 128:
   128:         BLOCK_M, BLOCK_N = 128, 64
   129:     else:
   130:         BLOCK_M, BLOCK_N = 64, 64
   131:     grid = (triton.cdiv(seqlen, BLOCK_M), batch * nheads)
   132:     _flash_v2_fwd[grid](
   133:         q, k, v, o,
   134:         q.stride(1), q.stride(2), q.stride(3),
   135:         k.stride(1), k.stride(2), k.stride(3),
   136:         v.stride(1), v.stride(2), v.stride(3),
   137:         o.stride(1), o.stride(2), o.stride(3),
   138:         seqlen,
   139:         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
   140:         BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
   141:     )
   142:     return o
   143: # ================================================================
   144: # FIXED — Benchmark Harness (do not modify below this line)
   145: # ================================================================
```

### `flash_v3` baseline — editable region  [READ-ONLY — reference implementation]

In `flash-attention/custom_triton_bench.py`:

```python
Lines 29–147:
    26: # returns correct output matching the reference (max abs diff < 1e-2).
    27: # ================================================================
    28: 
    29: @triton.autotune(
    30:     configs=[
    31:         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
    32:         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
    33:         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
    34:         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
    35:         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
    36:         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
    37:         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
    38:         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
    39:     ],
    40:     key=['seqlen', 'BLOCK_DMODEL', 'IS_CAUSAL'],
    41: )
    42: @triton.jit
    43: def _flash_v3_fwd(
    44:     Q, K, V, Out,
    45:     stride_qh, stride_qm, stride_qk,
    46:     stride_kh, stride_kn, stride_kk,
    47:     stride_vh, stride_vn, stride_vk,
    48:     stride_oh, stride_om, stride_ok,
    49:     seqlen,
    50:     BLOCK_M: tl.constexpr,
    51:     BLOCK_N: tl.constexpr,
    52:     BLOCK_DMODEL: tl.constexpr,
    53:     IS_CAUSAL: tl.constexpr,
    54: ):
    55:     """FA3-inspired: autotuned two-pass causal with software pipelining."""
    56:     start_m = tl.program_id(0)
    57:     off_hz = tl.program_id(1)
    58: 
    59:     q_offset = off_hz * stride_qh
    60:     k_offset = off_hz * stride_kh
    61:     v_offset = off_hz * stride_vh
    62:     o_offset = off_hz * stride_oh
    63: 
    64:     offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    65:     offs_n = tl.arange(0, BLOCK_N)
    66:     offs_d = tl.arange(0, BLOCK_DMODEL)
    67: 
    68:     # Load Q with scale already fused (done in wrapper)
    69:     q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    70:     q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    71: 
    72:     m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    73:     l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    74:     acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    75: 
    76:     # --- Pass 1: non-causal blocks (no mask, better pipelining) ---
    77:     if IS_CAUSAL:
    78:         causal_boundary = start_m * BLOCK_M
    79:         non_causal_end = (causal_boundary // BLOCK_N) * BLOCK_N
    80:     else:
    81:         non_causal_end = seqlen
    82:         causal_boundary = seqlen
    83: 
    84:     for start_n in range(0, non_causal_end, BLOCK_N):
    85:         start_n = tl.multiple_of(start_n, BLOCK_N)
    86:         k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
    87:         k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
    88:         qk = tl.dot(q, tl.trans(k))
    89:         m_ij = tl.max(qk, axis=1)
    90:         m_new = tl.maximum(m_i, m_ij)
    91:         alpha = tl.math.exp2(m_i - m_new)
    92:         p = tl.math.exp2(qk - m_new[:, None])
    93:         l_i = l_i * alpha + tl.sum(p, axis=1)
    94:         acc = acc * alpha[:, None]
    95:         v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
    96:         v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
    97:         acc += tl.dot(p.to(v.dtype), v)
    98:         m_i = m_new
    99: 
   100:     # --- Pass 2: causal boundary blocks ---
   101:     if IS_CAUSAL:
   102:         hi = (start_m + 1) * BLOCK_M
   103:     else:
   104:         hi = non_causal_end
   105: 
   106:     for start_n in range(non_causal_end, hi, BLOCK_N):
   107:         start_n = tl.multiple_of(start_n, BLOCK_N)
   108:         k_ptrs = K + k_offset + (start_n + offs_n[:, None]) * stride_kn + offs_d[None, :] * stride_kk
   109:         k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
   110:         qk = tl.dot(q, tl.trans(k))
   111:         qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
   112:         m_ij = tl.max(qk, axis=1)
   113:         m_new = tl.maximum(m_i, m_ij)
   114:         alpha = tl.math.exp2(m_i - m_new)
   115:         p = tl.math.exp2(qk - m_new[:, None])
   116:         l_i = l_i * alpha + tl.sum(p, axis=1)
   117:         acc = acc * alpha[:, None]
   118:         v_ptrs = V + v_offset + (start_n + offs_n[:, None]) * stride_vn + offs_d[None, :] * stride_vk
   119:         v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seqlen, other=0.0)
   120:         acc += tl.dot(p.to(v.dtype), v)
   121:         m_i = m_new
   122: 
   123:     acc = acc / l_i[:, None]
   124:     o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
   125:     tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seqlen)
   126: 
   127: 
   128: def custom_attention_forward(q, k, v, causal=True, sm_scale=None):
   129:     """FA3-inspired: autotuned pipelining + fused scale + two-pass causal."""
   130:     batch, nheads, seqlen, headdim = q.shape
   131:     q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
   132:     if sm_scale is None:
   133:         sm_scale = 1.0 / math.sqrt(headdim)
   134:     # Fuse scale into Q
   135:     q = (q * (sm_scale * 1.44269504)).contiguous()
   136:     o = torch.empty_like(q)
   137:     grid = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch * nheads)
   138:     _flash_v3_fwd[grid](
   139:         q, k, v, o,
   140:         q.stride(1), q.stride(2), q.stride(3),
   141:         k.stride(1), k.stride(2), k.stride(3),
   142:         v.stride(1), v.stride(2), v.stride(3),
   143:         o.stride(1), o.stride(2), o.stride(3),
   144:         seqlen,
   145:         BLOCK_DMODEL=headdim, IS_CAUSAL=causal,
   146:     )
   147:     return o
   148: # ================================================================
   149: # FIXED — Benchmark Harness (do not modify below this line)
   150: # ================================================================
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

## Time Budget

You have **5 hours** of wall-clock time before submission, covering
everything you do here: reading the code, editing it, and any trial runs
you launch.

Good luck.
