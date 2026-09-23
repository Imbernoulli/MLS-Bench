# MLS-Bench: llm-ptq-algorithm

# LLM Post-Training Quantization (PTQ) Algorithm

## Research Question

Design a post-training quantization algorithm that minimizes accuracy
degradation when quantizing a pretrained large language model to low-bit
integer precision, without any retraining or fine-tuning.

## Background

Post-training quantization (PTQ) compresses neural-network weights from
floating-point to low-bit integer representations after training is
complete. Unlike quantization-aware training (QAT), which modifies the
training procedure, PTQ works on already-trained models and requires no
gradient updates to the original weights, which is attractive for LLMs
where retraining is prohibitively expensive.

The challenge is severe at low bit-widths: INT4 has only 16 discrete
levels (vs 256 for INT8), and INT3 has only 8 levels, so naive rounding
causes significant accuracy loss. This is amplified at 7B+ scale where
weight distributions are complex and quantization errors accumulate
across many transformer layers. Reference families:

- RTN (Round-To-Nearest): round each weight to its nearest quantized
  value. Fast but high degradation.
- SmoothQuant (Xiao et al., ICML 2023; arXiv:2211.10438): migrate
  quantization difficulty from activations to weights via a per-channel
  equivalent transformation, making weight distributions easier to
  quantize.
- GPTQ (Frantar et al., ICLR 2023; arXiv:2210.17323): use calibration data
  to compute an approximate Hessian, then quantize weights column-by-column
  while compensating remaining error using second-order information.
- AWQ (Lin et al., MLSys 2024 Best Paper; arXiv:2306.00978): identify
  salient weight channels via activation magnitudes and protect them with
  per-channel scaling, without requiring Hessian computation.

Quantization here uses symmetric group quantization: weights are
partitioned into groups of consecutive columns (group size 64 or 128),
and one scale factor is computed per group per output row.

## What You Can Modify

The `LayerQuantizer` class and helper functions in `custom_ptq.py`:

- `quantize_tensor()` / `dequantize_tensor()`: basic quantization
  primitives
- `find_scale_zero()`: scale/zero-point computation (per-channel or
  per-group)
- `LayerQuantizer.__init__()`: set hyperparameters; receives `num_bits`
  and `group_size` from the evaluation script
- `LayerQuantizer.add_batch(inp)`: collect statistics from calibration
  data
- `LayerQuantizer.quantize()`: apply quantization to the layer's weight
  matrix

You can implement any approach: error compensation, weight transformation
(scaling, rotation, smoothing), mixed strategies, outlier handling, or
adaptive grouping schemes that vary by group size or bit-width.

## Architecture

The task loads a pretrained LLM from a local path and quantizes it.
No training is done — the task is purely about the quantization algorithm quality.

The script (`custom_ptq.py`):

1. Loads a pretrained LLM from a pre-downloaded local path
2. Evaluates the FP16 (unquantized) model as baseline
3. Runs your `LayerQuantizer.add_batch()` on calibration data layer by
   layer
4. Quantizes each linear layer using your `LayerQuantizer.quantize()`
5. Evaluates the quantized model and reports perplexity degradation

## Interface

```python
class LayerQuantizer:
    def __init__(self, layer, num_bits=4, group_size=-1):
        # layer: nn.Linear to quantize
        # num_bits: target bit width (4 or 3, set by evaluation)
        # group_size: columns per group (-1 = per-channel, 128 or 64)
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        # ... initialize calibration buffers

    def add_batch(self, inp):
        # inp: layer input tensor, shape (batch*seq_len, in_features)
        pass

    def quantize(self):
        # Returns: quantized-dequantized weight tensor
        # Must respect self.num_bits and self.group_size
        # Optional: set self.input_scale (see Constraints)
        return W_dq

    def free(self):
        # Release calibration buffers
        pass
```

Constraints:

- You must NOT retrain or fine-tune the model (no gradient updates to
  original weights)
- All linear layers in each transformer block are quantized (`q_proj`,
  `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- Embeddings, LayerNorm, and the LM head are NOT quantized
- The returned weight must have the same shape and dtype as the original
- The returned weight must be a genuine `num_bits` group quantization, and
  the fixed code checks it: in every output row, each group of `group_size`
  consecutive input columns must lie on one uniform grid `a + delta * k`
  with integer `k` in `[0, 2**num_bits - 1]`, i.e. at most `2**num_bits`
  levels and one scale and zero-point per group (symmetric or asymmetric).
  A per-input-channel scaling folded into the layer input (the AWQ /
  SmoothQuant form `Q(W * s) / s`) is allowed if `quantize()` sets
  `self.input_scale = s`, a positive tensor of shape `(in_features,)`; the
  grid must then hold for `W_dq * s`. A weight that violates this aborts the
  run with no score. Entries of an accepted weight more than two fp16 ulps
  off its fitted grid are snapped onto it, and the model is re-checked
  before evaluation, so weights must not be changed after `quantize()`
  returns (no forward hooks or replaced `nn.Linear` modules either)
- `copy`, `math`, `torch`, `torch.nn`, `F`, `np`, `os`, `time` are
  available
- Your algorithm must work for both INT4 and INT3, and for different
  group sizes

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/gptq/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `gptq/custom_ptq.py`
- editable lines **26–157**




## Readable Context


### `gptq/custom_ptq.py`  [EDITABLE — lines 26–157 only]

```python
     1: """Post-Training Quantization (PTQ) for LLMs -- quantize + evaluate pipeline.
     2: 
     3: This script loads a pretrained LLM (Mistral-7B-v0.1), applies INT4 weight
     4: quantization using a custom algorithm, and evaluates perplexity on WikiText-2.
     5: 
     6: The quantization algorithm is defined in the EDITABLE REGION below.
     7: Everything else (model loading, calibration data, evaluation) is fixed.
     8: """
     9: 
    10: import argparse
    11: import math
    12: import os
    13: import time
    14: 
    15: import numpy as np
    16: import torch
    17: import torch.nn as nn
    18: import torch.nn.functional as F
    19: 
    20: from transformers import AutoModelForCausalLM, AutoTokenizer
    21: 
    22: 
    23: # ═══════════════════════════════════════════════════════════════════════════════
    24: # EDITABLE REGION START -- Quantization Algorithm (lines 26-157)
    25: # ═══════════════════════════════════════════════════════════════════════════════
    26: 
    27: # ── Helper: basic quantize/dequantize primitives ──────────────────────────────
    28: 
    29: def quantize_tensor(x, scale, zero_point, qmin, qmax):
    30:     """Quantize a float tensor to integers given scale and zero point."""
    31:     x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    32:     return x_int
    33: 
    34: 
    35: def dequantize_tensor(x_int, scale, zero_point):
    36:     """Dequantize integer tensor back to float."""
    37:     return (x_int - zero_point) * scale
    38: 
    39: 
    40: def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    41:     """Compute per-channel (or per-group) quantization parameters.
    42: 
    43:     Args:
    44:         weight: float tensor of shape (out_features, in_features)
    45:         num_bits: number of quantization bits
    46:         group_size: if > 0, compute params per group of columns; else per-row
    47:         symmetric: if True, use symmetric quantization (zero_point = 0)
    48: 
    49:     Returns:
    50:         scale: float tensor broadcastable to weight shape
    51:         zero_point: float tensor broadcastable to weight shape
    52:         qmin, qmax: integer quantization range
    53:     """
    54:     qmin = -(1 << (num_bits - 1))
    55:     qmax = (1 << (num_bits - 1)) - 1
    56: 
    57:     if group_size > 0:
    58:         # Reshape weight into groups for per-group quantization
    59:         out_features, in_features = weight.shape
    60:         assert in_features % group_size == 0, \
    61:             f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    62:         w_groups = weight.reshape(out_features, -1, group_size)
    63: 
    64:         if symmetric:
    65:             w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    66:             scale = w_max / qmax
    67:             zero_point = torch.zeros_like(scale)
    68:         else:
    69:             w_min = w_groups.amin(dim=-1, keepdim=True)
    70:             w_max = w_groups.amax(dim=-1, keepdim=True)
    71:             w_range = (w_max - w_min).clamp(min=1e-12)
    72:             scale = w_range / (qmax - qmin)
    73:             zero_point = torch.round(qmin - w_min / scale)
    74: 
    75:         scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    76:         zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    77:     else:
    78:         # Per-channel (per output row)
    79:         if symmetric:
    80:             w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    81:             scale = w_max / qmax
    82:             zero_point = torch.zeros_like(scale)
    83:         else:
    84:             w_min = weight.amin(dim=1, keepdim=True)
    85:             w_max = weight.amax(dim=1, keepdim=True)
    86:             w_range = (w_max - w_min).clamp(min=1e-12)
    87:             scale = w_range / (qmax - qmin)
    88:             zero_point = torch.round(qmin - w_min / scale)
    89: 
    90:     return scale, zero_point, qmin, qmax
    91: 
    92: 
    93: class LayerQuantizer:
    94:     """Quantizes a single nn.Linear layer's weights to low-bit integers.
    95: 
    96:     This class encapsulates the quantization algorithm. Override the
    97:     `quantize` method to implement custom quantization strategies.
    98: 
    99:     The calibration data (layer inputs) is provided via `add_batch()`.
   100:     The `quantize()` method uses the collected statistics to quantize
   101:     the weight matrix and returns the quantized-dequantized weight.
   102: 
   103:     Args:
   104:         layer: nn.Linear module to quantize
   105:         num_bits: target bit width (default: 4)
   106:         group_size: quantization group size; -1 for per-channel (default: -1)
   107:     """
   108: 
   109:     def __init__(self, layer, num_bits=4, group_size=-1):
   110:         self.layer = layer
   111:         self.num_bits = num_bits
   112:         self.group_size = group_size
   113:         self.out_features, self.in_features = layer.weight.shape
   114:         self.dev = layer.weight.device
   115: 
   116:         # Accumulate Hessian (X^T X) for calibration
   117:         self.nsamples = 0
   118:         self.H = torch.zeros(
   119:             (self.in_features, self.in_features),
   120:             device=self.dev, dtype=torch.float32
   121:         )
   122: 
   123:     def add_batch(self, inp):
   124:         """Accumulate calibration statistics from a batch of layer inputs.
   125: 
   126:         Args:
   127:             inp: input tensor of shape (batch, seq_len, in_features) or
   128:                  (batch * seq_len, in_features)
   129:         """
   130:         if inp.dim() == 3:
   131:             inp = inp.reshape(-1, inp.shape[-1])
   132:         n = inp.shape[0]
   133:         inp = inp.float()
   134:         self.H += inp.T @ inp
   135:         self.nsamples += n
   136: 
   137:     def quantize(self):
   138:         """Quantize the layer weights and return the quantized-dequantized weight.
   139: 
   140:         Default implementation: simple round-to-nearest (RTN) quantization.
   141:         Override this to implement better algorithms (e.g., GPTQ, AWQ).
   142: 
   143:         Returns:
   144:             Quantized-dequantized weight tensor of same shape as original weight.
   145:         """
   146:         W = self.layer.weight.data.clone().float()
   147:         scale, zero_point, qmin, qmax = find_scale_zero(
   148:             W, num_bits=self.num_bits, group_size=self.group_size, symmetric=True
   149:         )
   150:         W_q = quantize_tensor(W, scale, zero_point, qmin, qmax)
   151:         W_dq = dequantize_tensor(W_q, scale, zero_point)
   152:         return W_dq.to(self.layer.weight.dtype)
   153: 
   154:     def free(self):
   155:         """Release calibration buffers."""
   156:         del self.H
   157:         self.H = None
   158: 
   159: 
   160: # ═══════════════════════════════════════════════════════════════════════════════
   161: # EDITABLE REGION END
   162: # ═══════════════════════════════════════════════════════════════════════════════
   163: 
   164: 
   165: # ── Quantization-format enforcement (fixed) ──────────────────────────────────
   166: #
   167: # `LayerQuantizer.quantize()` hands back a dequantized weight, so its return
   168: # type alone cannot stop it from returning a higher-precision matrix (returning
   169: # `W` unchanged would score the FP16 perplexity). Every returned weight is
   170: # therefore checked against the format this task scores:
   171: #
   172: #   * each output row is split into groups of `group_size` consecutive input
   173: #     columns (the whole row when group_size == -1);
   174: #   * within a group the values lie on ONE uniform grid  a + delta * k  with
   175: #     integer k in [0, 2**num_bits - 1]: at most 2**num_bits levels and one
   176: #     scale / zero-point per group (the symmetric grid is a special case);
   177: #   * a quantizer may also set `self.input_scale`, a positive vector of shape
   178: #     (in_features,): the grid must then hold for `W_hat * input_scale`. This
   179: #     is the AWQ / SmoothQuant form  W_hat = Q(W * s) / s,  whose per-channel
   180: #     `s` is folded into the layer input at inference.
   181: #
   182: # A group that is off every such grid by more than _GRID_TOL grid steps raises
   183: # QuantizationFormatError and the run reports no metrics. An accepted weight is
   184: # snapped to its fitted grid (a + delta * k, then / s): an entry more than two
   185: # fp16 ulps from its grid point is replaced by the grid point, so at most two
   186: # ulps of freedom survive off the grid. An honest quantizer's own fp16 rounding
   187: # stays within that, so its weights pass through bit for bit. Each written weight is fingerprinted and
   188: # the whole model is re-checked before evaluation, so a weight cannot be
   189: # swapped back after it was checked.
   190: 
   191: _GRID_TOL = 0.1  # grid steps; fp16 rounding of an honest grid is ~0.02
   192: 
   193: 
   194: class QuantizationFormatError(RuntimeError):
   195:     """A quantizer produced a weight that is not a valid num_bits quantization."""
   196: 
   197: 
   198: def _read_input_scale(quantizer, in_features, device, name):
   199:     s = getattr(quantizer, "input_scale", None)
   200:     if s is None:
   201:         return None
   202:     if not torch.is_tensor(s):
   203:         raise QuantizationFormatError(
   204:             f"{name}: input_scale must be a tensor, got {type(s).__name__}")
   205:     s = s.detach().reshape(-1).to(device=device, dtype=torch.float32).clone()
   206:     if s.numel() != in_features:
   207:         raise QuantizationFormatError(
   208:             f"{name}: input_scale has {s.numel()} entries; expected {in_features}")
   209:     if not bool(torch.isfinite(s).all()) or bool((s <= 0).any()):
   210:         raise QuantizationFormatError(
   211:             f"{name}: input_scale must be finite and strictly positive")
   212:     return s
   213: 
   214: 
   215: @torch.no_grad()
   216: def _project_to_grid(W_ret, W_ref, num_bits, group_size, s, name):
   217:     """Check W_ret is a num_bits group quantization; return it rebuilt on its grid."""
   218:     if not torch.is_tensor(W_ret):
   219:         raise QuantizationFormatError(
   220:             f"{name}: quantize() must return a tensor, got {type(W_ret).__name__}")
   221:     if tuple(W_ret.shape) != tuple(W_ref.shape) or W_ret.dtype != W_ref.dtype:
   222:         raise QuantizationFormatError(
   223:             f"{name}: quantize() returned {tuple(W_ret.shape)} {W_ret.dtype}; "
   224:             f"expected {tuple(W_ref.shape)} {W_ref.dtype}")
   225:     V = W_ret.detach().to(device=W_ref.device, dtype=torch.float32)
   226:     if not bool(torch.isfinite(V).all()):
   227:         raise QuantizationFormatError(f"{name}: quantized weight is not finite")
   228:     out_f, in_f = V.shape
   229:     gs = group_size if group_size > 0 else in_f
   230:     if in_f % gs != 0:
   231:         raise QuantizationFormatError(
   232:             f"{name}: in_features {in_f} is not divisible by group_size {gs}")
   233:     if s is not None:
   234:         V = V * s.unsqueeze(0)
   235:     V = V.reshape(out_f, in_f // gs, gs)
   236:     vmin = V.amin(dim=-1, keepdim=True)
   237:     span = V.amax(dim=-1, keepdim=True) - vmin
   238:     flat = span <= 0
   239:     u = (V - vmin) / torch.where(flat, torch.ones_like(span), span)
   240:     # Pick, per group, the number of grid steps m (<= 2**num_bits - 1) spanning
   241:     # the group whose grid the values sit on most closely (deviation measured
   242:     # in units of the group span, so a coarse grid cannot fit by accident),
   243:     # then require that deviation to be at most _GRID_TOL of one grid step.
   244:     best_dev = torch.full_like(span, float("inf"))
   245:     best_m = torch.ones_like(span)
   246:     for m in range(1, 1 << num_bits):
   247:         um = u * m
   248:         dev_m = (um - torch.round(um)).abs().amax(dim=-1, keepdim=True)
   249:         better = dev_m / m < best_dev / best_m
   250:         best_dev = torch.where(better, dev_m, best_dev)
   251:         best_m = torch.where(better, torch.full_like(best_m, m), best_m)
   252:         del um, dev_m, better
   253:     bad = (best_dev > _GRID_TOL) & ~flat
   254:     if bool(bad.any()):
   255:         n_bad = int(bad.sum().item())
   256:         r, g = [int(x) for x in bad.squeeze(-1).nonzero()[0].tolist()]
   257:         n_lv = int(torch.unique(V[r, g]).numel())
   258:         raise QuantizationFormatError(
   259:             f"{name}: {n_bad} of {bad.numel()} (row, group) blocks are not a "
   260:             f"{num_bits}-bit quantization with group_size={group_size}: e.g. row "
   261:             f"{r}, group {g} has {n_lv} distinct values that do not fit one "
   262:             f"uniform grid of at most {1 << num_bits} levels")
   263:     k = torch.where(flat, torch.zeros_like(u), torch.round(u * best_m))
   264:     del u
   265:     # Fit the grid in fp32, anchored at the level nearest zero (so small grid
   266:     # points are not lost to cancellation):  v = v0 + delta * (k - k0),  with
   267:     # v0 the mean value at level k0 and delta the least-squares step.
   268:     k0 = k.gather(-1, V.abs().argmin(dim=-1, keepdim=True))
   269:     at0 = (k == k0).float()
   270:     v0 = (V * at0).sum(dim=-1, keepdim=True) / at0.sum(dim=-1, keepdim=True)
   271:     kd = k - k0
   272:     denom = (kd * kd).sum(dim=-1, keepdim=True)
   273:     delta = torch.where(
   274:         denom > 0,
   275:         (kd * (V - v0)).sum(dim=-1, keepdim=True) / denom.clamp(min=1e-30),
   276:         torch.zeros_like(denom))
   277:     W_grid = (v0 + delta * kd).reshape(out_f, in_f)
   278:     del at0, kd
   279:     del k, V
   280:     if s is not None:
   281:         W_grid = W_grid / s.unsqueeze(0)
   282:     # A returned value within two fp16 ulps of its fitted grid point is that
   283:     # grid point as rounded by the quantizer's own arithmetic (plus the fit's
   284:     # own fp32 error), and is kept bit for bit; anything further off is
   285:     # replaced by the grid point itself.
   286:     _, e = torch.frexp(W_grid)
   287:     ulp = torch.where(
   288:         W_grid == 0, torch.full_like(W_grid, 2.0 ** -24),
   289:         torch.ldexp(torch.ones_like(W_grid), e - 11).clamp(min=2.0 ** -24))
   290:     W_ret32 = W_ret.detach().to(device=W_ref.device, dtype=torch.float32)
   291:     keep = (W_ret32 - W_grid).abs() <= 2 * ulp
   292:     del ulp, e, W_ret32
   293:     return torch.where(
   294:         keep, W_ret.detach().to(W_ref.device), W_grid.to(W_ref.dtype))
   295: 
   296: 
   297: def _fingerprint(lin):
   298:     import hashlib
   299:     h = hashlib.blake2b(digest_size=16)
   300:     for t in (lin.weight, lin.bias):
   301:         if t is None:
   302:             h.update(b"none")
   303:             continue
   304:         t = t.detach()
   305:         h.update(repr((tuple(t.shape), str(t.dtype))).encode())
   306:         h.update(t.contiguous().cpu().reshape(-1).view(torch.uint8).numpy())
   307:     return h.hexdigest()
   308: 
   309: 
   310: def _check_linear_intact(lin, key):
   311:     if type(lin) is not nn.Linear or "forward" in vars(lin):
   312:         raise QuantizationFormatError(f"{key}: the linear layer was replaced")
   313:     if lin._forward_hooks or lin._forward_pre_hooks:
   314:         raise QuantizationFormatError(f"{key}: a forward hook is attached")
   315: 
   316: 
   317: def _verify_quantized_model(layers, fingerprints):
   318:     """Re-check every quantized linear against the fingerprint taken when written."""
   319:     from torch.nn.modules import module as _module
   320:     if _module._global_forward_hooks or _module._global_forward_pre_hooks:
   321:         raise QuantizationFormatError("a global module forward hook is registered")
   322:     seen = set()
   323:     for i in range(len(layers)):
   324:         for name, lin in find_linear_layers(layers[i]).items():
   325:             key = f"layers.{i}.{name}"
   326:             if key not in fingerprints:
   327:                 raise QuantizationFormatError(f"{key}: linear layer was never quantized")
   328:             _check_linear_intact(lin, key)
   329:             if _fingerprint(lin) != fingerprints[key]:
   330:                 raise QuantizationFormatError(
   331:                     f"{key}: weight changed after it was quantized and checked")
   332:             seen.add(key)
   333:     missing = set(fingerprints) - seen
   334:     if missing:
   335:         raise QuantizationFormatError(
   336:             f"quantized layers disappeared: {sorted(missing)[:3]}")
   337: 
   338: 
   339: # ── Model loading ─────────────────────────────────────────────────────────────
   340: 
   341: def get_model(model_path):
   342:     """Load a pretrained causal LM with weight initialization skipped."""
   343:     def skip(*args, **kwargs):
   344:         pass
   345:     torch.nn.init.kaiming_uniform_ = skip
   346:     torch.nn.init.uniform_ = skip
   347:     torch.nn.init.normal_ = skip
   348: 
   349:     model = AutoModelForCausalLM.from_pretrained(
   350:         model_path, torch_dtype=torch.float16, device_map="cpu"
   351:     )
   352:     model.seqlen = min(getattr(model.config, "max_position_embeddings", 4096), 4096)
   353:     model.eval()
   354:     return model
   355: 
   356: 
   357: def find_linear_layers(module, prefix=""):
   358:     """Recursively find all nn.Linear layers in the model."""
   359:     result = {}
   360:     for name, child in module.named_children():
   361:         full_name = f"{prefix}.{name}" if prefix else name
   362:         if isinstance(child, nn.Linear):
   363:             result[full_name] = child
   364:         else:
   365:             result.update(find_linear_layers(child, full_name))
   366:     return result
   367: 
   368: 
   369: # ── Calibration data ──────────────────────────────────────────────────────────
   370: 
   371: def get_calibration_data(tokenizer, nsamples=128, seqlen=2048, seed=0):
   372:     """Load WikiText-2 calibration data."""
   373:     from datasets import load_dataset
   374: 
   375:     # Load from pre-downloaded cache (compute nodes have no network)
   376:     cache_dir = os.environ.get("HF_DATASETS_CACHE", "/data/wikitext2")
   377:     try:
   378:         traindata = load_dataset(
   379:             "wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir
   380:         )
   381:     except Exception:
   382:         # Fallback: load directly from arrow files in cache
   383:         from datasets import Dataset
   384:         import glob
   385:         arrow = glob.glob(f"{cache_dir}/**/wikitext-train.arrow", recursive=True)
   386:         if arrow:
   387:             traindata = Dataset.from_file(arrow[0])
   388:         else:
   389:             raise FileNotFoundError(f"WikiText-2 train data not found in {cache_dir}")
   390: 
   391:     import random
   392:     random.seed(seed)
   393: 
   394:     trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
   395: 
   396:     trainloader = []
   397:     for _ in range(nsamples):
   398:         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
   399:         j = i + seqlen
   400:         inp = trainenc.input_ids[:, i:j]
   401:         trainloader.append(inp)
   402: 
   403:     return trainloader
   404: 
   405: 
   406: def get_eval_data(tokenizer, seqlen=2048):
   407:     """Load WikiText-2 test data for perplexity evaluation."""
   408:     from datasets import load_dataset
   409: 
   410:     cache_dir = os.environ.get("HF_DATASETS_CACHE", "/data/wikitext2")
   411:     try:
   412:         testdata = load_dataset(
   413:             "wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir
   414:         )
   415:     except Exception:
   416:         from datasets import Dataset
   417:         import glob
   418:         arrow = glob.glob(f"{cache_dir}/**/wikitext-test.arrow", recursive=True)
   419:         if arrow:
   420:             testdata = Dataset.from_file(arrow[0])
   421:         else:
   422:             raise FileNotFoundError(f"WikiText-2 test data not found in {cache_dir}")
   423: 
   424:     testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
   425:     return testenc
   426: 
   427: 
   428: # ── Layer-by-layer quantization ───────────────────────────────────────────────
   429: 
   430: @torch.no_grad()
   431: def quantize_model(model, calibration_data, dev, num_bits=4, group_size=-1):
   432:     """Quantize all linear layers in the model using LayerQuantizer.
   433: 
   434:     Processes the model layer-by-layer (transformer block by block) to
   435:     minimize GPU memory usage. For each block:
   436:       1. Move block to GPU
   437:       2. Run calibration data through to collect Hessian statistics
   438:       3. Quantize each linear sublayer using LayerQuantizer
   439:       4. Replace weights with quantized-dequantized values
   440:       5. Move block back to CPU
   441: 
   442:     Args:
   443:         model: pretrained causal LM
   444:         calibration_data: list of input_ids tensors for calibration
   445:         dev: torch device (GPU)
   446:         num_bits: target bit width
   447:         group_size: quantization group size; -1 for per-channel
   448: 
   449:     Returns:
   450:         dict mapping layer name -> quantization error (Frobenius norm)
   451:     """
   452:     print("Starting quantization...", flush=True)
   453:     use_cache = model.config.use_cache
   454:     model.config.use_cache = False
   455: 
   456:     layers = model.model.layers
   457:     model.model.embed_tokens = model.model.embed_tokens.to(dev)
   458:     if hasattr(model.model, "rotary_emb"):
   459:         model.model.rotary_emb = model.model.rotary_emb.to(dev)
   460:     layers[0] = layers[0].to(dev)
   461: 
   462:     dtype = next(iter(model.parameters())).dtype
   463:     nsamples = len(calibration_data)
   464:     seqlen = calibration_data[0].shape[1]
   465:     hidden_size = model.config.hidden_size
   466: 
   467:     # Capture inputs to first layer
   468:     inps = torch.zeros(
   469:         (nsamples, seqlen, hidden_size), dtype=dtype, device=dev
   470:     )
   471:     cache = {"i": 0, "attention_mask": None, "position_ids": None}
   472: 
   473:     class Catcher(nn.Module):
   474:         def __init__(self, module):
   475:             super().__init__()
   476:             self.module = module
   477:         def forward(self, inp, **kwargs):
   478:             inps[cache["i"]] = inp
   479:             cache["i"] += 1
   480:             cache["attention_mask"] = kwargs.get("attention_mask")
   481:             cache["position_ids"] = kwargs.get("position_ids")
   482:             cache["position_embeddings"] = kwargs.get("position_embeddings")
   483:             raise ValueError
   484: 
   485:     layers[0] = Catcher(layers[0])
   486:     for batch in calibration_data:
   487:         try:
   488:             model(batch.to(dev))
   489:         except ValueError:
   490:             pass
   491:     layers[0] = layers[0].module
   492: 
   493:     layers[0] = layers[0].cpu()
   494:     model.model.embed_tokens = model.model.embed_tokens.cpu()
   495:     if hasattr(model.model, "rotary_emb"):
   496:         model.model.rotary_emb = model.model.rotary_emb.cpu()
   497:     torch.cuda.empty_cache()
   498: 
   499:     outs = torch.zeros_like(inps)
   500:     attention_mask = cache["attention_mask"]

[truncated: showing at most 500 lines / 60000 bytes from gptq/custom_ptq.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `round_to_nearest` baseline — editable region  [READ-ONLY — reference implementation]

In `gptq/custom_ptq.py`:

```python
Lines 26–111:
    23: # ═══════════════════════════════════════════════════════════════════════════════
    24: # EDITABLE REGION START -- Quantization Algorithm (lines 26-157)
    25: # ═══════════════════════════════════════════════════════════════════════════════
    26: 
    27: # ── Helper: basic quantize/dequantize primitives ──────────────────────────────
    28: 
    29: def quantize_tensor(x, scale, zero_point, qmin, qmax):
    30:     """Quantize a float tensor to integers given scale and zero point."""
    31:     x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    32:     return x_int
    33: 
    34: 
    35: def dequantize_tensor(x_int, scale, zero_point):
    36:     """Dequantize integer tensor back to float."""
    37:     return (x_int - zero_point) * scale
    38: 
    39: 
    40: def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    41:     """Compute per-channel (or per-group) quantization parameters."""
    42:     qmin = -(1 << (num_bits - 1))
    43:     qmax = (1 << (num_bits - 1)) - 1
    44: 
    45:     if group_size > 0:
    46:         out_features, in_features = weight.shape
    47:         assert in_features % group_size == 0
    48:         w_groups = weight.reshape(out_features, -1, group_size)
    49:         if symmetric:
    50:             w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    51:             scale = w_max / qmax
    52:             zero_point = torch.zeros_like(scale)
    53:         else:
    54:             w_min = w_groups.amin(dim=-1, keepdim=True)
    55:             w_max = w_groups.amax(dim=-1, keepdim=True)
    56:             w_range = (w_max - w_min).clamp(min=1e-12)
    57:             scale = w_range / (qmax - qmin)
    58:             zero_point = torch.round(qmin - w_min / scale)
    59:         scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    60:         zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    61:     else:
    62:         if symmetric:
    63:             w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    64:             scale = w_max / qmax
    65:             zero_point = torch.zeros_like(scale)
    66:         else:
    67:             w_min = weight.amin(dim=1, keepdim=True)
    68:             w_max = weight.amax(dim=1, keepdim=True)
    69:             w_range = (w_max - w_min).clamp(min=1e-12)
    70:             scale = w_range / (qmax - qmin)
    71:             zero_point = torch.round(qmin - w_min / scale)
    72: 
    73:     return scale, zero_point, qmin, qmax
    74: 
    75: 
    76: class LayerQuantizer:
    77:     """RTN quantizer -- simple round-to-nearest, ignores calibration data."""
    78: 
    79:     def __init__(self, layer, num_bits=4, group_size=-1):
    80:         self.layer = layer
    81:         self.num_bits = num_bits
    82:         self.group_size = group_size
    83:         self.out_features, self.in_features = layer.weight.shape
    84:         self.dev = layer.weight.device
    85:         self.nsamples = 0
    86:         self.H = torch.zeros(
    87:             (self.in_features, self.in_features),
    88:             device=self.dev, dtype=torch.float32
    89:         )
    90: 
    91:     def add_batch(self, inp):
    92:         """Collect calibration data (unused in RTN, kept for interface)."""
    93:         if inp.dim() == 3:
    94:             inp = inp.reshape(-1, inp.shape[-1])
    95:         self.nsamples += inp.shape[0]
    96: 
    97:     def quantize(self):
    98:         """RTN: symmetric per-channel (or per-group) round-to-nearest."""
    99:         W = self.layer.weight.data.clone().float()
   100:         scale, zero_point, qmin, qmax = find_scale_zero(
   101:             W, num_bits=self.num_bits, group_size=self.group_size, symmetric=True
   102:         )
   103:         W_q = quantize_tensor(W, scale, zero_point, qmin, qmax)
   104:         W_dq = dequantize_tensor(W_q, scale, zero_point)
   105:         return W_dq.to(self.layer.weight.dtype)
   106: 
   107:     def free(self):
   108:         """Release calibration buffers."""
   109:         del self.H
   110:         self.H = None
   111: 
   112: 
   113: 
   114: # ═══════════════════════════════════════════════════════════════════════════════
```

### `gptq` baseline — editable region  [READ-ONLY — reference implementation]

In `gptq/custom_ptq.py`:

```python
Lines 26–192:
    23: # ═══════════════════════════════════════════════════════════════════════════════
    24: # EDITABLE REGION START -- Quantization Algorithm (lines 26-157)
    25: # ═══════════════════════════════════════════════════════════════════════════════
    26: 
    27: # ── Helper: basic quantize/dequantize primitives ──────────────────────────────
    28: 
    29: def quantize_tensor(x, scale, zero_point, qmin, qmax):
    30:     """Quantize a float tensor to integers given scale and zero point."""
    31:     x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    32:     return x_int
    33: 
    34: 
    35: def dequantize_tensor(x_int, scale, zero_point):
    36:     """Dequantize integer tensor back to float."""
    37:     return (x_int - zero_point) * scale
    38: 
    39: 
    40: def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    41:     """Compute per-channel (or per-group) quantization parameters."""
    42:     qmin = -(1 << (num_bits - 1))
    43:     qmax = (1 << (num_bits - 1)) - 1
    44: 
    45:     if group_size > 0:
    46:         out_features, in_features = weight.shape
    47:         assert in_features % group_size == 0
    48:         w_groups = weight.reshape(out_features, -1, group_size)
    49:         if symmetric:
    50:             w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    51:             scale = w_max / qmax
    52:             zero_point = torch.zeros_like(scale)
    53:         else:
    54:             w_min = w_groups.amin(dim=-1, keepdim=True)
    55:             w_max = w_groups.amax(dim=-1, keepdim=True)
    56:             w_range = (w_max - w_min).clamp(min=1e-12)
    57:             scale = w_range / (qmax - qmin)
    58:             zero_point = torch.round(qmin - w_min / scale)
    59:         scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    60:         zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    61:     else:
    62:         if symmetric:
    63:             w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    64:             scale = w_max / qmax
    65:             zero_point = torch.zeros_like(scale)
    66:         else:
    67:             w_min = weight.amin(dim=1, keepdim=True)
    68:             w_max = weight.amax(dim=1, keepdim=True)
    69:             w_range = (w_max - w_min).clamp(min=1e-12)
    70:             scale = w_range / (qmax - qmin)
    71:             zero_point = torch.round(qmin - w_min / scale)
    72: 
    73:     return scale, zero_point, qmin, qmax
    74: 
    75: 
    76: class LayerQuantizer:
    77:     """GPTQ quantizer -- Hessian-based error compensation.
    78: 
    79:     Collects input activation statistics (H = X^T X), then quantizes
    80:     weights column-by-column, compensating for quantization error using
    81:     the Hessian inverse so that layer output error is minimized.
    82:     """
    83: 
    84:     BLOCK_SIZE = 128
    85:     PERCDAMP = 0.01
    86: 
    87:     def __init__(self, layer, num_bits=4, group_size=-1):
    88:         self.layer = layer
    89:         self.num_bits = num_bits
    90:         self.group_size = group_size
    91:         self.out_features, self.in_features = layer.weight.shape
    92:         self.dev = layer.weight.device
    93:         self.nsamples = 0
    94:         self.H = torch.zeros(
    95:             (self.in_features, self.in_features),
    96:             device=self.dev, dtype=torch.float32
    97:         )
    98: 
    99:     def add_batch(self, inp):
   100:         """Accumulate Hessian approximation from calibration inputs."""
   101:         if inp.dim() == 3:
   102:             inp = inp.reshape(-1, inp.shape[-1])
   103:         n = inp.shape[0]
   104:         inp = inp.float()
   105:         self.H += inp.T @ inp
   106:         self.nsamples += n
   107: 
   108:     def quantize(self):
   109:         """GPTQ: column-by-column quantization with Hessian error compensation."""
   110:         W = self.layer.weight.data.clone().float()
   111:         H = self.H.clone()
   112: 
   113:         if self.nsamples > 0:
   114:             H /= self.nsamples
   115: 
   116:         num_bits = self.num_bits
   117:         group_size = self.group_size
   118:         qmin = -(1 << (num_bits - 1))
   119:         qmax = (1 << (num_bits - 1)) - 1
   120: 
   121:         # Add dampening to diagonal for numerical stability
   122:         damp = self.PERCDAMP * torch.mean(torch.diag(H))
   123:         H += damp * torch.eye(self.in_features, device=self.dev)
   124: 
   125:         # Compute Hessian inverse via Cholesky decomposition
   126:         try:
   127:             L = torch.linalg.cholesky(H)
   128:             Hinv = torch.cholesky_inverse(L)
   129:         except Exception:
   130:             # Fallback to pseudo-inverse if Cholesky fails
   131:             Hinv = torch.linalg.pinv(H)
   132: 
   133:         Q = torch.zeros_like(W)
   134:         Err = torch.zeros_like(W)
   135: 
   136:         # Process columns in blocks
   137:         for col_start in range(0, self.in_features, self.BLOCK_SIZE):
   138:             col_end = min(col_start + self.BLOCK_SIZE, self.in_features)
   139: 
   140:             W_block = W[:, col_start:col_end].clone()
   141:             Hinv_block_diag = torch.diag(
   142:                 Hinv[col_start:col_end, col_start:col_end]
   143:             )
   144: 
   145:             for j in range(col_end - col_start):
   146:                 col = col_start + j
   147:                 w_col = W_block[:, j]
   148: 
   149:                 # Compute scale: per-group if group_size > 0, else per-column
   150:                 if group_size > 0 and col % group_size == 0:
   151:                     g_end = min(col + group_size, self.in_features)
   152:                     W_group = W[:, col:g_end]
   153:                     g_max = W_group.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
   154:                     group_scale = (g_max / qmax).squeeze(1)
   155: 
   156:                 if group_size > 0:
   157:                     scale = group_scale
   158:                 else:
   159:                     w_abs_max = w_col.abs().max().clamp(min=1e-12)
   160:                     scale = w_abs_max / qmax
   161: 
   162:                 # Quantize and dequantize
   163:                 q_col = torch.clamp(
   164:                     torch.round(w_col / scale), qmin, qmax
   165:                 ) * scale
   166:                 Q[:, col] = q_col
   167: 
   168:                 # Error compensation: distribute error weighted by Hessian
   169:                 err = (w_col - q_col) / Hinv_block_diag[j].clamp(min=1e-12)
   170:                 Err[:, col] = err
   171: 
   172:                 # Update remaining columns in block
   173:                 if j + 1 < col_end - col_start:
   174:                     W_block[:, j+1:] -= (
   175:                         err.unsqueeze(1)
   176:                         * Hinv[col, col_start+j+1:col_end].unsqueeze(0)
   177:                     )
   178: 
   179:             # Propagate error to remaining columns outside block
   180:             if col_end < self.in_features:
   181:                 W[:, col_end:] -= (
   182:                     Err[:, col_start:col_end]
   183:                     @ Hinv[col_start:col_end, col_end:]
   184:                 )
   185: 
   186:         return Q.to(self.layer.weight.dtype)
   187: 
   188:     def free(self):
   189:         """Release calibration buffers."""
   190:         del self.H
   191:         self.H = None
   192: 
   193: 
   194: 
   195: # ═══════════════════════════════════════════════════════════════════════════════
```

### `awq` baseline — editable region  [READ-ONLY — reference implementation]

In `gptq/custom_ptq.py`:

```python
Lines 26–275:
    23: # ═══════════════════════════════════════════════════════════════════════════════
    24: # EDITABLE REGION START -- Quantization Algorithm (lines 26-157)
    25: # ═══════════════════════════════════════════════════════════════════════════════
    26: 
    27: # ── Helper: basic quantize/dequantize primitives ──────────────────────────────
    28: 
    29: def quantize_tensor(x, scale, zero_point, qmin, qmax):
    30:     """Quantize a float tensor to integers given scale and zero point."""
    31:     x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    32:     return x_int
    33: 
    34: 
    35: def dequantize_tensor(x_int, scale, zero_point):
    36:     """Dequantize integer tensor back to float."""
    37:     return (x_int - zero_point) * scale
    38: 
    39: 
    40: def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    41:     """Compute per-channel (or per-group) quantization parameters."""
    42:     qmin = -(1 << (num_bits - 1))
    43:     qmax = (1 << (num_bits - 1)) - 1
    44: 
    45:     if group_size > 0:
    46:         out_features, in_features = weight.shape
    47:         assert in_features % group_size == 0
    48:         w_groups = weight.reshape(out_features, -1, group_size)
    49:         if symmetric:
    50:             w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    51:             scale = w_max / qmax
    52:             zero_point = torch.zeros_like(scale)
    53:         else:
    54:             w_min = w_groups.amin(dim=-1, keepdim=True)
    55:             w_max = w_groups.amax(dim=-1, keepdim=True)
    56:             w_range = (w_max - w_min).clamp(min=1e-12)
    57:             scale = w_range / (qmax - qmin)
    58:             zero_point = torch.round(qmin - w_min / scale)
    59:         scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    60:         zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    61:     else:
    62:         if symmetric:
    63:             w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    64:             scale = w_max / qmax
    65:             zero_point = torch.zeros_like(scale)
    66:         else:
    67:             w_min = weight.amin(dim=1, keepdim=True)
    68:             w_max = weight.amax(dim=1, keepdim=True)
    69:             w_range = (w_max - w_min).clamp(min=1e-12)
    70:             scale = w_range / (qmax - qmin)
    71:             zero_point = torch.round(qmin - w_min / scale)
    72: 
    73:     return scale, zero_point, qmin, qmax
    74: 
    75: 
    76: class LayerQuantizer:
    77:     """AWQ quantizer -- faithful to mit-han-lab/llm-awq.
    78: 
    79:     Pipeline:
    80:       1. add_batch: accumulate per-channel mean |X|; reservoir-sample raw input
    81:          tokens (up to N_SAMPLE_TOKEN rows) so we can use real activations as
    82:          the loss signal during search.
    83:       2. Per-channel scale alpha-search (auto_scale): for ratio in [0, 1):
    84:              s = x_max^ratio  (clamped, range-normalized: s /= sqrt(max*min))
    85:          loss = mean((X @ (W - W_final).T)^2)  on sampled X.
    86:       3. Per-group max clip-search (auto_clip), on the post-scale weights:
    87:          clip per-group max by 1 - i/N for i in 0..MAX_SHRINK*N, loss is
    88:          per-(out_channel, group) output-error using sampled X:
    89:              org_out[r, t, g] = sum_c W_scaled[r,g,c] * X[t,g,c]
    90:              cur_out[r, t, g] = sum_c Q(clamp(W,±M))[r,g,c] * X[t,g,c]
    91:              err[r, g] = mean_t (cur_out - org_out)^2
    92:       4. Quantize with the clipped per-group scales, undo channel scaling.
    93: 
    94:     Implemented to fit the LayerQuantizer interface (per-linear, no block ctx),
    95:     so the loss is computed at linear-layer granularity (not full block).
    96:     """
    97: 
    98:     N_ALPHA = 20             # auto_scale grid size
    99:     N_CLIP_GRID = 20         # auto_clip n_grid
   100:     CLIP_MAX_SHRINK = 0.5    # auto_clip max_shrink (official default)
   101:     N_SAMPLE_TOKEN = 256     # number of input tokens kept for loss computation
   102:     OC_BATCH = 256           # output-channel batching for clip search (memory)
   103: 
   104:     def __init__(self, layer, num_bits=4, group_size=-1):
   105:         self.layer = layer
   106:         self.num_bits = num_bits
   107:         self.group_size = group_size
   108:         self.out_features, self.in_features = layer.weight.shape
   109:         self.dev = layer.weight.device
   110:         self.nsamples = 0
   111: 
   112:         # Per-channel sum of |activation| (averaged over tokens at quantize-time)
   113:         self.act_sum = torch.zeros(
   114:             self.in_features, device=self.dev, dtype=torch.float32
   115:         )
   116:         # Reservoir of input tokens (CPU to save GPU memory across layers)
   117:         self._x_buf = []
   118:         self._x_buf_rows = 0
   119:         # Keep H for interface compatibility (unused by AWQ)
   120:         self.H = torch.zeros(
   121:             (self.in_features, self.in_features),
   122:             device=self.dev, dtype=torch.float32
   123:         )
   124: 
   125:     def add_batch(self, inp):
   126:         """Accumulate per-channel |X| stats and reservoir-sample raw inputs."""
   127:         if inp.dim() == 3:
   128:             inp = inp.reshape(-1, inp.shape[-1])
   129:         inp_f = inp.float()
   130:         n = inp_f.shape[0]
   131:         self.act_sum += inp_f.abs().sum(dim=0)
   132:         self.nsamples += n
   133:         # Keep ~4x N_SAMPLE_TOKEN candidate rows; we'll stride-sample at quantize.
   134:         cap = self.N_SAMPLE_TOKEN * 4
   135:         if self._x_buf_rows < cap:
   136:             take = min(n, cap - self._x_buf_rows)
   137:             # Take an evenly-spaced stride from this batch
   138:             stride = max(1, n // max(take, 1))
   139:             sampled = inp_f[::stride][:take].detach().to('cpu')
   140:             self._x_buf.append(sampled)
   141:             self._x_buf_rows += sampled.shape[0]
   142: 
   143:     def _get_x_samples(self):
   144:         if not self._x_buf:
   145:             return None
   146:         X = torch.cat(self._x_buf, dim=0)
   147:         if X.shape[0] > self.N_SAMPLE_TOKEN:
   148:             stride = X.shape[0] // self.N_SAMPLE_TOKEN
   149:             X = X[::stride][:self.N_SAMPLE_TOKEN]
   150:         return X.to(self.dev)
   151: 
   152:     def quantize(self):
   153:         """AWQ: per-channel scale search + per-group clip search + quantize."""
   154:         W = self.layer.weight.data.clone().float()
   155:         num_bits = self.num_bits
   156:         group_size = self.group_size
   157:         qmin = -(1 << (num_bits - 1))
   158:         qmax = (1 << (num_bits - 1)) - 1
   159: 
   160:         if self.nsamples > 0:
   161:             x_max = (self.act_sum / self.nsamples).clamp(min=1e-5)
   162:         else:
   163:             x_max = torch.ones(self.in_features, device=self.dev)
   164: 
   165:         X = self._get_x_samples()  # (T, in_features) on dev, may be None
   166: 
   167:         # ── (1) auto_scale: per-channel scale search ─────────────────────────
   168:         best_err = float('inf')
   169:         best_s = torch.ones(self.in_features, device=self.dev)
   170: 
   171:         for i in range(self.N_ALPHA):
   172:             ratio = i / self.N_ALPHA
   173:             s = x_max.pow(ratio).clamp(min=1e-4)
   174:             s = s / (s.max() * s.min()).sqrt().clamp(min=1e-5)
   175: 
   176:             W_scaled = W * s.unsqueeze(0)
   177:             scale_q, zp, _, _ = find_scale_zero(
   178:                 W_scaled, num_bits=num_bits, group_size=group_size, symmetric=True
   179:             )
   180:             W_q = quantize_tensor(W_scaled, scale_q, zp, qmin, qmax)
   181:             W_dq = dequantize_tensor(W_q, scale_q, zp)
   182:             W_final = W_dq / s.unsqueeze(0)
   183: 
   184:             if X is not None:
   185:                 # Output-error: ||X @ (W - W_final).T||^2 / (T * out)
   186:                 delta = (W - W_final).to(X.dtype)
   187:                 err = (X @ delta.T).pow(2).mean().item()
   188:             else:
   189:                 err = (W - W_final).pow(2).mul(x_max.unsqueeze(0).pow(2)).sum().item()
   190: 
   191:             if err < best_err:
   192:                 best_err = err
   193:                 best_s = s.clone()
   194: 
   195:         # Apply best per-channel scaling
   196:         W_scaled = W * best_s.unsqueeze(0)
   197: 
   198:         # ── (2) auto_clip: per-group max clip search ─────────────────────────
   199:         if group_size > 0:
   200:             n_groups = self.in_features // group_size
   201:             gs = group_size
   202:         else:
   203:             n_groups = 1
   204:             gs = self.in_features
   205: 
   206:         W_groups = W_scaled.reshape(self.out_features, n_groups, gs)  # (O, G, gs)
   207:         base_max = W_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
   208:         best_max = base_max.clone()
   209: 
   210:         if X is not None:
   211:             X_groups = X.reshape(X.shape[0], n_groups, gs)  # (T, G, gs)
   212: 
   213:             n_clip_iters = max(1, int(self.CLIP_MAX_SHRINK * self.N_CLIP_GRID))
   214:             oc_batch = self.OC_BATCH
   215:             if self.out_features % oc_batch != 0:
   216:                 # fall back to a divisor of out_features
   217:                 for cand in (128, 64, 32, 16, 8, 4, 2, 1):
   218:                     if self.out_features % cand == 0:
   219:                         oc_batch = cand
   220:                         break
   221: 
   222:             for i_b in range(0, self.out_features, oc_batch):
   223:                 W_b = W_groups[i_b:i_b + oc_batch]                # (B, G, gs)
   224:                 base_max_b = base_max[i_b:i_b + oc_batch]          # (B, G, 1)
   225:                 # org_out[r, t, g] = sum_c W_b[r,g,c] * X_groups[t,g,c]
   226:                 org_out = torch.einsum('rgc,tgc->rtg', W_b, X_groups.float())
   227:                 min_errs = torch.full_like(base_max_b, float('inf'))
   228:                 best_max_b = base_max_b.clone()
   229:                 for i_s in range(n_clip_iters):
   230:                     cur_max = base_max_b * (1 - i_s / self.N_CLIP_GRID)  # (B, G, 1)
   231:                     cur_w = torch.clamp(W_b, -cur_max, cur_max)
   232:                     scale_b = (cur_max / qmax).clamp(min=1e-12)
   233:                     q_w = (
   234:                         torch.clamp(torch.round(cur_w / scale_b), qmin, qmax) * scale_b
   235:                     )
   236:                     cur_out = torch.einsum('rgc,tgc->rtg', q_w, X_groups.float())
   237:                     err_b = (cur_out - org_out).pow(2).mean(dim=1, keepdim=True)
   238:                     err_b = err_b.permute(0, 2, 1).contiguous()  # (B, G, 1)
   239:                     mask = err_b < min_errs
   240:                     min_errs = torch.where(mask, err_b, min_errs)
   241:                     best_max_b = torch.where(mask, cur_max, best_max_b)
   242:                 best_max[i_b:i_b + oc_batch] = best_max_b
   243:                 del org_out, cur_out, q_w, cur_w
   244:             del X_groups
   245:         # else: no calibration samples — fall back to base_max (no clipping)
   246: 
   247:         # ── (3) Final quantization with clipped scales ───────────────────────
   248:         scale_g = (best_max / qmax).clamp(min=1e-12)
   249:         scale_q = scale_g.expand_as(W_groups).reshape(self.out_features, self.in_features)
   250:         zp = torch.zeros_like(scale_q)
   251: 
   252:         # Clamp scaled weights to the searched per-group range, then quantize
   253:         W_clamped = torch.clamp(
   254:             W_scaled,
   255:             -best_max.expand_as(W_groups).reshape(self.out_features, self.in_features),
   256:             best_max.expand_as(W_groups).reshape(self.out_features, self.in_features),
   257:         )
   258:         W_q = quantize_tensor(W_clamped, scale_q, zp, qmin, qmax)
   259:         W_dq = dequantize_tensor(W_q, scale_q, zp)
   260:         W_final = W_dq / best_s.unsqueeze(0)
   261: 
   262:         # W_final * best_s is on the per-group grid; declare the per-channel
   263:         # scale (folded into the layer input at inference) to the harness.
   264:         self.input_scale = best_s
   265:         return W_final.to(self.layer.weight.dtype)
   266: 
   267:     def free(self):
   268:         """Release calibration buffers."""
   269:         del self.H
   270:         del self.act_sum
   271:         del self._x_buf
   272:         self.H = None
   273:         self.act_sum = None
   274:         self._x_buf = None
   275: 
   276: 
   277: 
   278: # ═══════════════════════════════════════════════════════════════════════════════
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
