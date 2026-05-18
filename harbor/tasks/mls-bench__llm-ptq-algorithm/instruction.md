# MLS-Bench: llm-ptq-algorithm

# LLM Post-Training Quantization (PTQ) Algorithm

## Research Question

Design a post-training quantization algorithm that minimizes accuracy
degradation when quantizing a pretrained Mistral-7B-v0.1 model
(7.24B parameters) to low-bit integer precision, without any retraining
or fine-tuning.

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
  data (128 sequences)
- `LayerQuantizer.quantize()`: apply quantization to the layer's weight
  matrix

You can implement any approach: error compensation, weight transformation
(scaling, rotation, smoothing), mixed strategies, outlier handling, or
adaptive grouping schemes that vary by group size or bit-width.

## Architecture

The task loads real Mistral-7B-v0.1 weights (HuggingFace) and quantizes
them. No training is done — the task is purely about the quantization
algorithm quality.

Mistral-7B-v0.1 specs: 32 layers, 32 attention heads, 8 KV heads (GQA),
4096 hidden, 14336 intermediate, ~7.24B parameters.

The script (`custom_ptq.py`):

1. Loads Mistral-7B-v0.1 from `/data/mistral-7b-v01` (pre-downloaded
   HuggingFace snapshot)
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
- `copy`, `math`, `torch`, `torch.nn`, `F`, `np`, `os`, `time` are
  available
- Your algorithm must work for both INT4 and INT3, and for different
  group sizes

## Evaluation

The algorithm is evaluated across multiple quantization settings to test
generalizability:

- `ptq-7b-int4`: INT4 (4-bit) quantization with group size 128 — standard
  PTQ setting
- `ptq-7b-int3`: INT3 (3-bit) quantization with group size 128 — harder
  setting with only 8 levels
- `ptq-7b-int4-g64`: INT4 with group size 64 — finer granularity setting

Primary metric: `wikitext2_ppl` — WikiText-2 perplexity after
quantization (lower is better).
Secondary metric: `degradation` — perplexity increase over FP16 baseline
(lower is better).
Calibration: 128 sequences from WikiText-2 training set, 2048 tokens each.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/gptq/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

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
   165: # ── Model loading ─────────────────────────────────────────────────────────────
   166: 
   167: def get_model(model_path):
   168:     """Load a pretrained causal LM with weight initialization skipped."""
   169:     def skip(*args, **kwargs):
   170:         pass
   171:     torch.nn.init.kaiming_uniform_ = skip
   172:     torch.nn.init.uniform_ = skip
   173:     torch.nn.init.normal_ = skip
   174: 
   175:     model = AutoModelForCausalLM.from_pretrained(
   176:         model_path, torch_dtype=torch.float16, device_map="cpu"
   177:     )
   178:     model.seqlen = min(getattr(model.config, "max_position_embeddings", 4096), 4096)
   179:     model.eval()
   180:     return model
   181: 
   182: 
   183: def find_linear_layers(module, prefix=""):
   184:     """Recursively find all nn.Linear layers in the model."""
   185:     result = {}
   186:     for name, child in module.named_children():
   187:         full_name = f"{prefix}.{name}" if prefix else name
   188:         if isinstance(child, nn.Linear):
   189:             result[full_name] = child
   190:         else:
   191:             result.update(find_linear_layers(child, full_name))
   192:     return result
   193: 
   194: 
   195: # ── Calibration data ──────────────────────────────────────────────────────────
   196: 
   197: def get_calibration_data(tokenizer, nsamples=128, seqlen=2048, seed=0):
   198:     """Load WikiText-2 calibration data."""
   199:     from datasets import load_dataset
   200: 
   201:     # Load from pre-downloaded cache (compute nodes have no network)
   202:     cache_dir = os.environ.get("HF_DATASETS_CACHE", "/data/wikitext2")
   203:     try:
   204:         traindata = load_dataset(
   205:             "wikitext", "wikitext-2-raw-v1", split="train", cache_dir=cache_dir
   206:         )
   207:     except Exception:
   208:         # Fallback: load directly from arrow files in cache
   209:         from datasets import Dataset
   210:         import glob
   211:         arrow = glob.glob(f"{cache_dir}/**/wikitext-train.arrow", recursive=True)
   212:         if arrow:
   213:             traindata = Dataset.from_file(arrow[0])
   214:         else:
   215:             raise FileNotFoundError(f"WikiText-2 train data not found in {cache_dir}")
   216: 
   217:     import random
   218:     random.seed(seed)
   219: 
   220:     trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
   221: 
   222:     trainloader = []
   223:     for _ in range(nsamples):
   224:         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
   225:         j = i + seqlen
   226:         inp = trainenc.input_ids[:, i:j]
   227:         trainloader.append(inp)
   228: 
   229:     return trainloader
   230: 
   231: 
   232: def get_eval_data(tokenizer, seqlen=2048):
   233:     """Load WikiText-2 test data for perplexity evaluation."""
   234:     from datasets import load_dataset
   235: 
   236:     cache_dir = os.environ.get("HF_DATASETS_CACHE", "/data/wikitext2")
   237:     try:
   238:         testdata = load_dataset(
   239:             "wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir
   240:         )
   241:     except Exception:
   242:         from datasets import Dataset
   243:         import glob
   244:         arrow = glob.glob(f"{cache_dir}/**/wikitext-test.arrow", recursive=True)
   245:         if arrow:
   246:             testdata = Dataset.from_file(arrow[0])
   247:         else:
   248:             raise FileNotFoundError(f"WikiText-2 test data not found in {cache_dir}")
   249: 
   250:     testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
   251:     return testenc
   252: 
   253: 
   254: # ── Layer-by-layer quantization ───────────────────────────────────────────────
   255: 
   256: @torch.no_grad()
   257: def quantize_model(model, calibration_data, dev, num_bits=4, group_size=-1):
   258:     """Quantize all linear layers in the model using LayerQuantizer.
   259: 
   260:     Processes the model layer-by-layer (transformer block by block) to
   261:     minimize GPU memory usage. For each block:
   262:       1. Move block to GPU
   263:       2. Run calibration data through to collect Hessian statistics
   264:       3. Quantize each linear sublayer using LayerQuantizer
   265:       4. Replace weights with quantized-dequantized values
   266:       5. Move block back to CPU
   267: 
   268:     Args:
   269:         model: pretrained causal LM
   270:         calibration_data: list of input_ids tensors for calibration
   271:         dev: torch device (GPU)
   272:         num_bits: target bit width
   273:         group_size: quantization group size; -1 for per-channel
   274: 
   275:     Returns:
   276:         dict mapping layer name -> quantization error (Frobenius norm)
   277:     """
   278:     print("Starting quantization...", flush=True)
   279:     use_cache = model.config.use_cache
   280:     model.config.use_cache = False
   281: 
   282:     layers = model.model.layers
   283:     model.model.embed_tokens = model.model.embed_tokens.to(dev)
   284:     if hasattr(model.model, "rotary_emb"):
   285:         model.model.rotary_emb = model.model.rotary_emb.to(dev)
   286:     layers[0] = layers[0].to(dev)
   287: 
   288:     dtype = next(iter(model.parameters())).dtype
   289:     nsamples = len(calibration_data)
   290:     seqlen = calibration_data[0].shape[1]
   291:     hidden_size = model.config.hidden_size
   292: 
   293:     # Capture inputs to first layer
   294:     inps = torch.zeros(
   295:         (nsamples, seqlen, hidden_size), dtype=dtype, device=dev
   296:     )
   297:     cache = {"i": 0, "attention_mask": None, "position_ids": None}
   298: 
   299:     class Catcher(nn.Module):
   300:         def __init__(self, module):
   301:             super().__init__()
   302:             self.module = module
   303:         def forward(self, inp, **kwargs):
   304:             inps[cache["i"]] = inp
   305:             cache["i"] += 1
   306:             cache["attention_mask"] = kwargs.get("attention_mask")
   307:             cache["position_ids"] = kwargs.get("position_ids")
   308:             cache["position_embeddings"] = kwargs.get("position_embeddings")
   309:             raise ValueError
   310: 
   311:     layers[0] = Catcher(layers[0])
   312:     for batch in calibration_data:
   313:         try:
   314:             model(batch.to(dev))
   315:         except ValueError:
   316:             pass
   317:     layers[0] = layers[0].module
   318: 
   319:     layers[0] = layers[0].cpu()
   320:     model.model.embed_tokens = model.model.embed_tokens.cpu()
   321:     if hasattr(model.model, "rotary_emb"):
   322:         model.model.rotary_emb = model.model.rotary_emb.cpu()
   323:     torch.cuda.empty_cache()
   324: 
   325:     outs = torch.zeros_like(inps)
   326:     attention_mask = cache["attention_mask"]
   327:     position_ids = cache["position_ids"]
   328: 
   329:     quant_errors = {}
   330: 
   331:     for i in range(len(layers)):
   332:         print(f"Quantizing layer {i}/{len(layers)}...", flush=True)
   333:         layer = layers[i].to(dev)
   334: 
   335:         # Find all linear sublayers in this transformer block
   336:         subset = find_linear_layers(layer)
   337: 
   338:         # Create quantizers and register hooks to collect calibration stats
   339:         quantizers = {}
   340:         for name in subset:
   341:             quantizers[name] = LayerQuantizer(subset[name], num_bits=num_bits, group_size=group_size)
   342: 
   343:         def make_hook(name):
   344:             def hook(_, inp, out):
   345:                 quantizers[name].add_batch(inp[0].data)
   346:             return hook
   347: 
   348:         handles = []
   349:         for name in subset:
   350:             handles.append(subset[name].register_forward_hook(make_hook(name)))
   351: 
   352:         # Run calibration data through this layer
   353:         for j in range(nsamples):
   354:             kwargs = {}
   355:             if attention_mask is not None:
   356:                 kwargs["attention_mask"] = attention_mask
   357:             if position_ids is not None:
   358:                 kwargs["position_ids"] = position_ids
   359:             position_embeddings = cache.get("position_embeddings")
   360:             if position_embeddings is not None:
   361:                 kwargs["position_embeddings"] = position_embeddings
   362:             outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
   363: 
   364:         for h in handles:
   365:             h.remove()
   366: 
   367:         # Quantize each sublayer
   368:         for name in subset:
   369:             W_orig = subset[name].weight.data.clone()
   370:             W_quant = quantizers[name].quantize()
   371:             error = (W_orig.float() - W_quant.float()).norm().item()
   372:             quant_errors[f"layers.{i}.{name}"] = error
   373:             subset[name].weight.data = W_quant
   374:             quantizers[name].free()
   375: 
   376:         # Re-run calibration through quantized layer to get outputs for next layer
   377:         for j in range(nsamples):
   378:             kwargs = {}
   379:             if attention_mask is not None:
   380:                 kwargs["attention_mask"] = attention_mask
   381:             if position_ids is not None:
   382:                 kwargs["position_ids"] = position_ids
   383:             position_embeddings = cache.get("position_embeddings")
   384:             if position_embeddings is not None:
   385:                 kwargs["position_embeddings"] = position_embeddings
   386:             outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
   387: 
   388:         layers[i] = layer.cpu()
   389:         del layer
   390:         del quantizers
   391:         torch.cuda.empty_cache()
   392: 
   393:         inps, outs = outs, inps
   394: 
   395:     model.config.use_cache = use_cache
   396:     print("Quantization complete.", flush=True)
   397:     return quant_errors
   398: 
   399: 
   400: # ── Perplexity evaluation ─────────────────────────────────────────────────────
   401: 
   402: @torch.no_grad()
   403: def evaluate_perplexity(model, testenc, dev):
   404:     """Evaluate perplexity on test data (layer-by-layer to save memory).
   405: 
   406:     Args:
   407:         model: (possibly quantized) causal LM
   408:         testenc: tokenized test data
   409:         dev: torch device
   410: 
   411:     Returns:
   412:         float perplexity value
   413:     """
   414:     print("Evaluating perplexity...", flush=True)
   415:     testenc = testenc.input_ids
   416:     seqlen = model.seqlen
   417:     nsamples = testenc.numel() // seqlen
   418: 
   419:     use_cache = model.config.use_cache
   420:     model.config.use_cache = False
   421:     layers = model.model.layers
   422: 
   423:     model.model.embed_tokens = model.model.embed_tokens.to(dev)
   424:     if hasattr(model.model, "rotary_emb"):
   425:         model.model.rotary_emb = model.model.rotary_emb.to(dev)
   426:     layers[0] = layers[0].to(dev)
   427: 
   428:     dtype = next(iter(model.parameters())).dtype
   429:     hidden_size = model.config.hidden_size
   430:     inps = torch.zeros(
   431:         (nsamples, seqlen, hidden_size), dtype=dtype, device=dev
   432:     )
   433:     cache = {"i": 0, "attention_mask": None, "position_ids": None}
   434: 
   435:     class Catcher(nn.Module):
   436:         def __init__(self, module):
   437:             super().__init__()
   438:             self.module = module
   439:         def forward(self, inp, **kwargs):
   440:             inps[cache["i"]] = inp
   441:             cache["i"] += 1
   442:             cache["attention_mask"] = kwargs.get("attention_mask")
   443:             cache["position_ids"] = kwargs.get("position_ids")
   444:             cache["position_embeddings"] = kwargs.get("position_embeddings")
   445:             raise ValueError
   446: 
   447:     layers[0] = Catcher(layers[0])
   448:     for i in range(nsamples):
   449:         batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(dev)
   450:         try:
   451:             model(batch)
   452:         except ValueError:
   453:             pass
   454:     layers[0] = layers[0].module
   455: 
   456:     layers[0] = layers[0].cpu()
   457:     model.model.embed_tokens = model.model.embed_tokens.cpu()
   458:     if hasattr(model.model, "rotary_emb"):
   459:         model.model.rotary_emb = model.model.rotary_emb.cpu()
   460:     torch.cuda.empty_cache()
   461: 
   462:     outs = torch.zeros_like(inps)
   463:     attention_mask = cache["attention_mask"]
   464:     position_ids = cache["position_ids"]
   465: 
   466:     for i in range(len(layers)):
   467:         layer = layers[i].to(dev)
   468:         for j in range(nsamples):
   469:             kwargs = {}
   470:             if attention_mask is not None:
   471:                 kwargs["attention_mask"] = attention_mask
   472:             if position_ids is not None:
   473:                 kwargs["position_ids"] = position_ids
   474:             position_embeddings = cache.get("position_embeddings")
   475:             if position_embeddings is not None:
   476:                 kwargs["position_embeddings"] = position_embeddings
   477:             outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
   478:         layers[i] = layer.cpu()
   479:         del layer
   480:         torch.cuda.empty_cache()
   481:         inps, outs = outs, inps
   482: 
   483:     if model.model.norm is not None:
   484:         model.model.norm = model.model.norm.to(dev)
   485:     model.lm_head = model.lm_head.to(dev)
   486: 
   487:     testenc = testenc.to(dev)
   488:     nlls = []
   489:     for i in range(nsamples):
   490:         hidden_states = inps[i].unsqueeze(0)
   491:         if model.model.norm is not None:
   492:             hidden_states = model.model.norm(hidden_states)
   493:         lm_logits = model.lm_head(hidden_states)
   494:         shift_logits = lm_logits[:, :-1, :].contiguous()
   495:         shift_labels = testenc[:, (i * seqlen):((i + 1) * seqlen)][:, 1:]
   496:         loss_fct = nn.CrossEntropyLoss()
   497:         loss = loss_fct(
   498:             shift_logits.view(-1, shift_logits.size(-1)),
   499:             shift_labels.view(-1)
   500:         )

[truncated: showing at most 500 lines / 60000 bytes from gptq/custom_ptq.py]
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **ptq-7b-int4** — wall-clock budget `0:19:00`, compute share `1.0`
- **ptq-7b-int3** — wall-clock budget `0:19:00`, compute share `1.0`
- **ptq-7b-int4-g64** — wall-clock budget `0:19:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



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
Lines 26–272:
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
   262:         return W_final.to(self.layer.weight.dtype)
   263: 
   264:     def free(self):
   265:         """Release calibration buffers."""
   266:         del self.H
   267:         del self.act_sum
   268:         del self._x_buf
   269:         self.H = None
   270:         self.act_sum = None
   271:         self._x_buf = None
   272: 
   273: 
   274: 
   275: # ═══════════════════════════════════════════════════════════════════════════════
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
