# MLS-Bench: llm-qat-algorithm

# LLM Quantization-Aware Training (QAT) Algorithm

## Research Question

Design a quantization-aware training (QAT) algorithm that minimizes the
perplexity gap between a full-precision Pythia-1.4B and the same model
quantized to very low bit-widths (INT4 / INT3 / INT2) at inference time.
The algorithm must be a *training-side* contribution: how the fake-quant
forward, the gradient flow, the learnable parameters, and the optimizer
schedule are designed. It must work uniformly across 4-, 3-, and 2-bit
settings, not just one.

## Background

Post-training quantization (PTQ) collapses at very low bit-widths because
every weight is rounded to one of `2^B` levels with no chance to repair the
resulting error. Quantization-Aware Training (QAT) attacks this by
inserting *fake quantization* into the forward pass during a short
fine-tune. The key knobs are:

- Gradient estimator: round-then-clamp is non-differentiable. The
  Straight-Through Estimator (STE) (Bengio et al., 2013) simply pretends
  the operation is identity in backward. Learning the step size jointly
  with the weights — Learned Step Size Quantization (LSQ; Esser et al.,
  ICLR 2020; arXiv:1902.08153) — gives a measurably tighter quantization
  grid and tends to dominate STE at INT2.
- Stability: low-bit QAT diverges easily; warming up the quantization
  noise and EMA-smoothing the scales (StableQAT-style) buys back several
  PPL points at INT2.

Group quantization (per-row, per-group of `group_size=128` columns,
symmetric, signed) is the standard low-bit format and is fixed for this
task. Linear layers in every transformer block are quantized; embeddings,
LayerNorm, and the LM head stay full precision.

A control baseline `finetune_then_ptq` runs a full-precision finetune on
the training corpus with the same schedule as the QAT methods and then
applies the same RTN quantize-dequantize as `no_qat`. This isolates the
finetune signal from the QAT signal: a useful QAT method must beat
`finetune_then_ptq`, otherwise its apparent gains over `no_qat` are just
the in-domain finetune talking.

## What You Can Modify

The single file `llm-qat-runtime/custom_qat.py` is created at task setup;
you may only edit the `# EDITABLE REGION START / END` block. It contains:

- `CONFIG_OVERRIDES` dict: per-method training hyperparameters
  (`learning_rate`, `num_steps`, `batch_size`,
  `gradient_accumulation_steps`, `max_grad_norm`, `warmup_steps`,
  `weight_decay`).
- `fake_quantize_weight(weight, num_bits, group_size)`: differentiable
  fake-quant for the QAT forward pass. Must allow gradient flow back to
  the original weight.
- `fake_quantize_activation(x, num_bits)`: optional (default identity for
  weight-only QAT).
- `class QATWrapper(nn.Module)`: wraps an `nn.Linear`; applies fake quant
  in `forward`; may hold extra learnable parameters (per-group scales for
  LSQ, EMA buffers for StableQAT, etc.). May expose an
  `aux_loss(step, total_steps)` method that the training loop adds to the
  cross-entropy loss. May expose `quant_scale()` returning its learned
  per-group quantization steps (one finite, nonzero step per row and
  group of `group_size` columns) for the final quantizer; returning
  `None` (the default) selects the max-abs RTN step.
- `prepare_qat_model(model, num_bits, group_size)`: replace every
  `nn.Linear` (and HF GPT-2 `Conv1D`) in the model with `QATWrapper`,
  initializing any extra learnable parameters. The function must restore
  the LM head (`embed_out` for Pythia / GPTNeoX, `lm_head` for GPT-style
  models) to a plain Linear so the output projection stays in full
  precision.

The fixed (non-editable) region implements: model load (Pythia-1.4B in
FP32 with gradient checkpointing), training-corpus sampling (block-1024
random crops), the QAT training loop (`AdamW`, cosine LR with warmup,
gradient accumulation, grad-norm clipping), real-quantization roundtrip
after training, and held-out perplexity evaluation.

## Architecture

- Backbone: HuggingFace `EleutherAI/pythia-1.4b` (1.4B parameters,
  GPTNeoX architecture, 24 layers x 16 heads x 2048 hidden, native
  context length 2048). Linear layers are wrapped via the recursive
  traversal in `prepare_qat_model`.
- Optimizer: AdamW, cosine schedule with linear warmup; training length
  and batching are exposed via `CONFIG_OVERRIDES` (see the interface
  below) and the agent may shorten/lengthen them.
- Training data: a standard language modelling corpus, random 1024-token crops.

## Interface

```python
CONFIG_OVERRIDES = {
    "learning_rate": 2e-5,
    "num_steps": 500,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "weight_decay": 0.0,
}

def fake_quantize_weight(weight, num_bits, group_size): ...   # differentiable
def fake_quantize_activation(x, num_bits): ...                # optional, default id

class QATWrapper(nn.Module):
    def __init__(self, linear, num_bits, group_size): ...
    @property
    def weight(self) -> torch.Tensor: ...
    @property
    def bias(self): ...
    def forward(self, x): ...
    def quant_scale(self): ...   # optional learned per-group steps, or None

def prepare_qat_model(model, num_bits, group_size): ...
```

Constraints:

- The forward path of every wrapped `nn.Linear` must use
  `fake_quantize_weight` (or an equivalent inside `QATWrapper.forward`)
  so the QAT signal actually trains the integer grid.
- After training, fixed (non-editable) code applies the real
  quantize-dequantize to every `linear.weight` of every `QATWrapper`:
  symmetric signed `num_bits` codes `clamp(round(w / s), qmin, qmax)`
  per row and group of `group_size` columns, with `s` taken from
  `QATWrapper.quant_scale()` or the max-abs RTN step. Each wrapper is then
  replaced by a plain `nn.Linear` holding the quantized weight (bias in
  full precision) and perplexity is measured, so `QATWrapper.forward` and
  the fake-quant functions only affect training. Your method must produce
  weights that, after this real QDQ roundtrip, still give a low
  perplexity. The run fails if any transformer-block `nn.Linear` was not
  wrapped, or if non-stock modules or forward hooks remain in the model
  at evaluation.
- Keep the LM head at full precision (the template already excludes
  `embed_out` / `lm_head`).
- Available imports in the editable region: `torch`, `torch.nn` (as
  `nn`), `torch.nn.functional` (as `F`), `numpy` (as `np`), `math`,
  `os`, `time`, plus `transformers.pytorch_utils.Conv1D`.
- All seeds and training hyperparameters must be deterministic given
  `--seed`.

## Reference baselines

### no_qat
Round-to-nearest (RTN) post-training quantization with no fine-tuning —
the pure PTQ lower bound.

### ste
Straight-Through Estimator (Bengio et al., 2013): fake-quantize in the
forward pass, pass the gradient through unchanged (identity) in the
backward pass. The canonical minimal QAT gradient estimator.

### lsq
Learned Step-Size Quantization (Esser et al., ICLR 2020, arXiv:1902.08153):
learnable per-group quantization scales trained jointly with the weights,
giving a tighter quantization grid than STE.

### finetune_then_ptq
Full-precision fine-tune on the training corpus (same schedule as QAT
methods) followed by RTN quantization. Isolates the in-domain fine-tune
signal from the QAT signal; a valid QAT method must outperform this baseline.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/llm-qat-runtime/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `llm-qat-runtime/custom_qat.py`
- editable lines **33–176**




## Readable Context


### `llm-qat-runtime/custom_qat.py`  [EDITABLE — lines 33–176 only]

```python
     1: """Quantization-Aware Training (QAT) for Pythia-1.4B -- finetune + evaluate.
     2: 
     3: This script:
     4:   1. Loads pretrained Pythia-1.4B (HF ``EleutherAI/pythia-1.4b``).
     5:   2. Replaces every nn.Linear with QATWrapper that applies fake-quant in
     6:      forward (so gradients can flow back through the quantization).
     7:   3. Runs a QAT fine-tune on WikiText-2 train (default ~1500 steps).
     8:   4. Applies a REAL quantize-dequantize roundtrip to every linear weight.
     9:   5. Evaluates perplexity on WikiText-2 test.
    10: 
    11: The QAT algorithm is defined in the EDITABLE REGION below.  Everything
    12: else (data loading, training loop, real-quant roundtrip, perplexity eval)
    13: is fixed and shared by every baseline and the agent.
    14: """
    15: 
    16: import argparse
    17: import math
    18: import os
    19: import time
    20: 
    21: import numpy as np
    22: import torch
    23: import torch.nn as nn
    24: import torch.nn.functional as F
    25: 
    26: from transformers import AutoModelForCausalLM, AutoTokenizer
    27: 
    28: 
    29: # ═══════════════════════════════════════════════════════════════════════════════
    30: # EDITABLE REGION START -- QAT Algorithm (lines 33-176)
    31: # ═══════════════════════════════════════════════════════════════════════════════
    32: 
    33: # Per-method training hyperparameters.  The training loop reads this dict.
    34: # Override any of these in your method to retune.
    35: CONFIG_OVERRIDES = {
    36:     "learning_rate": 2e-5,
    37:     "num_steps": 500,
    38:     "batch_size": 2,
    39:     "gradient_accumulation_steps": 4,
    40:     "max_grad_norm": 1.0,
    41:     "warmup_steps": 50,
    42:     "weight_decay": 0.0,
    43: }
    44: 
    45: 
    46: def _qrange(num_bits):
    47:     """Symmetric integer range for `num_bits`-bit signed quantization."""
    48:     qmax = (1 << (num_bits - 1)) - 1
    49:     qmin = -(1 << (num_bits - 1))
    50:     return qmin, qmax
    51: 
    52: 
    53: def fake_quantize_weight(weight, num_bits, group_size):
    54:     """Differentiable fake-quant of a 2D weight tensor.
    55: 
    56:     Forward: simulates `num_bits` symmetric per-group quantization.
    57:     Backward: straight-through estimator (gradient passes through unchanged).
    58: 
    59:     Args:
    60:         weight: float tensor of shape (out_features, in_features).
    61:         num_bits: bit width.
    62:         group_size: column group size (>0); in_features must be divisible.
    63: 
    64:     Returns:
    65:         Tensor of same shape and dtype as `weight`, quantize-dequantized.
    66:     """
    67:     qmin, qmax = _qrange(num_bits)
    68:     out_features, in_features = weight.shape
    69:     assert in_features % group_size == 0, (
    70:         f"in_features {in_features} not divisible by group_size {group_size}"
    71:     )
    72:     w = weight.float().reshape(out_features, -1, group_size)
    73:     w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    74:     scale = w_max / qmax
    75:     w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    76:     # Straight-through estimator: forward = quantized, backward = identity.
    77:     w_dq = w + (w_q - w).detach()
    78:     return w_dq.reshape(out_features, in_features).to(weight.dtype)
    79: 
    80: 
    81: def fake_quantize_activation(x, num_bits):
    82:     """Default identity (weight-only QAT).  Override to add activation QAT."""
    83:     return x
    84: 
    85: 
    86: # NOTE: the final quantize-dequantize used for evaluation is FIXED code
    87: # (`apply_real_quantization`, below the editable region).  It rounds every
    88: # QATWrapper's `linear.weight` to the symmetric signed per-group grid:
    89: #     codes = clamp(round(w / s), qmin, qmax),   w_q = codes * s
    90: # with one step `s` per (row, group of `group_size` columns).  By default
    91: # `s = |w|.amax(group) / qmax` (max-abs RTN).  A wrapper may instead supply
    92: # its own learned steps (LSQ scales, learned clipping, ...) through
    93: # `QATWrapper.quant_scale()`.  Evaluation then runs plain `nn.Linear` layers
    94: # holding `w_q` (bias kept in full precision), so `QATWrapper.forward`,
    95: # `fake_quantize_weight` and `fake_quantize_activation` are training-only.
    96: # The returned steps must be finite and nonzero, one per group.
    97: 
    98: 
    99: class QATWrapper(nn.Module):
   100:     """Wraps an nn.Linear and applies fake-quant to its weight in forward.
   101: 
   102:     The wrapped module exposes the original Linear's weight/bias as
   103:     submodule parameters so the QAT optimizer can update them; the bias
   104:     is left in full precision.
   105: 
   106:     Attributes
   107:     ----------
   108:     linear : nn.Linear
   109:         Underlying linear layer.  `linear.weight` is the trainable param.
   110:     num_bits : int
   111:     group_size : int
   112:     """
   113: 
   114:     def __init__(self, linear, num_bits, group_size):
   115:         super().__init__()
   116:         self.linear = linear
   117:         self.num_bits = num_bits
   118:         self.group_size = group_size
   119: 
   120:     @property
   121:     def weight(self):
   122:         return self.linear.weight
   123: 
   124:     @property
   125:     def bias(self):
   126:         return self.linear.bias
   127: 
   128:     def forward(self, x):
   129:         x = fake_quantize_activation(x, self.num_bits)
   130:         w_q = fake_quantize_weight(self.linear.weight, self.num_bits, self.group_size)
   131:         return F.linear(x, w_q, self.linear.bias)
   132: 
   133:     def quant_scale(self):
   134:         """Per-group step for the fixed final QDQ; None = max-abs RTN."""
   135:         return None
   136: 
   137: 
   138: def prepare_qat_model(model, num_bits, group_size):
   139:     """Replace every nn.Linear in `model` with a QATWrapper in-place.
   140: 
   141:     The LM head (``model.lm_head`` for GPT-style, ``model.embed_out`` for
   142:     Pythia / GPTNeoX) is restored to a plain Linear after the recursive
   143:     replace so the output projection stays in full precision.  HF GPT-2
   144:     Conv1D layers are converted to nn.Linear before wrapping.
   145:     """
   146:     from transformers.pytorch_utils import Conv1D  # type: ignore
   147: 
   148:     def _replace(parent):
   149:         for name, child in list(parent.named_children()):
   150:             if isinstance(child, nn.Linear):
   151:                 wrapper = QATWrapper(child, num_bits=num_bits, group_size=group_size)
   152:                 setattr(parent, name, wrapper)
   153:             elif isinstance(child, Conv1D):
   154:                 # Convert Conv1D -> Linear (Conv1D weight is (in, out), Linear is (out, in)).
   155:                 in_f, out_f = child.weight.shape
   156:                 lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
   157:                                 device=child.weight.device, dtype=child.weight.dtype)
   158:                 with torch.no_grad():
   159:                     lin.weight.copy_(child.weight.t().contiguous())
   160:                     if child.bias is not None:
   161:                         lin.bias.copy_(child.bias)
   162:                 wrapper = QATWrapper(lin, num_bits=num_bits, group_size=group_size)
   163:                 setattr(parent, name, wrapper)
   164:             else:
   165:                 _replace(child)
   166: 
   167:     _replace(model)
   168:     # Restore the LM head to full precision (covers GPT-2 `lm_head` and
   169:     # Pythia / GPTNeoX `embed_out`).
   170:     for head_attr in ("lm_head", "embed_out"):
   171:         head = getattr(model, head_attr, None)
   172:         if isinstance(head, QATWrapper):
   173:             setattr(model, head_attr, head.linear)
   174: 
   175:     return model
   176: 
   177: 
   178: # ═══════════════════════════════════════════════════════════════════════════════
   179: # EDITABLE REGION END
   180: # ═══════════════════════════════════════════════════════════════════════════════
   181: 
   182: 
   183: # ── Model loading ─────────────────────────────────────────────────────────────
   184: 
   185: def get_model(model_path):
   186:     """Load model in float32 for QAT training stability."""
   187:     model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
   188:     model.config.use_cache = False
   189:     model.seqlen = 2048
   190:     return model
   191: 
   192: 
   193: def find_qat_wrappers(module, prefix=""):
   194:     """Return dict {name: QATWrapper} of all QAT-wrapped layers."""
   195:     out = {}
   196:     for name, child in module.named_children():
   197:         full = f"{prefix}.{name}" if prefix else name
   198:         if isinstance(child, QATWrapper):
   199:             out[full] = child
   200:         else:
   201:             out.update(find_qat_wrappers(child, full))
   202:     return out
   203: 
   204: 
   205: # ── Data loading ──────────────────────────────────────────────────────────────
   206: 
   207: def load_wikitext2(tokenizer, seqlen, split):
   208:     from datasets import load_dataset, Dataset
   209:     import glob
   210: 
   211:     cache_dir = os.environ.get("HF_DATASETS_CACHE", "/data/wikitext2")
   212:     try:
   213:         data = load_dataset(
   214:             "wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir
   215:         )
   216:     except Exception:
   217:         # Fallback: read arrow file directly
   218:         arrow = glob.glob(f"{cache_dir}/**/wikitext-{split}.arrow", recursive=True)
   219:         if not arrow:
   220:             raise FileNotFoundError(f"WikiText-2 {split} not found in {cache_dir}")
   221:         data = Dataset.from_file(arrow[0])
   222: 
   223:     enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
   224:     return enc.input_ids  # (1, total_tokens)
   225: 
   226: 
   227: def make_train_batches(ids, batch_size, seqlen, num_steps, gradient_accumulation_steps, seed):
   228:     """Generator yielding randomly-sampled (input, target) blocks of length seqlen."""
   229:     rng = np.random.RandomState(seed)
   230:     total = ids.shape[1]
   231:     n_required = num_steps * gradient_accumulation_steps * batch_size
   232:     starts = rng.randint(0, total - seqlen - 1, size=n_required)
   233:     for k in range(num_steps * gradient_accumulation_steps):
   234:         batch = []
   235:         for b in range(batch_size):
   236:             i = int(starts[k * batch_size + b])
   237:             batch.append(ids[0, i:i + seqlen + 1])
   238:         x = torch.stack([t[:-1] for t in batch], dim=0)
   239:         y = torch.stack([t[1:]  for t in batch], dim=0)
   240:         yield x, y
   241: 
   242: 
   243: # ── Training loop ─────────────────────────────────────────────────────────────
   244: 
   245: def train_qat(model, tokenizer, dev, num_bits, group_size, seed):
   246:     cfg = {
   247:         "learning_rate": 2e-5,
   248:         "num_steps": 500,
   249:         "batch_size": 2,
   250:         "gradient_accumulation_steps": 4,
   251:         "max_grad_norm": 1.0,
   252:         "warmup_steps": 50,
   253:         "weight_decay": 0.0,
   254:     }
   255:     cfg.update(CONFIG_OVERRIDES)
   256: 
   257:     ids = load_wikitext2(tokenizer, model.seqlen, split="train").to(dev)
   258: 
   259:     # Optimizer over all trainable parameters (includes any extras the
   260:     # editable region added, e.g., LSQ scales or AdaRound betas).
   261:     trainable = [p for p in model.parameters() if p.requires_grad]
   262:     optim = torch.optim.AdamW(
   263:         trainable,
   264:         lr=cfg["learning_rate"],
   265:         betas=(0.9, 0.95),
   266:         weight_decay=cfg["weight_decay"],
   267:     )
   268: 
   269:     def lr_at(step):
   270:         if step < cfg["warmup_steps"]:
   271:             return cfg["learning_rate"] * (step + 1) / max(1, cfg["warmup_steps"])
   272:         # Cosine decay to 10% of base lr
   273:         progress = (step - cfg["warmup_steps"]) / max(1, cfg["num_steps"] - cfg["warmup_steps"])
   274:         return cfg["learning_rate"] * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
   275: 
   276:     model.train()
   277:     batches = make_train_batches(
   278:         ids, cfg["batch_size"], model.seqlen,
   279:         cfg["num_steps"], cfg["gradient_accumulation_steps"], seed,
   280:     )
   281:     t0 = time.time()
   282:     optim.zero_grad(set_to_none=True)
   283:     micro = 0
   284:     step = 0
   285:     running_loss = 0.0
   286:     running_aux = 0.0
   287:     for x, y in batches:
   288:         x = x.to(dev); y = y.to(dev)
   289:         logits = model(x).logits
   290:         loss = F.cross_entropy(
   291:             logits.reshape(-1, logits.size(-1)).float(),
   292:             y.reshape(-1),
   293:         )
   294:         # Sum any auxiliary losses contributed by per-module ``aux_loss``
   295:         # hooks (e.g. PACT alpha L2, AdaRound beta-annealed regularizer).
   296:         # Modules without an ``aux_loss`` callable are unaffected.
   297:         _aux = 0.0
   298:         for _m in model.modules():
   299:             _al = getattr(_m, "aux_loss", None)
   300:             if callable(_al):
   301:                 _v = _al(step=step, total_steps=cfg["num_steps"])
   302:                 if _v is not None:
   303:                     _aux = _aux + _v
   304:         loss = loss + _aux
   305:         (loss / cfg["gradient_accumulation_steps"]).backward()
   306:         running_loss += loss.item()
   307:         running_aux += float(_aux) if isinstance(_aux, (int, float)) else float(_aux.detach().item())
   308:         micro += 1
   309:         if micro == cfg["gradient_accumulation_steps"]:
   310:             torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
   311:             for g in optim.param_groups:
   312:                 g["lr"] = lr_at(step)
   313:             optim.step()
   314:             optim.zero_grad(set_to_none=True)
   315:             if (step + 1) % 25 == 0 or step == 0:
   316:                 avg = running_loss / max(1, micro)
   317:                 avg_aux = running_aux / max(1, micro)
   318:                 print(
   319:                     f"TRAIN_METRICS: step={step+1}/{cfg['num_steps']} "
   320:                     f"loss={avg:.4f} aux={avg_aux:.4f} lr={lr_at(step):.2e} "
   321:                     f"elapsed={time.time()-t0:.1f}",
   322:                     flush=True,
   323:                 )
   324:             running_loss = 0.0
   325:             running_aux = 0.0
   326:             micro = 0
   327:             step += 1
   328:             if step >= cfg["num_steps"]:
   329:                 break
   330: 
   331:     return time.time() - t0
   332: 
   333: 
   334: # ── Real-quant materialization (fixed) ────────────────────────────────────────
   335: 
   336: _HEAD_ATTRS = ("lm_head", "embed_out")
   337: _TRUSTED_MODULE_PREFIXES = ("torch.", "transformers.")
   338: 
   339: 
   340: def snapshot_quant_targets(model):
   341:     """Record every nn.Linear (except the LM head) of the pristine model.
   342: 
   343:     Called before ``prepare_qat_model``; ``apply_real_quantization`` later
   344:     requires each of these layers to come back as a quantized Linear.
   345:     """
   346:     heads = {id(getattr(model, a)) for a in _HEAD_ATTRS if getattr(model, a, None) is not None}
   347:     return {
   348:         name: (tuple(m.weight.shape), m.bias is not None)
   349:         for name, m in model.named_modules()
   350:         if isinstance(m, nn.Linear) and id(m) not in heads
   351:     }
   352: 
   353: 
   354: @torch.no_grad()
   355: def fixed_group_qdq(weight, num_bits, group_size, scale=None):
   356:     """Symmetric signed per-group ``num_bits`` QDQ with a verified grid.
   357: 
   358:     ``scale`` is None (max-abs RTN step) or one finite nonzero step per
   359:     (row, group).  Returns the dequantized weight in ``weight``'s dtype.
   360:     """
   361:     qmax = (1 << (num_bits - 1)) - 1
   362:     qmin = -(1 << (num_bits - 1))
   363:     out_features, in_features = weight.shape
   364:     if in_features % group_size != 0:
   365:         raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")
   366:     n_groups = in_features // group_size
   367:     w = weight.detach().float().reshape(out_features, n_groups, group_size)
   368:     if scale is None:
   369:         w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
   370:         s = w_max / qmax
   371:     else:
   372:         s = torch.as_tensor(scale).detach().to(device=w.device, dtype=torch.float32)
   373:         if s.numel() != out_features * n_groups:
   374:             raise ValueError(
   375:                 f"quant_scale() returned {s.numel()} steps, expected "
   376:                 f"{out_features}x{n_groups} (one per row and group)"
   377:             )
   378:         s = s.reshape(out_features, n_groups, 1)
   379:         if not bool(torch.isfinite(s).all()) or bool((s == 0).any()):
   380:             raise ValueError("quant_scale() steps must be finite and nonzero")
   381:     if not bool(torch.isfinite(w).all()):
   382:         raise ValueError("non-finite weight entering the final QDQ")
   383:     codes = torch.clamp(torch.round(w / s), qmin, qmax)
   384:     w_q = (codes * s).reshape(out_features, in_features).to(weight.dtype)
   385:     # Verify the stored tensor really is on the num_bits grid.
   386:     back = w_q.float().reshape(out_features, n_groups, group_size) / s
   387:     if (back - codes).abs().max().item() > 1e-3 or codes.min() < qmin or codes.max() > qmax:
   388:         raise RuntimeError("final QDQ produced weights off the num_bits grid")
   389:     return w_q
   390: 
   391: 
   392: @torch.no_grad()
   393: def apply_real_quantization(model, num_bits, group_size, targets):
   394:     """After QAT, replace each QATWrapper by a plain nn.Linear on the grid.
   395: 
   396:     The quantizer (format, rounding, clamping) is fixed here; only the
   397:     per-group steps may come from the method (``QATWrapper.quant_scale()``).
   398:     Evaluation then runs through the plain Linear, not the wrapper, and
   399:     every layer recorded by ``snapshot_quant_targets`` must be quantized.
   400:     """
   401:     wrappers = find_qat_wrappers(model)
   402:     quantized = {}
   403:     for name, w in wrappers.items():
   404:         weight = w.linear.weight
   405:         bias = w.linear.bias
   406:         get_scale = getattr(w, "quant_scale", None)
   407:         scale = get_scale() if callable(get_scale) else None
   408:         w_q = fixed_group_qdq(weight, num_bits, group_size, scale)
   409:         out_f, in_f = w_q.shape
   410:         lin = nn.Linear(in_f, out_f, bias=bias is not None,
   411:                         device=weight.device, dtype=weight.dtype)
   412:         lin.weight.copy_(w_q)
   413:         if bias is not None:
   414:             lin.bias.copy_(bias.detach().reshape(out_f))
   415:         lin.requires_grad_(False)
   416:         parent_name, _, attr = name.rpartition(".")
   417:         parent = model.get_submodule(parent_name) if parent_name else model
   418:         setattr(parent, attr, lin)
   419:         quantized[name] = lin
   420:     verify_eval_model(model, targets, quantized)
   421:     return len(quantized)
   422: 
   423: 
   424: def verify_eval_model(model, targets, quantized):
   425:     """Reject evaluation models the fixed quantizer does not fully cover."""
   426:     q_ids = {id(m) for m in quantized.values()}
   427:     for name, (shape, has_bias) in targets.items():
   428:         try:
   429:             m = model.get_submodule(name)
   430:         except AttributeError:
   431:             m = None
   432:         if m is None or id(m) not in q_ids:
   433:             raise RuntimeError(
   434:                 f"layer {name!r} was not quantized (every block nn.Linear must be "
   435:                 f"wrapped in QATWrapper by prepare_qat_model)"
   436:             )
   437:         if tuple(m.weight.shape) != shape or (m.bias is not None) != has_bias:
   438:             raise RuntimeError(f"layer {name!r} changed shape during QAT")
   439:     import torch.nn.modules.module as _mm
   440:     for hooks in ("_global_forward_hooks", "_global_forward_pre_hooks"):
   441:         if getattr(_mm, hooks, None):
   442:             raise RuntimeError("global module forward hooks are not allowed at evaluation")
   443:     for name, m in model.named_modules():
   444:         cls = type(m)
   445:         fwd = getattr(cls, "forward", None)
   446:         fwd_mod = getattr(fwd, "__module__", "") or ""
   447:         if (not cls.__module__.startswith(_TRUSTED_MODULE_PREFIXES)
   448:                 or not fwd_mod.startswith(_TRUSTED_MODULE_PREFIXES)
   449:                 or "forward" in m.__dict__):
   450:             raise RuntimeError(
   451:                 f"module {name!r} ({cls.__module__}.{cls.__name__}) is not a stock "
   452:                 f"torch/transformers module at evaluation"
   453:             )
   454:         if m._forward_hooks or m._forward_pre_hooks:
   455:             raise RuntimeError(f"module {name!r} carries forward hooks at evaluation")
   456: 
   457: 
   458: # ── Perplexity evaluation ─────────────────────────────────────────────────────
   459: 
   460: @torch.no_grad()
   461: def evaluate_perplexity(model, tokenizer, dev, seqlen):
   462:     model.eval()
   463:     ids = load_wikitext2(tokenizer, seqlen, split="test").to(dev)
   464:     nsamples = ids.shape[1] // seqlen
   465:     if nsamples == 0:
   466:         return float("nan")
   467:     nlls = []
   468:     for i in range(nsamples):
   469:         x = ids[:, i * seqlen:(i + 1) * seqlen]
   470:         logits = model(x).logits
   471:         shift_logits = logits[:, :-1, :].float().contiguous()
   472:         shift_labels = x[:, 1:]
   473:         loss = F.cross_entropy(
   474:             shift_logits.reshape(-1, shift_logits.size(-1)),
   475:             shift_labels.reshape(-1),
   476:         )
   477:         nlls.append(loss.float() * (seqlen - 1))
   478:     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
   479:     return ppl.item()
   480: 
   481: 
   482: # ── Main ──────────────────────────────────────────────────────────────────────
   483: 
   484: def main():
   485:     p = argparse.ArgumentParser(description="QAT for Pythia-1.4B")
   486:     p.add_argument("--model-path", type=str, default="/data/pythia-1.4b")
   487:     p.add_argument("--num-bits", type=int, default=4)
   488:     p.add_argument("--group-size", type=int, default=128)
   489:     p.add_argument("--seqlen", type=int, default=2048)
   490:     p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
   491:     args = p.parse_args()
   492: 
   493:     torch.manual_seed(args.seed)
   494:     np.random.seed(args.seed)
   495: 
   496:     dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   497:     overall_t0 = time.time()
   498: 
   499:     print(f"Loading model from {args.model_path}...", flush=True)
   500:     model = get_model(args.model_path)
   501:     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
   502:     if tokenizer.pad_token is None:
   503:         tokenizer.pad_token = tokenizer.eos_token
   504:     model.seqlen = args.seqlen
   505: 
   506:     # Enable gradient checkpointing to fit Pythia-1.4B + AdamW on 80GB.
   507:     try:
   508:         model.gradient_checkpointing_enable()
   509:     except Exception as e:
   510:         print(f"warn: gradient_checkpointing_enable failed: {e}", flush=True)
   511: 
   512:     # FP32 baseline ppl
   513:     print("\n=== FP baseline evaluation ===", flush=True)
   514:     model.to(dev)
   515:     fp_ppl = evaluate_perplexity(model, tokenizer, dev, args.seqlen)
   516:     print(f"FP baseline perplexity: {fp_ppl:.4f}", flush=True)
   517:     print(f"TRAIN_METRICS: fp_perplexity={fp_ppl:.4f}", flush=True)
   518: 
   519:     # Wrap model for QAT
   520:     quant_targets = snapshot_quant_targets(model)
   521:     print(f"\n=== Preparing QAT (INT{args.num_bits}, group_size={args.group_size}) ===", flush=True)
   522:     model = prepare_qat_model(model, num_bits=args.num_bits, group_size=args.group_size)
   523:     model.to(dev)
   524:     n_wrapped = len(find_qat_wrappers(model))
   525:     print(f"Wrapped {n_wrapped} linear layers as QATWrapper", flush=True)
   526: 
   527:     # QAT finetune
   528:     print("\n=== QAT fine-tuning ===", flush=True)
   529:     qat_time = train_qat(model, tokenizer, dev, args.num_bits, args.group_size, args.seed)
   530:     print(f"QAT finetune done in {qat_time:.1f}s", flush=True)
   531: 
   532:     # Real-quant roundtrip
   533:     print("\n=== Materializing real INT-N weights ===", flush=True)
   534:     n_q = apply_real_quantization(model, args.num_bits, args.group_size, quant_targets)
   535:     print(f"Quantized {n_q} layers to INT{args.num_bits}", flush=True)
   536: 
   537:     # Quantized ppl
   538:     print("\n=== Quantized evaluation ===", flush=True)
   539:     q_ppl = evaluate_perplexity(model, tokenizer, dev, args.seqlen)
   540: 
   541:     elapsed = time.time() - overall_t0
   542:     degradation = q_ppl - fp_ppl
   543:     print(f"\n=== Results ===", flush=True)
   544:     print(f"FP   perplexity: {fp_ppl:.4f}", flush=True)
   545:     print(f"INT{args.num_bits} perplexity: {q_ppl:.4f}", flush=True)
   546:     print(f"Degradation:     {degradation:.4f}", flush=True)
   547:     print(
   548:         f"TEST_METRICS: wikitext2_ppl={q_ppl:.4f} fp16_ppl={fp_ppl:.4f} "
   549:         f"degradation={degradation:.4f} qat_time={qat_time:.1f} elapsed={elapsed:.1f}",
   550:         flush=True,
   551:     )
   552: 
   553: 
   554: if __name__ == "__main__":
   555:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `no_qat` baseline — editable region  [READ-ONLY — reference implementation]

In `llm-qat-runtime/custom_qat.py`:

```python
Lines 33–116:
    30: # EDITABLE REGION START -- QAT Algorithm (lines 33-176)
    31: # ═══════════════════════════════════════════════════════════════════════════════
    32: 
    33: 
    34: # ── PTQ-only baseline: no QAT fine-tune, real QDQ at eval time ────────────────
    35: 
    36: CONFIG_OVERRIDES = {
    37:     "learning_rate": 0.0,
    38:     "num_steps": 0,
    39:     "batch_size": 2,
    40:     "gradient_accumulation_steps": 1,
    41:     "max_grad_norm": 1.0,
    42:     "warmup_steps": 0,
    43:     "weight_decay": 0.0,
    44: }
    45: 
    46: 
    47: def _qrange(num_bits):
    48:     qmax = (1 << (num_bits - 1)) - 1
    49:     qmin = -(1 << (num_bits - 1))
    50:     return qmin, qmax
    51: 
    52: 
    53: def fake_quantize_weight(weight, num_bits, group_size):
    54:     qmin, qmax = _qrange(num_bits)
    55:     out_features, in_features = weight.shape
    56:     assert in_features % group_size == 0
    57:     w = weight.float().reshape(out_features, -1, group_size)
    58:     w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    59:     scale = w_max / qmax
    60:     w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    61:     w_dq = w + (w_q - w).detach()
    62:     return w_dq.reshape(out_features, in_features).to(weight.dtype)
    63: 
    64: 
    65: def fake_quantize_activation(x, num_bits):
    66:     return x
    67: 
    68: 
    69: class QATWrapper(nn.Module):
    70:     def __init__(self, linear, num_bits, group_size):
    71:         super().__init__()
    72:         self.linear = linear
    73:         self.num_bits = num_bits
    74:         self.group_size = group_size
    75: 
    76:     @property
    77:     def weight(self):
    78:         return self.linear.weight
    79: 
    80:     @property
    81:     def bias(self):
    82:         return self.linear.bias
    83: 
    84:     def forward(self, x):
    85:         # PTQ-only: in eval the real QDQ has already been applied to
    86:         # linear.weight, so we just call the underlying linear.  During
    87:         # the (zero-step) training phase this is a no-op anyway.
    88:         return F.linear(x, self.linear.weight, self.linear.bias)
    89: 
    90: 
    91: def prepare_qat_model(model, num_bits, group_size):
    92:     from transformers.pytorch_utils import Conv1D
    93: 
    94:     def _replace(parent):
    95:         for name, child in list(parent.named_children()):
    96:             if isinstance(child, nn.Linear):
    97:                 setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
    98:             elif isinstance(child, Conv1D):
    99:                 in_f, out_f = child.weight.shape
   100:                 lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
   101:                                 device=child.weight.device, dtype=child.weight.dtype)
   102:                 with torch.no_grad():
   103:                     lin.weight.copy_(child.weight.t().contiguous())
   104:                     if child.bias is not None:
   105:                         lin.bias.copy_(child.bias)
   106:                 setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
   107:             else:
   108:                 _replace(child)
   109: 
   110:     _replace(model)
   111:     for head_attr in ("lm_head", "embed_out"):
   112:         head = getattr(model, head_attr, None)
   113:         if isinstance(head, QATWrapper):
   114:             setattr(model, head_attr, head.linear)
   115:     return model
   116: 
   117: 
   118: # ═══════════════════════════════════════════════════════════════════════════════
   119: # EDITABLE REGION END
```

### `ste` baseline — editable region  [READ-ONLY — reference implementation]

In `llm-qat-runtime/custom_qat.py`:

```python
Lines 33–117:
    30: # EDITABLE REGION START -- QAT Algorithm (lines 33-176)
    31: # ═══════════════════════════════════════════════════════════════════════════════
    32: 
    33: 
    34: # ── Straight-Through Estimator (STE) QAT baseline ─────────────────────────────
    35: 
    36: CONFIG_OVERRIDES = {
    37:     "learning_rate": 2e-5,
    38:     "num_steps": 500,
    39:     "batch_size": 2,
    40:     "gradient_accumulation_steps": 4,
    41:     "max_grad_norm": 1.0,
    42:     "warmup_steps": 50,
    43:     "weight_decay": 0.0,
    44: }
    45: 
    46: 
    47: def _qrange(num_bits):
    48:     qmax = (1 << (num_bits - 1)) - 1
    49:     qmin = -(1 << (num_bits - 1))
    50:     return qmin, qmax
    51: 
    52: 
    53: def fake_quantize_weight(weight, num_bits, group_size):
    54:     qmin, qmax = _qrange(num_bits)
    55:     out_features, in_features = weight.shape
    56:     assert in_features % group_size == 0
    57:     w = weight.float().reshape(out_features, -1, group_size)
    58:     # Recompute scale on-the-fly each forward (max-abs / qmax).
    59:     w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    60:     scale = w_max / qmax
    61:     w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    62:     # Straight-through: forward = quantized, backward = identity.
    63:     w_dq = w + (w_q - w).detach()
    64:     return w_dq.reshape(out_features, in_features).to(weight.dtype)
    65: 
    66: 
    67: def fake_quantize_activation(x, num_bits):
    68:     return x
    69: 
    70: 
    71: class QATWrapper(nn.Module):
    72:     def __init__(self, linear, num_bits, group_size):
    73:         super().__init__()
    74:         self.linear = linear
    75:         self.num_bits = num_bits
    76:         self.group_size = group_size
    77: 
    78:     @property
    79:     def weight(self):
    80:         return self.linear.weight
    81: 
    82:     @property
    83:     def bias(self):
    84:         return self.linear.bias
    85: 
    86:     def forward(self, x):
    87:         x = fake_quantize_activation(x, self.num_bits)
    88:         w_q = fake_quantize_weight(self.linear.weight, self.num_bits, self.group_size)
    89:         return F.linear(x, w_q, self.linear.bias)
    90: 
    91: 
    92: def prepare_qat_model(model, num_bits, group_size):
    93:     from transformers.pytorch_utils import Conv1D
    94: 
    95:     def _replace(parent):
    96:         for name, child in list(parent.named_children()):
    97:             if isinstance(child, nn.Linear):
    98:                 setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
    99:             elif isinstance(child, Conv1D):
   100:                 in_f, out_f = child.weight.shape
   101:                 lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
   102:                                 device=child.weight.device, dtype=child.weight.dtype)
   103:                 with torch.no_grad():
   104:                     lin.weight.copy_(child.weight.t().contiguous())
   105:                     if child.bias is not None:
   106:                         lin.bias.copy_(child.bias)
   107:                 setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
   108:             else:
   109:                 _replace(child)
   110: 
   111:     _replace(model)
   112:     for head_attr in ("lm_head", "embed_out"):
   113:         head = getattr(model, head_attr, None)
   114:         if isinstance(head, QATWrapper):
   115:             setattr(model, head_attr, head.linear)
   116:     return model
   117: 
   118: 
   119: # ═══════════════════════════════════════════════════════════════════════════════
   120: # EDITABLE REGION END
```

### `lsq` baseline — editable region  [READ-ONLY — reference implementation]

In `llm-qat-runtime/custom_qat.py`:

```python
Lines 33–180:
    30: # EDITABLE REGION START -- QAT Algorithm (lines 33-176)
    31: # ═══════════════════════════════════════════════════════════════════════════════
    32: 
    33: 
    34: # ── Learned Step Size Quantization (LSQ) ──────────────────────────────────────
    35: 
    36: CONFIG_OVERRIDES = {
    37:     "learning_rate": 2e-5,
    38:     "num_steps": 500,
    39:     "batch_size": 2,
    40:     "gradient_accumulation_steps": 4,
    41:     "max_grad_norm": 1.0,
    42:     "warmup_steps": 50,
    43:     "weight_decay": 0.0,
    44: }
    45: 
    46: 
    47: def _qrange(num_bits):
    48:     qmax = (1 << (num_bits - 1)) - 1
    49:     qmin = -(1 << (num_bits - 1))
    50:     return qmin, qmax
    51: 
    52: 
    53: class _LSQQuant(torch.autograd.Function):
    54:     """LSQ quantize-dequantize with the gradient of arxiv:1902.08153 eq. 5."""
    55: 
    56:     @staticmethod
    57:     def forward(ctx, w, scale, qmin, qmax, g_scale):
    58:         # w: (G, group_size); scale: (G, 1) broadcastable.
    59:         w_div = w / scale
    60:         w_clip = torch.clamp(w_div, qmin, qmax)
    61:         w_round = torch.round(w_clip)
    62:         ctx.save_for_backward(w_div, scale)
    63:         ctx.qmin = qmin
    64:         ctx.qmax = qmax
    65:         ctx.g_scale = g_scale
    66:         return w_round * scale
    67: 
    68:     @staticmethod
    69:     def backward(ctx, grad_out):
    70:         w_div, scale = ctx.saved_tensors
    71:         qmin, qmax, g = ctx.qmin, ctx.qmax, ctx.g_scale
    72:         # Gradient w.r.t. w: pass-through inside the clip range.
    73:         in_range = (w_div > qmin) & (w_div < qmax)
    74:         grad_w = torch.where(in_range, grad_out, torch.zeros_like(grad_out))
    75:         # Gradient w.r.t. s: see LSQ paper eq. 5.
    76:         below = (w_div <= qmin).float() * float(qmin)
    77:         above = (w_div >= qmax).float() * float(qmax)
    78:         inside = in_range.float() * (torch.round(w_div) - w_div)
    79:         grad_s_per_elem = (below + above + inside) * grad_out
    80:         grad_s = grad_s_per_elem.sum(dim=-1, keepdim=True) * g
    81:         return grad_w, grad_s, None, None, None
    82: 
    83: 
    84: def fake_quantize_weight(weight, num_bits, group_size, scale=None):
    85:     qmin, qmax = _qrange(num_bits)
    86:     out_features, in_features = weight.shape
    87:     assert in_features % group_size == 0
    88:     w = weight.float().reshape(out_features, -1, group_size)
    89:     if scale is None:
    90:         # No learnable scale supplied (prepare-time call) -- fall back to STE.
    91:         w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    92:         s = w_max / qmax
    93:         w_q = torch.clamp(torch.round(w / s), qmin, qmax) * s
    94:         w_dq = w + (w_q - w).detach()
    95:     else:
    96:         n_elem = w.numel()
    97:         g_scale = 1.0 / max(1.0, math.sqrt(n_elem * qmax))
    98:         w_dq = _LSQQuant.apply(w, scale, qmin, qmax, g_scale)
    99:     return w_dq.reshape(out_features, in_features).to(weight.dtype)
   100: 
   101: 
   102: def fake_quantize_activation(x, num_bits):
   103:     return x
   104: 
   105: 
   106: class QATWrapper(nn.Module):
   107:     def __init__(self, linear, num_bits, group_size):
   108:         super().__init__()
   109:         self.linear = linear
   110:         self.num_bits = num_bits
   111:         self.group_size = group_size
   112:         qmin, qmax = _qrange(num_bits)
   113:         out_features, in_features = linear.weight.shape
   114:         n_groups = in_features // group_size
   115:         # LSQ initial scale: 2 * |W|.mean() / sqrt(qmax)  (paper Sec. 3.4).
   116:         with torch.no_grad():
   117:             w = linear.weight.float().reshape(out_features, n_groups, group_size)
   118:             init = 2.0 * w.abs().mean(dim=-1, keepdim=True) / max(1.0, math.sqrt(qmax))
   119:             init = init.clamp(min=1e-8)
   120:         # Shape (out_features, n_groups, 1) so it broadcasts over group_size.
   121:         self.lsq_scale = nn.Parameter(init.to(linear.weight.dtype))
   122: 
   123:     @property
   124:     def weight(self):
   125:         return self.linear.weight
   126: 
   127:     @property
   128:     def bias(self):
   129:         return self.linear.bias
   130: 
   131:     def forward(self, x):
   132:         x = fake_quantize_activation(x, self.num_bits)
   133:         if self.training:
   134:             w_q = fake_quantize_weight(
   135:                 self.linear.weight, self.num_bits, self.group_size,
   136:                 scale=self.lsq_scale.float(),
   137:             )
   138:         else:
   139:             # Eval: produce a *real* quantize-dequantize on the LSQ grid.
   140:             qmin, qmax = _qrange(self.num_bits)
   141:             with torch.no_grad():
   142:                 w = self.linear.weight.float().reshape(
   143:                     self.linear.weight.shape[0], -1, self.group_size
   144:                 )
   145:                 s = self.lsq_scale.float()
   146:                 w_q = torch.clamp(torch.round(w / s), qmin, qmax) * s
   147:                 w_q = w_q.reshape_as(self.linear.weight).to(self.linear.weight.dtype)
   148:         return F.linear(x, w_q, self.linear.bias)
   149: 
   150:     def quant_scale(self):
   151:         # Hand the learned LSQ steps to the fixed final QDQ.
   152:         return self.lsq_scale
   153: 
   154: 
   155: def prepare_qat_model(model, num_bits, group_size):
   156:     from transformers.pytorch_utils import Conv1D
   157: 
   158:     def _replace(parent):
   159:         for name, child in list(parent.named_children()):
   160:             if isinstance(child, nn.Linear):
   161:                 setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
   162:             elif isinstance(child, Conv1D):
   163:                 in_f, out_f = child.weight.shape
   164:                 lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
   165:                                 device=child.weight.device, dtype=child.weight.dtype)
   166:                 with torch.no_grad():
   167:                     lin.weight.copy_(child.weight.t().contiguous())
   168:                     if child.bias is not None:
   169:                         lin.bias.copy_(child.bias)
   170:                 setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
   171:             else:
   172:                 _replace(child)
   173: 
   174:     _replace(model)
   175:     for head_attr in ("lm_head", "embed_out"):
   176:         head = getattr(model, head_attr, None)
   177:         if isinstance(head, QATWrapper):
   178:             setattr(model, head_attr, head.linear)
   179:     return model
   180: 
   181: 
   182: # ═══════════════════════════════════════════════════════════════════════════════
   183: # EDITABLE REGION END
```

### `finetune_then_ptq` baseline — editable region  [READ-ONLY — reference implementation]

In `llm-qat-runtime/custom_qat.py`:

```python
Lines 33–112:
    30: # EDITABLE REGION START -- QAT Algorithm (lines 33-176)
    31: # ═══════════════════════════════════════════════════════════════════════════════
    32: 
    33: 
    34: # ── Finetune-then-PTQ control baseline ────────────────────────────────────────
    35: # Forward pass during training is pure FP (no fake quant), but the same
    36: # training schedule as STE/LSQ/PACT is run.  After training, real RTN
    37: # QDQ is applied to materialize the integer model.
    38: 
    39: CONFIG_OVERRIDES = {
    40:     "learning_rate": 2e-5,
    41:     "num_steps": 500,
    42:     "batch_size": 2,
    43:     "gradient_accumulation_steps": 4,
    44:     "max_grad_norm": 1.0,
    45:     "warmup_steps": 50,
    46:     "weight_decay": 0.0,
    47: }
    48: 
    49: 
    50: def _qrange(num_bits):
    51:     qmax = (1 << (num_bits - 1)) - 1
    52:     qmin = -(1 << (num_bits - 1))
    53:     return qmin, qmax
    54: 
    55: 
    56: def fake_quantize_weight(weight, num_bits, group_size):
    57:     # Identity: no fake quant in forward -- pure FP finetune.
    58:     return weight
    59: 
    60: 
    61: def fake_quantize_activation(x, num_bits):
    62:     return x
    63: 
    64: 
    65: class QATWrapper(nn.Module):
    66:     def __init__(self, linear, num_bits, group_size):
    67:         super().__init__()
    68:         self.linear = linear
    69:         self.num_bits = num_bits
    70:         self.group_size = group_size
    71: 
    72:     @property
    73:     def weight(self):
    74:         return self.linear.weight
    75: 
    76:     @property
    77:     def bias(self):
    78:         return self.linear.bias
    79: 
    80:     def forward(self, x):
    81:         # Pure FP forward during training (no fake quant).  At eval time
    82:         # the real QDQ has already been applied to ``linear.weight``, so
    83:         # this still produces the genuine INT-N output.
    84:         return F.linear(x, self.linear.weight, self.linear.bias)
    85: 
    86: 
    87: def prepare_qat_model(model, num_bits, group_size):
    88:     from transformers.pytorch_utils import Conv1D
    89: 
    90:     def _replace(parent):
    91:         for name, child in list(parent.named_children()):
    92:             if isinstance(child, nn.Linear):
    93:                 setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
    94:             elif isinstance(child, Conv1D):
    95:                 in_f, out_f = child.weight.shape
    96:                 lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
    97:                                 device=child.weight.device, dtype=child.weight.dtype)
    98:                 with torch.no_grad():
    99:                     lin.weight.copy_(child.weight.t().contiguous())
   100:                     if child.bias is not None:
   101:                         lin.bias.copy_(child.bias)
   102:                 setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
   103:             else:
   104:                 _replace(child)
   105: 
   106:     _replace(model)
   107:     for head_attr in ("lm_head", "embed_out"):
   108:         head = getattr(model, head_attr, None)
   109:         if isinstance(head, QATWrapper):
   110:             setattr(model, head_attr, head.linear)
   111:     return model
   112: 
   113: 
   114: # ═══════════════════════════════════════════════════════════════════════════════
   115: # EDITABLE REGION END
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
