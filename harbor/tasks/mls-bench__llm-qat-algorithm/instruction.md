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
WikiText-2 train with the same schedule as the QAT methods (`lr=2e-5`,
500 steps, batch 2, grad-accum 4) and then applies the same RTN
quantize-dequantize as `no_qat`. This isolates the finetune signal from
the QAT signal: a useful QAT method must beat `finetune_then_ptq`,
otherwise its apparent gains over `no_qat` are just the in-domain
finetune talking.

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
- `quantize_dequantize_weight(weight, num_bits, group_size)`: REAL
  (no-grad) per-group symmetric QDQ used after training to materialize the
  integer model for evaluation.
- `class QATWrapper(nn.Module)`: wraps an `nn.Linear`; applies fake quant
  in `forward`; may hold extra learnable parameters (per-group scales for
  LSQ, EMA buffers for StableQAT, etc.). May expose an
  `aux_loss(step, total_steps)` method that the training loop adds to the
  cross-entropy loss.
- `prepare_qat_model(model, num_bits, group_size)`: replace every
  `nn.Linear` (and HF GPT-2 `Conv1D`) in the model with `QATWrapper`,
  initializing any extra learnable parameters. The function must restore
  the LM head (`embed_out` for Pythia / GPTNeoX, `lm_head` for GPT-style
  models) to a plain Linear so the output projection stays in full
  precision.

The fixed (non-editable) region implements: model load (Pythia-1.4B in
FP32 with gradient checkpointing), WikiText-2 train data sampling
(block-1024 random crops), the QAT training loop (`AdamW`, cosine LR with
warmup, gradient accumulation, grad-norm clipping), real-quantization
roundtrip after training, and WikiText-2 test perplexity evaluation.

## Architecture

- Backbone: HuggingFace `EleutherAI/pythia-1.4b` (1.4B parameters,
  GPTNeoX architecture, 24 layers x 16 heads x 2048 hidden, native
  context length 2048). Linear layers are wrapped via the recursive
  traversal in `prepare_qat_model`.
- Optimizer: AdamW, cosine schedule with linear warmup. Default 500 steps
  x batch 2 x grad-accum 4 (~4000 sequences seen, seqlen 1024) — the
  agent may shorten/lengthen via `CONFIG_OVERRIDES`.
- Calibration / training data: WikiText-2 raw v1 train split. Random
  1024-token crops.
- Evaluation: WikiText-2 raw v1 test split, sliding non-overlapping
  blocks of 1024 tokens, exponentiated mean cross-entropy loss.

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
def quantize_dequantize_weight(weight, num_bits, group_size): # no-grad QDQ

class QATWrapper(nn.Module):
    def __init__(self, linear, num_bits, group_size): ...
    @property
    def weight(self) -> torch.Tensor: ...
    @property
    def bias(self): ...
    def forward(self, x): ...

def prepare_qat_model(model, num_bits, group_size): ...
```

Constraints:

- The forward path of every wrapped `nn.Linear` must use
  `fake_quantize_weight` (or an equivalent inside `QATWrapper.forward`)
  so the QAT signal actually trains the integer grid.
- After training, `quantize_dequantize_weight` is applied to every
  `linear.weight` of every `QATWrapper`, then perplexity is measured.
  Your method must produce weights that, after this real QDQ roundtrip,
  still give a low perplexity.
- Keep the LM head at full precision (the template already excludes
  `embed_out` / `lm_head`).
- Available imports in the editable region: `torch`, `torch.nn` (as
  `nn`), `torch.nn.functional` (as `F`), `numpy` (as `np`), `math`,
  `os`, `time`, plus `transformers.pytorch_utils.Conv1D`.
- All seeds and training hyperparameters must be deterministic given
  `--seed`.

## Evaluation

The algorithm is evaluated across three bit-widths:

- `qat-1b-int4`: INT4, group size 128 — easy.
- `qat-1b-int3`: INT3, group size 128 — medium (8 levels).
- `qat-1b-int2`: INT2, group size 128 — extreme (4 levels).

Primary metric: `wikitext2_ppl` — WikiText-2 perplexity after the real
QDQ roundtrip, lower is better.
Secondary metric: `degradation` — `wikitext2_ppl - fp16_ppl`, where
`fp16_ppl` is the FP baseline measured before any quantization.

Note on absolute PPL vs. literature (OmniQuant / EfficientQAT tables):
QAT here finetunes on WikiText-2 train and evaluates on WikiText-2 test
(disjoint articles, but same domain). With 500 steps x bsz 2 x ga 4 =
4000 sequences x 1024 tokens, the FP16 finetune alone can drop test PPL
below the FP16 baseline (cf. `finetune_then_ptq` INT4 < `no_qat` FP16),
because the QAT train domain matches the eval domain. Published OmniQuant
/ EfficientQAT tables on LLaMA-{7B,13B} use C4 calibration and a
held-out WikiText eval, so their absolute W2g128 / W3g128 / W4g128
numbers are not directly comparable to ours. The intended internal
comparison is QAT-method vs `finetune_then_ptq`: a method that beats
`finetune_then_ptq` is showing real QAT signal, beyond the in-domain
finetune effect.

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
Full-precision fine-tune on WikiText-2 (same schedule as QAT methods)
followed by RTN quantization. Isolates the in-domain fine-tune signal
from the QAT signal; a valid QAT method must outperform this baseline.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/llm-qat-runtime/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

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
    86: def quantize_dequantize_weight(weight, num_bits, group_size):
    87:     """REAL (non-differentiable) symmetric per-group QDQ for post-training.
    88: 
    89:     Used after QAT finetune to materialize the quantized weights for eval.
    90:     Returns the same shape/dtype as `weight`.
    91:     """
    92:     qmin, qmax = _qrange(num_bits)
    93:     out_features, in_features = weight.shape
    94:     assert in_features % group_size == 0
    95:     with torch.no_grad():
    96:         w = weight.float().reshape(out_features, -1, group_size)
    97:         w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    98:         scale = w_max / qmax
    99:         w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
   100:         return w_q.reshape(out_features, in_features).to(weight.dtype)
   101: 
   102: 
   103: class QATWrapper(nn.Module):
   104:     """Wraps an nn.Linear and applies fake-quant to its weight in forward.
   105: 
   106:     The wrapped module exposes the original Linear's weight/bias as
   107:     submodule parameters so the QAT optimizer can update them; the bias
   108:     is left in full precision.
   109: 
   110:     Attributes
   111:     ----------
   112:     linear : nn.Linear
   113:         Underlying linear layer.  `linear.weight` is the trainable param.
   114:     num_bits : int
   115:     group_size : int
   116:     """
   117: 
   118:     def __init__(self, linear, num_bits, group_size):
   119:         super().__init__()
   120:         self.linear = linear
   121:         self.num_bits = num_bits
   122:         self.group_size = group_size
   123: 
   124:     @property
   125:     def weight(self):
   126:         return self.linear.weight
   127: 
   128:     @property
   129:     def bias(self):
   130:         return self.linear.bias
   131: 
   132:     def forward(self, x):
   133:         x = fake_quantize_activation(x, self.num_bits)
   134:         w_q = fake_quantize_weight(self.linear.weight, self.num_bits, self.group_size)
   135:         return F.linear(x, w_q, self.linear.bias)
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
   334: # ── Real-quant materialization ────────────────────────────────────────────────
   335: 
   336: @torch.no_grad()
   337: def apply_real_quantization(model, num_bits, group_size):
   338:     """After QAT, replace each QATWrapper weight with the real QDQ value.
   339: 
   340:     The wrapper still applies fake-quant in forward, but with the weight
   341:     already materialized to the quantization grid the result is the true
   342:     INT-N model output (no train-time noise / scale drift).
   343:     """
   344:     wrappers = find_qat_wrappers(model)
   345:     for name, w in wrappers.items():
   346:         w_dq = quantize_dequantize_weight(w.linear.weight.data, num_bits, group_size)
   347:         w.linear.weight.data.copy_(w_dq)
   348:     return len(wrappers)
   349: 
   350: 
   351: # ── Perplexity evaluation ─────────────────────────────────────────────────────
   352: 
   353: @torch.no_grad()
   354: def evaluate_perplexity(model, tokenizer, dev, seqlen):
   355:     model.eval()
   356:     ids = load_wikitext2(tokenizer, seqlen, split="test").to(dev)
   357:     nsamples = ids.shape[1] // seqlen
   358:     if nsamples == 0:
   359:         return float("nan")
   360:     nlls = []
   361:     for i in range(nsamples):
   362:         x = ids[:, i * seqlen:(i + 1) * seqlen]
   363:         logits = model(x).logits
   364:         shift_logits = logits[:, :-1, :].float().contiguous()
   365:         shift_labels = x[:, 1:]
   366:         loss = F.cross_entropy(
   367:             shift_logits.reshape(-1, shift_logits.size(-1)),
   368:             shift_labels.reshape(-1),
   369:         )
   370:         nlls.append(loss.float() * (seqlen - 1))
   371:     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
   372:     return ppl.item()
   373: 
   374: 
   375: # ── Main ──────────────────────────────────────────────────────────────────────
   376: 
   377: def main():
   378:     p = argparse.ArgumentParser(description="QAT for Pythia-1.4B")
   379:     p.add_argument("--model-path", type=str, default="/data/pythia-1.4b")
   380:     p.add_argument("--num-bits", type=int, default=4)
   381:     p.add_argument("--group-size", type=int, default=128)
   382:     p.add_argument("--seqlen", type=int, default=2048)
   383:     p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
   384:     args = p.parse_args()
   385: 
   386:     torch.manual_seed(args.seed)
   387:     np.random.seed(args.seed)
   388: 
   389:     dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   390:     overall_t0 = time.time()
   391: 
   392:     print(f"Loading model from {args.model_path}...", flush=True)
   393:     model = get_model(args.model_path)
   394:     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
   395:     if tokenizer.pad_token is None:
   396:         tokenizer.pad_token = tokenizer.eos_token
   397:     model.seqlen = args.seqlen
   398: 
   399:     # Enable gradient checkpointing to fit Pythia-1.4B + AdamW on 80GB.
   400:     try:
   401:         model.gradient_checkpointing_enable()
   402:     except Exception as e:
   403:         print(f"warn: gradient_checkpointing_enable failed: {e}", flush=True)
   404: 
   405:     # FP32 baseline ppl
   406:     print("\n=== FP baseline evaluation ===", flush=True)
   407:     model.to(dev)
   408:     fp_ppl = evaluate_perplexity(model, tokenizer, dev, args.seqlen)
   409:     print(f"FP baseline perplexity: {fp_ppl:.4f}", flush=True)
   410:     print(f"TRAIN_METRICS: fp_perplexity={fp_ppl:.4f}", flush=True)
   411: 
   412:     # Wrap model for QAT
   413:     print(f"\n=== Preparing QAT (INT{args.num_bits}, group_size={args.group_size}) ===", flush=True)
   414:     model = prepare_qat_model(model, num_bits=args.num_bits, group_size=args.group_size)
   415:     model.to(dev)
   416:     n_wrapped = len(find_qat_wrappers(model))
   417:     print(f"Wrapped {n_wrapped} linear layers as QATWrapper", flush=True)
   418: 
   419:     # QAT finetune
   420:     print("\n=== QAT fine-tuning ===", flush=True)
   421:     qat_time = train_qat(model, tokenizer, dev, args.num_bits, args.group_size, args.seed)
   422:     print(f"QAT finetune done in {qat_time:.1f}s", flush=True)
   423: 
   424:     # Real-quant roundtrip
   425:     print("\n=== Materializing real INT-N weights ===", flush=True)
   426:     n_q = apply_real_quantization(model, args.num_bits, args.group_size)
   427:     print(f"Quantized {n_q} layers to INT{args.num_bits}", flush=True)
   428: 
   429:     # Quantized ppl
   430:     print("\n=== Quantized evaluation ===", flush=True)
   431:     q_ppl = evaluate_perplexity(model, tokenizer, dev, args.seqlen)
   432: 
   433:     elapsed = time.time() - overall_t0
   434:     degradation = q_ppl - fp_ppl
   435:     print(f"\n=== Results ===", flush=True)
   436:     print(f"FP   perplexity: {fp_ppl:.4f}", flush=True)
   437:     print(f"INT{args.num_bits} perplexity: {q_ppl:.4f}", flush=True)
   438:     print(f"Degradation:     {degradation:.4f}", flush=True)
   439:     print(
   440:         f"TEST_METRICS: wikitext2_ppl={q_ppl:.4f} fp16_ppl={fp_ppl:.4f} "
   441:         f"degradation={degradation:.4f} qat_time={qat_time:.1f} elapsed={elapsed:.1f}",
   442:         flush=True,
   443:     )
   444: 
   445: 
   446: if __name__ == "__main__":
   447:     main()
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
Lines 33–128:
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
    69: def quantize_dequantize_weight(weight, num_bits, group_size):
    70:     qmin, qmax = _qrange(num_bits)
    71:     out_features, in_features = weight.shape
    72:     assert in_features % group_size == 0
    73:     with torch.no_grad():
    74:         w = weight.float().reshape(out_features, -1, group_size)
    75:         w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    76:         scale = w_max / qmax
    77:         w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    78:         return w_q.reshape(out_features, in_features).to(weight.dtype)
    79: 
    80: 
    81: class QATWrapper(nn.Module):
    82:     def __init__(self, linear, num_bits, group_size):
    83:         super().__init__()
    84:         self.linear = linear
    85:         self.num_bits = num_bits
    86:         self.group_size = group_size
    87: 
    88:     @property
    89:     def weight(self):
    90:         return self.linear.weight
    91: 
    92:     @property
    93:     def bias(self):
    94:         return self.linear.bias
    95: 
    96:     def forward(self, x):
    97:         # PTQ-only: in eval the real QDQ has already been applied to
    98:         # linear.weight, so we just call the underlying linear.  During
    99:         # the (zero-step) training phase this is a no-op anyway.
   100:         return F.linear(x, self.linear.weight, self.linear.bias)
   101: 
   102: 
   103: def prepare_qat_model(model, num_bits, group_size):
   104:     from transformers.pytorch_utils import Conv1D
   105: 
   106:     def _replace(parent):
   107:         for name, child in list(parent.named_children()):
   108:             if isinstance(child, nn.Linear):
   109:                 setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
   110:             elif isinstance(child, Conv1D):
   111:                 in_f, out_f = child.weight.shape
   112:                 lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
   113:                                 device=child.weight.device, dtype=child.weight.dtype)
   114:                 with torch.no_grad():
   115:                     lin.weight.copy_(child.weight.t().contiguous())
   116:                     if child.bias is not None:
   117:                         lin.bias.copy_(child.bias)
   118:                 setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
   119:             else:
   120:                 _replace(child)
   121: 
   122:     _replace(model)
   123:     for head_attr in ("lm_head", "embed_out"):
   124:         head = getattr(model, head_attr, None)
   125:         if isinstance(head, QATWrapper):
   126:             setattr(model, head_attr, head.linear)
   127:     return model
   128: 
   129: 
   130: # ═══════════════════════════════════════════════════════════════════════════════
   131: # EDITABLE REGION END
```

### `ste` baseline — editable region  [READ-ONLY — reference implementation]

In `llm-qat-runtime/custom_qat.py`:

```python
Lines 33–129:
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
    71: def quantize_dequantize_weight(weight, num_bits, group_size):
    72:     qmin, qmax = _qrange(num_bits)
    73:     out_features, in_features = weight.shape
    74:     assert in_features % group_size == 0
    75:     with torch.no_grad():
    76:         w = weight.float().reshape(out_features, -1, group_size)
    77:         w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    78:         scale = w_max / qmax
    79:         w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    80:         return w_q.reshape(out_features, in_features).to(weight.dtype)
    81: 
    82: 
    83: class QATWrapper(nn.Module):
    84:     def __init__(self, linear, num_bits, group_size):
    85:         super().__init__()
    86:         self.linear = linear
    87:         self.num_bits = num_bits
    88:         self.group_size = group_size
    89: 
    90:     @property
    91:     def weight(self):
    92:         return self.linear.weight
    93: 
    94:     @property
    95:     def bias(self):
    96:         return self.linear.bias
    97: 
    98:     def forward(self, x):
    99:         x = fake_quantize_activation(x, self.num_bits)
   100:         w_q = fake_quantize_weight(self.linear.weight, self.num_bits, self.group_size)
   101:         return F.linear(x, w_q, self.linear.bias)
   102: 
   103: 
   104: def prepare_qat_model(model, num_bits, group_size):
   105:     from transformers.pytorch_utils import Conv1D
   106: 
   107:     def _replace(parent):
   108:         for name, child in list(parent.named_children()):
   109:             if isinstance(child, nn.Linear):
   110:                 setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
   111:             elif isinstance(child, Conv1D):
   112:                 in_f, out_f = child.weight.shape
   113:                 lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
   114:                                 device=child.weight.device, dtype=child.weight.dtype)
   115:                 with torch.no_grad():
   116:                     lin.weight.copy_(child.weight.t().contiguous())
   117:                     if child.bias is not None:
   118:                         lin.bias.copy_(child.bias)
   119:                 setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
   120:             else:
   121:                 _replace(child)
   122: 
   123:     _replace(model)
   124:     for head_attr in ("lm_head", "embed_out"):
   125:         head = getattr(model, head_attr, None)
   126:         if isinstance(head, QATWrapper):
   127:             setattr(model, head_attr, head.linear)
   128:     return model
   129: 
   130: 
   131: # ═══════════════════════════════════════════════════════════════════════════════
   132: # EDITABLE REGION END
```

### `lsq` baseline — editable region  [READ-ONLY — reference implementation]

In `llm-qat-runtime/custom_qat.py`:

```python
Lines 33–185:
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
   106: def quantize_dequantize_weight(weight, num_bits, group_size):
   107:     # LSQ stores learned scales on the wrapper; the fixed-region
   108:     # `apply_real_quantization` would clobber them if we did our own
   109:     # max-abs QDQ here.  Returning the weight unchanged keeps the float
   110:     # weight intact, and the wrapper applies LSQ-grid QDQ in eval mode
   111:     # below -- so evaluation still sees a properly quantized model.
   112:     return weight.clone()
   113: 
   114: 
   115: class QATWrapper(nn.Module):
   116:     def __init__(self, linear, num_bits, group_size):
   117:         super().__init__()
   118:         self.linear = linear
   119:         self.num_bits = num_bits
   120:         self.group_size = group_size
   121:         qmin, qmax = _qrange(num_bits)
   122:         out_features, in_features = linear.weight.shape
   123:         n_groups = in_features // group_size
   124:         # LSQ initial scale: 2 * |W|.mean() / sqrt(qmax)  (paper Sec. 3.4).
   125:         with torch.no_grad():
   126:             w = linear.weight.float().reshape(out_features, n_groups, group_size)
   127:             init = 2.0 * w.abs().mean(dim=-1, keepdim=True) / max(1.0, math.sqrt(qmax))
   128:             init = init.clamp(min=1e-8)
   129:         # Shape (out_features, n_groups, 1) so it broadcasts over group_size.
   130:         self.lsq_scale = nn.Parameter(init.to(linear.weight.dtype))
   131: 
   132:     @property
   133:     def weight(self):
   134:         return self.linear.weight
   135: 
   136:     @property
   137:     def bias(self):
   138:         return self.linear.bias
   139: 
   140:     def forward(self, x):
   141:         x = fake_quantize_activation(x, self.num_bits)
   142:         if self.training:
   143:             w_q = fake_quantize_weight(
   144:                 self.linear.weight, self.num_bits, self.group_size,
   145:                 scale=self.lsq_scale.float(),
   146:             )
   147:         else:
   148:             # Eval: produce a *real* quantize-dequantize on the LSQ grid.
   149:             qmin, qmax = _qrange(self.num_bits)
   150:             with torch.no_grad():
   151:                 w = self.linear.weight.float().reshape(
   152:                     self.linear.weight.shape[0], -1, self.group_size
   153:                 )
   154:                 s = self.lsq_scale.float()
   155:                 w_q = torch.clamp(torch.round(w / s), qmin, qmax) * s
   156:                 w_q = w_q.reshape_as(self.linear.weight).to(self.linear.weight.dtype)
   157:         return F.linear(x, w_q, self.linear.bias)
   158: 
   159: 
   160: def prepare_qat_model(model, num_bits, group_size):
   161:     from transformers.pytorch_utils import Conv1D
   162: 
   163:     def _replace(parent):
   164:         for name, child in list(parent.named_children()):
   165:             if isinstance(child, nn.Linear):
   166:                 setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
   167:             elif isinstance(child, Conv1D):
   168:                 in_f, out_f = child.weight.shape
   169:                 lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
   170:                                 device=child.weight.device, dtype=child.weight.dtype)
   171:                 with torch.no_grad():
   172:                     lin.weight.copy_(child.weight.t().contiguous())
   173:                     if child.bias is not None:
   174:                         lin.bias.copy_(child.bias)
   175:                 setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
   176:             else:
   177:                 _replace(child)
   178: 
   179:     _replace(model)
   180:     for head_attr in ("lm_head", "embed_out"):
   181:         head = getattr(model, head_attr, None)
   182:         if isinstance(head, QATWrapper):
   183:             setattr(model, head_attr, head.linear)
   184:     return model
   185: 
   186: 
   187: # ═══════════════════════════════════════════════════════════════════════════════
   188: # EDITABLE REGION END
```

### `finetune_then_ptq` baseline — editable region  [READ-ONLY — reference implementation]

In `llm-qat-runtime/custom_qat.py`:

```python
Lines 33–124:
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
    65: def quantize_dequantize_weight(weight, num_bits, group_size):
    66:     qmin, qmax = _qrange(num_bits)
    67:     out_features, in_features = weight.shape
    68:     assert in_features % group_size == 0
    69:     with torch.no_grad():
    70:         w = weight.float().reshape(out_features, -1, group_size)
    71:         w_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    72:         scale = w_max / qmax
    73:         w_q = torch.clamp(torch.round(w / scale), qmin, qmax) * scale
    74:         return w_q.reshape(out_features, in_features).to(weight.dtype)
    75: 
    76: 
    77: class QATWrapper(nn.Module):
    78:     def __init__(self, linear, num_bits, group_size):
    79:         super().__init__()
    80:         self.linear = linear
    81:         self.num_bits = num_bits
    82:         self.group_size = group_size
    83: 
    84:     @property
    85:     def weight(self):
    86:         return self.linear.weight
    87: 
    88:     @property
    89:     def bias(self):
    90:         return self.linear.bias
    91: 
    92:     def forward(self, x):
    93:         # Pure FP forward during training (no fake quant).  At eval time
    94:         # the real QDQ has already been applied to ``linear.weight``, so
    95:         # this still produces the genuine INT-N output.
    96:         return F.linear(x, self.linear.weight, self.linear.bias)
    97: 
    98: 
    99: def prepare_qat_model(model, num_bits, group_size):
   100:     from transformers.pytorch_utils import Conv1D
   101: 
   102:     def _replace(parent):
   103:         for name, child in list(parent.named_children()):
   104:             if isinstance(child, nn.Linear):
   105:                 setattr(parent, name, QATWrapper(child, num_bits=num_bits, group_size=group_size))
   106:             elif isinstance(child, Conv1D):
   107:                 in_f, out_f = child.weight.shape
   108:                 lin = nn.Linear(in_f, out_f, bias=child.bias is not None,
   109:                                 device=child.weight.device, dtype=child.weight.dtype)
   110:                 with torch.no_grad():
   111:                     lin.weight.copy_(child.weight.t().contiguous())
   112:                     if child.bias is not None:
   113:                         lin.bias.copy_(child.bias)
   114:                 setattr(parent, name, QATWrapper(lin, num_bits=num_bits, group_size=group_size))
   115:             else:
   116:                 _replace(child)
   117: 
   118:     _replace(model)
   119:     for head_attr in ("lm_head", "embed_out"):
   120:         head = getattr(model, head_attr, None)
   121:         if isinstance(head, QATWrapper):
   122:             setattr(model, head_attr, head.linear)
   123:     return model
   124: 
   125: 
   126: # ═══════════════════════════════════════════════════════════════════════════════
   127: # EDITABLE REGION END
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
