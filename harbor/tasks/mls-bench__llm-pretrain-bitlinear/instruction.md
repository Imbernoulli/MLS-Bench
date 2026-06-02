# MLS-Bench: llm-pretrain-bitlinear

# LLM Pretraining: Native Low-Bit Linear (BitLinear)

## Research Question
Design a low-bit linear layer for GPT-2 pretraining that uses native low-precision weights (binary / ternary / few-bit) during both training and inference, instead of standard float weights. The goal is to minimize validation loss and preserve downstream language ability while constraining the effective forward weights to a small discrete set.

## Background
Standard neural networks store and compute with full-precision (FP32 / BF16) weights. Post-training quantization (PTQ) and quantization-aware training (QAT) compress these weights after or during training, but the model fundamentally trains with float weights. Native low-bit training takes a different approach: weights are inherently discrete (e.g., {-1, +1} or {-1, 0, +1}) during every forward pass, while a float latent weight is maintained only for gradient accumulation (with a straight-through estimator).

Reference papers:
- **BitNet** — Wang et al., 2023, arXiv:2310.11453, "BitNet: Scaling 1-bit Transformers for Large Language Models". Introduces `BitLinear` as a drop-in replacement for `nn.Linear`, binarizing weights to {-1, +1} via the sign function with per-tensor scale.
- **BitNet b1.58** — Ma et al., 2024, arXiv:2402.17764, "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits". Ternary weights {-1, 0, +1} via absmean quantization (`scale = mean(|W|)`, weights rounded to the nearest of {-1, 0, +1}). Reported to match full-precision LLaMA-style baselines starting around the 3B scale.

Distinction from neighboring tasks:
- **vs. QAT**: QAT keeps float weights during training and only uses fake quantization; BitLinear's forward weights are always discrete.
- **vs. mixed precision**: Mixed precision changes the float format (FP32 → BF16/FP8) but values remain continuous; BitLinear restricts weights to a small discrete set (1–2 bits typically).

## What you can modify
The BitLinear module in `nanoGPT/custom_pretrain.py`:
- `weight_quant(weight)` — quantizes float latent weights to discrete values; returns `(quantized_weight, scale)`.
- `activation_quant(x)` — optional activation quantization; returns `(quantized_x, scale)`.
- `BitLinear` class — linear layer that uses the above functions.

### Interface contract
- `BitLinear.__init__(self, in_features, out_features, bias=True)` must keep `self.weight` as a `Parameter`.
- `BitLinear.forward(self, x) -> output` where `x` has shape `(..., in_features)` and the output has shape `(..., out_features)`.
- Quantization is applied in every forward pass (no separate train/eval path).
- `weight_quant` should return `(quantized_weight, scale)` such that `quantized_weight * scale` approximates the original weight; same convention for `activation_quant`.
- All linear projections in the model (attention, MLP, lm_head) use `BitLinear`.
- Helper classes (`autograd.Function`s, learned parameters) may be added.
- Must remain compatible with `torch.compile` (no `@torch.compiler.disable`).

## Reference baselines (algorithmic templates)
- `binary_sign` — BitNet sign-based binary weights {-1, +1} with absmean scale.
- `ternary_158bit` — BitNet b1.58 ternary {-1, 0, +1} with absmean scale.
- `int2_uniform` — uniform 2-bit quantization grid.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens (Chinchilla-optimal D=20N).
- **Training**: 13,535 iterations, micro-batch 64, gradient accumulation 8, 2-GPU DDP.

## Evaluation
- **Validation loss** — cross-entropy on a held-out FineWeb shard (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/nanoGPT/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `nanoGPT/custom_pretrain.py`
- editable lines **38–115**
- editable lines **328–328**


Other files you may **read** for context (do not modify):
- `nanoGPT/model.py`


## Readable Context


### `nanoGPT/custom_pretrain.py`  [EDITABLE — lines 38–115, lines 328–328 only]

```python
     1: """Custom GPT-2 Pretraining Script with Native Low-Bit Linear (BitLinear)
     2: Based on Andrej Karpathy's nanoGPT, evaluated on FineWeb dataset.
     3: 
     4: This script replaces standard nn.Linear with BitLinear, which uses native
     5: low-precision weights (binary/ternary) during both training and inference,
     6: rather than fake-quantizing float weights (QAT).
     7: """
     8: 
     9: import math
    10: import inspect
    11: import os
    12: import time
    13: import copy
    14: from contextlib import nullcontext
    15: from dataclasses import dataclass
    16: 
    17: import numpy as np
    18: import torch
    19: import torch.nn as nn
    20: from torch.nn import functional as F
    21: 
    22: # ============================================================================
    23: # Model Components
    24: # ============================================================================
    25: 
    26: # -- Normalization ------------------------------------------------------------
    27: class LayerNorm(nn.Module):
    28:     """LayerNorm but with an optional bias."""
    29:     def __init__(self, ndim, bias):
    30:         super().__init__()
    31:         self.weight = nn.Parameter(torch.ones(ndim))
    32:         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    33: 
    34:     def forward(self, input):
    35:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    36: 
    37: # -- Native Low-Bit Linear (BitLinear) Module ---------------------------------
    38: def weight_quant(weight):
    39:     """Quantize weight to low-bit representation for forward pass.
    40: 
    41:     This function maps float latent weights to a discrete low-bit set
    42:     during the forward pass. The backward pass uses the Straight-Through
    43:     Estimator (STE) to flow gradients through the non-differentiable
    44:     quantization.
    45: 
    46:     The default implementation is a pass-through (no quantization).
    47:     Replace this with your low-bit quantization scheme, e.g.:
    48:     - Binary: {-1, +1} via sign function
    49:     - Ternary: {-1, 0, +1} via absmean thresholding
    50:     - 2-bit: {-1, -1/3, +1/3, +1} via uniform quantization
    51: 
    52:     Args:
    53:         weight: float latent weight tensor [out_features, in_features]
    54:     Returns:
    55:         (quantized_weight, scale): quantized weight tensor and a per-tensor
    56:             or per-channel scale factor used to rescale the output.
    57:             quantized_weight * scale should approximate the original weight.
    58:     """
    59:     scale = weight.detach().abs().mean()
    60:     return weight, scale
    61: 
    62: 
    63: def activation_quant(x):
    64:     """Quantize activations for the forward pass (optional).
    65: 
    66:     Some low-bit training schemes also quantize activations to maximize
    67:     the benefit of low-bit compute. The default is a pass-through.
    68: 
    69:     Args:
    70:         x: activation tensor [..., in_features]
    71:     Returns:
    72:         (quantized_x, scale): quantized activation and scale factor.
    73:     """
    74:     scale = x.detach().abs().max().clamp(min=1e-12)
    75:     return x, scale
    76: 
    77: 
    78: class BitLinear(nn.Module):
    79:     """Linear layer with native low-bit weights for training and inference.
    80: 
    81:     Unlike QAT (which fake-quantizes float weights), BitLinear maintains
    82:     float latent weights but applies true low-bit quantization (binary,
    83:     ternary, or higher) in every forward pass. The weight is stored in
    84:     float for gradient updates, but the forward computation uses only the
    85:     quantized discrete values.
    86: 
    87:     Key differences from QAT:
    88:     - QAT: float weights -> fake quantize -> float matmul -> real quantize at deploy
    89:     - BitLinear: float latent weights -> discrete quantize -> scaled matmul (always)
    90: 
    91:     The quantization is applied identically in both training and eval modes
    92:     (no separate train/eval paths needed, unlike QAT).
    93:     """
    94:     def __init__(self, in_features, out_features, bias=True):
    95:         super().__init__()
    96:         self.in_features = in_features
    97:         self.out_features = out_features
    98:         self.weight = nn.Parameter(torch.empty(out_features, in_features))
    99:         if bias:
   100:             self.bias = nn.Parameter(torch.zeros(out_features))
   101:         else:
   102:             self.bias = None
   103:         nn.init.normal_(self.weight, mean=0.0, std=0.02)
   104: 
   105:     def forward(self, x):
   106:         w_q, w_scale = weight_quant(self.weight)
   107:         x_q, x_scale = activation_quant(x)
   108:         # Perform matmul with quantized values, then rescale
   109:         out = F.linear(x_q, w_q, None)
   110:         # Rescale output: the true output ~ (x_scale * w_scale) * out_quantized
   111:         # But since default is pass-through, just add bias
   112:         if self.bias is not None:
   113:             out = out + self.bias
   114:         return out
   115: 
   116: # -- Self-Attention -----------------------------------------------------------
   117: class CausalSelfAttention(nn.Module):
   118:     def __init__(self, config):
   119:         super().__init__()
   120:         assert config.n_embd % config.n_head == 0
   121:         self.c_attn = BitLinear(config.n_embd, 3 * config.n_embd, bias=config.bias)
   122:         self.c_proj = BitLinear(config.n_embd, config.n_embd, bias=config.bias)
   123:         self.attn_dropout = nn.Dropout(config.dropout)
   124:         self.resid_dropout = nn.Dropout(config.dropout)
   125:         self.n_head = config.n_head
   126:         self.n_embd = config.n_embd
   127:         self.dropout = config.dropout
   128:         self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
   129:         if not self.flash:
   130:             self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
   131:                                         .view(1, 1, config.block_size, config.block_size))
   132:         self.use_pos_emb = True
   133: 
   134:     def forward(self, x):
   135:         B, T, C = x.size()
   136:         q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
   137:         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
   138:         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
   139:         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
   140:         if self.flash:
   141:             y = torch.nn.functional.scaled_dot_product_attention(
   142:                 q, k, v, attn_mask=None,
   143:                 dropout_p=self.dropout if self.training else 0, is_causal=True)
   144:         else:
   145:             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
   146:             att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
   147:             att = F.softmax(att, dim=-1)
   148:             att = self.attn_dropout(att)
   149:             y = att @ v
   150:         y = y.transpose(1, 2).contiguous().view(B, T, C)
   151:         y = self.resid_dropout(self.c_proj(y))
   152:         return y
   153: 
   154: # -- Feed-Forward Network ----------------------------------------------------
   155: class MLP(nn.Module):
   156:     def __init__(self, config):
   157:         super().__init__()
   158:         self.c_fc = BitLinear(config.n_embd, 4 * config.n_embd, bias=config.bias)
   159:         self.gelu = nn.GELU()
   160:         self.c_proj = BitLinear(4 * config.n_embd, config.n_embd, bias=config.bias)
   161:         self.dropout = nn.Dropout(config.dropout)
   162: 
   163:     def forward(self, x):
   164:         x = self.c_fc(x)
   165:         x = self.gelu(x)
   166:         x = self.c_proj(x)
   167:         x = self.dropout(x)
   168:         return x
   169: 
   170: # -- Transformer Block -------------------------------------------------------
   171: class Block(nn.Module):
   172:     def __init__(self, config):
   173:         super().__init__()
   174:         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
   175:         self.attn = CausalSelfAttention(config)
   176:         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
   177:         self.mlp = MLP(config)
   178: 
   179:     def forward(self, x):
   180:         x = x + self.attn(self.ln_1(x))
   181:         x = x + self.mlp(self.ln_2(x))
   182:         return x
   183: 
   184: # ============================================================================
   185: # GPT Model
   186: # ============================================================================
   187: 
   188: @dataclass
   189: class GPTConfig:
   190:     block_size: int = 1024
   191:     vocab_size: int = 50304
   192:     n_layer: int = 12
   193:     n_head: int = 12
   194:     n_embd: int = 768
   195:     dropout: float = 0.0
   196:     bias: bool = False
   197: 
   198: class GPT(nn.Module):
   199:     def __init__(self, config):
   200:         super().__init__()
   201:         self.config = config
   202:         self.transformer = nn.ModuleDict(dict(
   203:             wte=nn.Embedding(config.vocab_size, config.n_embd),
   204:             wpe=nn.Embedding(config.block_size, config.n_embd),
   205:             drop=nn.Dropout(config.dropout),
   206:             h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
   207:             ln_f=LayerNorm(config.n_embd, bias=config.bias),
   208:         ))
   209:         self.lm_head = BitLinear(config.n_embd, config.vocab_size, bias=False)
   210:         self.transformer.wte.weight = self.lm_head.weight
   211:         self.apply(self._init_weights)
   212:         for pn, p in self.named_parameters():
   213:             if pn.endswith('c_proj.weight'):
   214:                 torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
   215:         print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
   216: 
   217:     def get_num_params(self, non_embedding=True):
   218:         n_params = sum(p.numel() for p in self.parameters())
   219:         if non_embedding:
   220:             n_params -= self.transformer.wpe.weight.numel()
   221:         return n_params
   222: 
   223:     def _init_weights(self, module):
   224:         if isinstance(module, (nn.Linear, BitLinear)):
   225:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   226:             if module.bias is not None:
   227:                 torch.nn.init.zeros_(module.bias)
   228:         elif isinstance(module, nn.Embedding):
   229:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   230: 
   231:     def forward(self, idx, targets=None):
   232:         device = idx.device
   233:         b, t = idx.size()
   234:         assert t <= self.config.block_size
   235:         tok_emb = self.transformer.wte(idx)
   236:         x = self.transformer.drop(tok_emb)
   237:         use_pos = getattr(self.transformer.h[0].attn, 'use_pos_emb', True)
   238:         if use_pos:
   239:             pos = torch.arange(0, t, dtype=torch.long, device=device)
   240:             x = x + self.transformer.wpe(pos)
   241:         for block in self.transformer.h:
   242:             x = block(x)
   243:         x = self.transformer.ln_f(x)
   244:         if targets is not None:
   245:             logits = self.lm_head(x)
   246:             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
   247:         else:
   248:             logits = self.lm_head(x[:, [-1], :])
   249:             loss = None
   250:         return logits, loss
   251: 
   252:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   253:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   254:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   255:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   256:         optim_groups = [
   257:             {'params': decay_params, 'weight_decay': weight_decay},
   258:             {'params': nodecay_params, 'weight_decay': 0.0},
   259:         ]
   260:         num_decay_params = sum(p.numel() for p in decay_params)
   261:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   262:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   263:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   264:         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   265:         use_fused = fused_available and device_type == 'cuda'
   266:         extra_args = dict(fused=True) if use_fused else dict()
   267:         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
   268:         print(f"using fused AdamW: {use_fused}")
   269:         return optimizer
   270: 
   271: # -- Learning Rate Schedule ---------------------------------------------------
   272: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   273:     """Cosine learning rate schedule with linear warmup."""
   274:     if it < warmup_iters:
   275:         return learning_rate * (it + 1) / (warmup_iters + 1)
   276:     if it > lr_decay_iters:
   277:         return min_lr
   278:     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
   279:     assert 0 <= decay_ratio <= 1
   280:     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
   281:     return min_lr + coeff * (learning_rate - min_lr)
   282: 
   283: # ============================================================================
   284: # Data Loading
   285: # ============================================================================
   286: 
   287: def get_batch(data, batch_size, block_size, device):
   288:     """Get a random batch from a pre-opened memmap (nanoGPT style)."""
   289:     ix = torch.randint(len(data) - block_size, (batch_size,))
   290:     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
   291:     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
   292:     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
   293:     return x, y
   294: 
   295: # ============================================================================
   296: # Training Script
   297: # ============================================================================
   298: 
   299: if __name__ == '__main__':
   300:     # -- Configuration from environment --
   301:     output_dir = os.environ.get('OUTPUT_DIR', 'out')
   302:     seed = int(os.environ.get('SEED', 1337))
   303:     data_dir = os.environ.get('DATA_DIR', '/data/climbmix')
   304: 
   305:     # Model config from environment
   306:     n_layer = int(os.environ.get('N_LAYER', 12))
   307:     n_head = int(os.environ.get('N_HEAD', 12))
   308:     n_embd = int(os.environ.get('N_EMBD', 768))
   309: 
   310:     # Training hyperparameters (overridable via env for different model sizes)
   311:     max_iters = int(os.environ.get('MAX_ITERS', 5000))
   312:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 500))
   313:     eval_iters = 200
   314:     log_interval = 10
   315:     batch_size = int(os.environ.get('BATCH_SIZE', 12))
   316:     block_size = 1024
   317:     gradient_accumulation_steps = int(os.environ.get('GRAD_ACCUM', 5))
   318:     learning_rate = float(os.environ.get('LEARNING_RATE', 6e-4))
   319:     min_lr = learning_rate / 10
   320:     weight_decay = 1e-1
   321:     beta1 = 0.9
   322:     beta2 = 0.95
   323:     grad_clip = 1.0
   324:     warmup_iters = int(max_iters * 0.04)
   325:     lr_decay_iters = max_iters
   326:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   327:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   328:     CONFIG_OVERRIDES = {}
   329: 
   330:     # Apply per-method hyperparameter overrides
   331:     for _k, _v in CONFIG_OVERRIDES.items():
   332:         if _k == 'learning_rate': learning_rate = _v; min_lr = learning_rate / 10
   333:         elif _k == 'weight_decay': weight_decay = _v
   334:         elif _k == 'warmup_iters': warmup_iters = _v
   335:         elif _k == 'min_lr': min_lr = _v
   336:         elif _k == 'grad_clip': grad_clip = _v
   337: 
   338:     compile_model = True
   339:     dtype = 'bfloat16'
   340: 
   341:     # -- DDP Setup --
   342:     ddp = int(os.environ.get('RANK', -1)) != -1
   343:     if ddp:
   344:         import torch.distributed as dist
   345:         from torch.nn.parallel import DistributedDataParallel as DDP
   346:         dist.init_process_group(backend='nccl')
   347:         ddp_rank = int(os.environ['RANK'])
   348:         ddp_local_rank = int(os.environ['LOCAL_RANK'])
   349:         ddp_world_size = int(os.environ['WORLD_SIZE'])
   350:         device = f'cuda:{ddp_local_rank}'
   351:         torch.cuda.set_device(device)
   352:         master_process = ddp_rank == 0
   353:         seed_offset = ddp_rank
   354:         assert gradient_accumulation_steps % ddp_world_size == 0
   355:         gradient_accumulation_steps //= ddp_world_size
   356:     else:
   357:         master_process = True
   358:         device = 'cuda'
   359:         seed_offset = 0
   360: 
   361:     # -- Setup --
   362:     device_type = 'cuda'
   363:     torch.manual_seed(seed + seed_offset)
   364:     torch.backends.cuda.matmul.allow_tf32 = True
   365:     torch.backends.cudnn.allow_tf32 = True
   366:     ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
   367:     ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
   368:     if master_process:
   369:         os.makedirs(output_dir, exist_ok=True)
   370: 
   371:     tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
   372:     if ddp:
   373:         tokens_per_iter *= int(os.environ.get('WORLD_SIZE', 1))
   374:     if master_process:
   375:         print(f"tokens per iteration will be: {tokens_per_iter:,}")
   376: 
   377:     # -- Load Data --
   378:     train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
   379:     val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
   380:     if master_process:
   381:         print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
   382: 
   383:     # -- Model Init --
   384:     model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
   385:                       block_size=block_size, bias=False, vocab_size=50304, dropout=0.0)
   386:     gptconf = GPTConfig(**model_args)
   387:     model = GPT(gptconf)
   388:     model.to(device)
   389: 
   390: 
   391:     scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
   392:     optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
   393: 
   394:     if ddp:
   395:         model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
   396: 
   397:     if compile_model:
   398:         if master_process:
   399:             print("compiling the model...")
   400:         model = torch.compile(model)
   401: 
   402:     # -- Evaluation --
   403:     @torch.no_grad()
   404:     def estimate_loss():
   405:         out = {}
   406:         raw = model.module if ddp else model
   407:         raw_inner = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
   408:         raw_inner.eval()
   409:         for split, data in [('train', train_data), ('val', val_data)]:
   410:             losses = torch.zeros(eval_iters)
   411:             for k in range(eval_iters):
   412:                 X, Y = get_batch(data, batch_size, block_size, device)
   413:                 with ctx:
   414:                     logits, loss = raw_inner(X, Y)
   415:                 losses[k] = loss.item()
   416:             out[split] = losses.mean()
   417:         raw_inner.train()
   418:         return out
   419: 
   420:     # -- Training Loop --
   421:     t0 = time.time()
   422:     best_val_loss = 1e9
   423: 
   424:     for iter_num in range(max_iters + 1):
   425:         lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
   426:         for param_group in optimizer.param_groups:
   427:             param_group['lr'] = lr
   428: 
   429:         if iter_num % eval_interval == 0 and master_process:
   430:             losses = estimate_loss()
   431:             train_loss = losses['train'].item()
   432:             val_loss = losses['val'].item()
   433:             print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
   434:             print(f"TRAIN_METRICS: step={iter_num}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
   435:             if val_loss < best_val_loss:
   436:                 best_val_loss = val_loss
   437: 
   438:         for micro_step in range(gradient_accumulation_steps):
   439:             if ddp:
   440:                 model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
   441:             with ctx:
   442:                 X, Y = get_batch(train_data, batch_size, block_size, device)
   443:                 logits, loss = model(X, Y)
   444:                 loss = loss / gradient_accumulation_steps
   445:             scaler.scale(loss).backward()
   446: 
   447:         if grad_clip != 0.0:
   448:             scaler.unscale_(optimizer)
   449:             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
   450:         scaler.step(optimizer)
   451:         scaler.update()
   452:         optimizer.zero_grad(set_to_none=True)
   453: 
   454:         t1 = time.time()
   455:         dt = t1 - t0
   456:         t0 = t1
   457:         if iter_num % log_interval == 0 and iter_num > 0 and master_process:
   458:             lossf = loss.item() * gradient_accumulation_steps
   459:             print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
   460: 
   461:     # -- Free training state to reclaim GPU memory --
   462:     del optimizer, scaler
   463:     import gc; gc.collect()
   464:     torch.cuda.empty_cache()
   465: 
   466:     # -- Final Evaluation --
   467:     if master_process:
   468:         losses = estimate_loss()
   469:         val_loss = losses['val'].item()
   470:         train_loss = losses['train'].item()
   471:         print(f"Final: train loss {train_loss:.4f}, val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}")
   472: 
   473:         # -- PPL on benchmark datasets --
   474:         eval_dir = os.environ.get('EVAL_DIR', '/data/eval')
   475:         raw = model.module if ddp else model
   476:         raw_inner = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
   477:         raw_inner.eval()
   478:         eval_datasets = ['wikitext2', 'lambada']
   479:         ppl_results = {}
   480:         for ds_name in eval_datasets:
   481:             ds_path = os.path.join(eval_dir, f'{ds_name}.bin')
   482:             if not os.path.exists(ds_path):
   483:                 print(f"Eval dataset not found: {ds_path}")
   484:                 continue
   485:             data = np.memmap(ds_path, dtype=np.uint16, mode='r')
   486:             n_tokens = len(data)
   487:             total_loss = 0.0
   488:             n_chunks = 0
   489:             with torch.no_grad():
   490:                 for start in range(0, n_tokens - block_size, block_size):
   491:                     x = torch.from_numpy(data[start:start+block_size].astype(np.int64)).unsqueeze(0).to(device)
   492:                     y = torch.from_numpy(data[start+1:start+1+block_size].astype(np.int64)).unsqueeze(0).to(device)
   493:                     with ctx:
   494:                         _, loss = raw_inner(x, y)
   495:                     total_loss += loss.item()
   496:                     n_chunks += 1
   497:             avg_loss = total_loss / n_chunks
   498:             ppl = math.exp(avg_loss)
   499:             ppl_results[ds_name] = ppl
   500:             print(f"PPL {ds_name}: {ppl:.2f} (avg_loss={avg_loss:.4f}, {n_chunks} chunks)")

[truncated: showing at most 500 lines / 60000 bytes from nanoGPT/custom_pretrain.py]
```

## Parameter Budget

This task enforces a parameter-count cap. Your edits will be rejected if
the resulting model exceeds **1.05×** the strongest
baseline's parameter count. The check runs automatically inside the eval
scripts — you don't need to invoke it.

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `binary_sign` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 38–90:
    35:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    36: 
    37: # -- Native Low-Bit Linear (BitLinear) Module ---------------------------------
    38: def weight_quant(weight):
    39:     """Binary quantization: sign(W) * mean(|W|) with STE.
    40: 
    41:     Forward: w_q = sign(W), scale = mean(|W|)
    42:     Backward: STE (gradient passes through sign as identity)
    43:     """
    44:     scale = weight.detach().abs().mean()
    45:     # STE: forward uses sign, backward treats sign as identity
    46:     w_q = (weight.sign() - weight).detach() + weight
    47:     return w_q, scale
    48: 
    49: 
    50: def activation_quant(x):
    51:     """Absmax 8-bit activation quantization with STE.
    52: 
    53:     Quantizes activations to 127 levels (int8 range) using per-tensor
    54:     absmax scaling, following the original BitNet paper.
    55:     """
    56:     Qb = 127  # int8 range
    57:     scale = x.detach().abs().max().clamp(min=1e-12)
    58:     x_normed = x / scale
    59:     x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    60:     # STE: forward uses quantized, backward passes through
    61:     x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    62:     return x_q, scale / Qb
    63: 
    64: 
    65: class BitLinear(nn.Module):
    66:     """BitNet linear layer with binary {-1, +1} weights.
    67: 
    68:     During both training and eval: weights are binarized via sign function,
    69:     activations are quantized to int8 range. Output is rescaled by
    70:     weight_scale * activation_scale.
    71:     """
    72:     def __init__(self, in_features, out_features, bias=True):
    73:         super().__init__()
    74:         self.in_features = in_features
    75:         self.out_features = out_features
    76:         self.weight = nn.Parameter(torch.empty(out_features, in_features))
    77:         if bias:
    78:             self.bias = nn.Parameter(torch.zeros(out_features))
    79:         else:
    80:             self.bias = None
    81:         nn.init.normal_(self.weight, mean=0.0, std=0.02)
    82: 
    83:     def forward(self, x):
    84:         w_q, w_scale = weight_quant(self.weight)
    85:         x_q, x_scale = activation_quant(x)
    86:         out = F.linear(x_q, w_q, None)
    87:         out = out * (w_scale * x_scale)
    88:         if self.bias is not None:
    89:             out = out + self.bias
    90:         return out
    91: # -- Self-Attention -----------------------------------------------------------
    92: class CausalSelfAttention(nn.Module):
    93:     def __init__(self, config):

Lines 303–303:
   300:     lr_decay_iters = max_iters
   301:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   302:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   303:     CONFIG_OVERRIDES = {}
   304: 
   305:     # Apply per-method hyperparameter overrides
   306:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `ternary_158bit` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 38–92:
    35:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    36: 
    37: # -- Native Low-Bit Linear (BitLinear) Module ---------------------------------
    38: def weight_quant(weight):
    39:     """Ternary quantization: {-1, 0, +1} via absmean with STE.
    40: 
    41:     Forward: normalize by absmean, round-then-clip to {-1, 0, +1}
    42:     Backward: STE (gradient passes through rounding as identity)
    43:     """
    44:     scale = weight.detach().abs().mean().clamp(min=1e-12)
    45:     w_normed = weight / scale
    46:     # STE round: (round(x) - x).detach() + x
    47:     w_q = w_normed.clamp(-1, 1)
    48:     w_q = (w_q.round() - w_q).detach() + w_q
    49:     return w_q, scale
    50: 
    51: 
    52: def activation_quant(x):
    53:     """Absmax 8-bit activation quantization with STE.
    54: 
    55:     Quantizes activations to 127 levels (int8 range) using per-tensor
    56:     absmax scaling, following the BitNet b1.58 paper.
    57:     """
    58:     Qb = 127  # int8 range
    59:     scale = x.detach().abs().max().clamp(min=1e-12)
    60:     x_normed = x / scale
    61:     x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    62:     # STE: forward uses quantized, backward passes through
    63:     x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    64:     return x_q, scale / Qb
    65: 
    66: 
    67: class BitLinear(nn.Module):
    68:     """BitNet b1.58 linear layer with ternary {-1, 0, +1} weights.
    69: 
    70:     During both training and eval: weights are ternarized via absmean
    71:     + round-clip, activations are quantized to int8 range. Output is
    72:     rescaled by weight_scale * activation_scale.
    73:     """
    74:     def __init__(self, in_features, out_features, bias=True):
    75:         super().__init__()
    76:         self.in_features = in_features
    77:         self.out_features = out_features
    78:         self.weight = nn.Parameter(torch.empty(out_features, in_features))
    79:         if bias:
    80:             self.bias = nn.Parameter(torch.zeros(out_features))
    81:         else:
    82:             self.bias = None
    83:         nn.init.normal_(self.weight, mean=0.0, std=0.02)
    84: 
    85:     def forward(self, x):
    86:         w_q, w_scale = weight_quant(self.weight)
    87:         x_q, x_scale = activation_quant(x)
    88:         out = F.linear(x_q, w_q, None)
    89:         out = out * (w_scale * x_scale)
    90:         if self.bias is not None:
    91:             out = out + self.bias
    92:         return out
    93: # -- Self-Attention -----------------------------------------------------------
    94: class CausalSelfAttention(nn.Module):
    95:     def __init__(self, config):

Lines 305–305:
   302:     lr_decay_iters = max_iters
   303:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   304:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   305:     CONFIG_OVERRIDES = {}
   306: 
   307:     # Apply per-method hyperparameter overrides
   308:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `int2_uniform` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 38–101:
    35:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    36: 
    37: # -- Native Low-Bit Linear (BitLinear) Module ---------------------------------
    38: def weight_quant(weight):
    39:     """2-bit uniform quantization: {-1, -1/3, +1/3, +1} with STE.
    40: 
    41:     Normalizes weights by absmean, maps to 4 uniform levels in [-1, 1],
    42:     then rescales. Uses STE for gradient flow through rounding.
    43:     """
    44:     scale = weight.detach().abs().mean().clamp(min=1e-12)
    45:     w_normed = weight / scale
    46:     # Map to [-1.5, 1.5] grid with spacing 1.0, round, then map back
    47:     # Levels: -1.5 -> -1, -0.5 -> -1/3, 0.5 -> 1/3, 1.5 -> 1
    48:     # Multiply by 1.5 so that [-1,1] -> [-1.5,1.5], round, clip to {-1,0,1} range
    49:     # Actually: use 4 uniform levels directly
    50:     # Grid points at: -1, -1/3, 1/3, 1 (spacing = 2/3)
    51:     # Scale so spacing becomes 1: multiply by 3/2
    52:     w_scaled = w_normed * 1.5  # now grid at -1.5, -0.5, 0.5, 1.5
    53:     w_rounded = w_scaled.clamp(-2, 2).round().clamp(-1.5, 1.5)
    54:     # STE: (rounded - scaled).detach() + scaled
    55:     w_q = (w_rounded - w_scaled).detach() + w_scaled
    56:     # Map back: divide by 1.5
    57:     w_q = w_q / 1.5
    58:     return w_q, scale
    59: 
    60: 
    61: def activation_quant(x):
    62:     """Absmax 8-bit activation quantization with STE.
    63: 
    64:     Quantizes activations to 127 levels (int8 range) using per-tensor
    65:     absmax scaling.
    66:     """
    67:     Qb = 127  # int8 range
    68:     scale = x.detach().abs().max().clamp(min=1e-12)
    69:     x_normed = x / scale
    70:     x_q = (x_normed * Qb).round().clamp(-Qb, Qb)
    71:     # STE: forward uses quantized, backward passes through
    72:     x_q = (x_q - x_normed * Qb).detach() + x_normed * Qb
    73:     return x_q, scale / Qb
    74: 
    75: 
    76: class BitLinear(nn.Module):
    77:     """Linear layer with 2-bit uniform weight quantization.
    78: 
    79:     Weights are quantized to {-1, -1/3, +1/3, +1} during both training
    80:     and eval. Activations quantized to int8 range. Output rescaled by
    81:     weight_scale * activation_scale.
    82:     """
    83:     def __init__(self, in_features, out_features, bias=True):
    84:         super().__init__()
    85:         self.in_features = in_features
    86:         self.out_features = out_features
    87:         self.weight = nn.Parameter(torch.empty(out_features, in_features))
    88:         if bias:
    89:             self.bias = nn.Parameter(torch.zeros(out_features))
    90:         else:
    91:             self.bias = None
    92:         nn.init.normal_(self.weight, mean=0.0, std=0.02)
    93: 
    94:     def forward(self, x):
    95:         w_q, w_scale = weight_quant(self.weight)
    96:         x_q, x_scale = activation_quant(x)
    97:         out = F.linear(x_q, w_q, None)
    98:         out = out * (w_scale * x_scale)
    99:         if self.bias is not None:
   100:             out = out + self.bias
   101:         return out
   102: # -- Self-Attention -----------------------------------------------------------
   103: class CausalSelfAttention(nn.Module):
   104:     def __init__(self, config):

Lines 314–314:
   311:     lr_decay_iters = max_iters
   312:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   313:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   314:     CONFIG_OVERRIDES = {}
   315: 
   316:     # Apply per-method hyperparameter overrides
   317:     for _k, _v in CONFIG_OVERRIDES.items():
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
