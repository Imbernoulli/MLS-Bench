# MLS-Bench: llm-pretrain-linear-attention

# LLM Pretraining: Linear / Subquadratic Attention Mechanism

## Research Question
Design a linear or otherwise subquadratic sequence-mixing mechanism for GPT-style language model pretraining that remains competitive in language-model quality with standard quadratic softmax attention. The mechanism should scale better than O(n²) in sequence length.

## Background
Standard transformer attention has O(n²) compute and memory in sequence length. A growing body of work proposes subquadratic alternatives that retain transformer-level quality on language modeling:

- **RetNet** — Sun et al., 2023, arXiv:2307.08621, "Retentive Network: A Successor to Transformer for Large Language Models". Retention with parallel / recurrent / chunkwise-recurrent dual forms.
- **GLA** (Gated Linear Attention) — Yang et al., 2023, arXiv:2312.06635, "Gated Linear Attention Transformers with Hardware-Efficient Training". Data-dependent gating + FlashLinearAttention kernels.
- **Mamba** — Gu & Dao, 2023, arXiv:2312.00752, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces". Selective SSM with input-dependent parameters.
- **RWKV-6 (Finch)** — Peng et al., 2024, arXiv:2404.05892, "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence". Multi-headed matrix-valued state, dynamic recurrence.
- **DeltaNet** — Yang et al., 2024, arXiv:2406.06484, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length". Delta-rule update with hardware-efficient parallel training.

## What you can modify
Two editable regions in `nanoGPT/custom_pretrain.py`:

1. **`CausalSelfAttention` class** — the attention computation itself, including:
   - Replacing softmax attention with linear / subquadratic alternatives.
   - Feature maps, gating mechanisms, decay factors.
   - Q/K/V projections and transformations.
   - Internal state management (recurrent state, convolutions, etc.).

2. **`Block` class** — the transformer block structure, including:
   - How attention and MLP sublayers are composed.
   - Normalization placement (pre-norm, post-norm).
   - Residual-connection patterns required to make the mechanism train stably.

### Tooling notes
- The `flash-linear-attention` (FLA) library is pre-installed and provides 27+ optimized linear-attention layers with Triton kernels (`fla.layers.GatedLinearAttention`, `DeltaNet`, `MultiScaleRetention`, `LinearAttention`, `HGRN2`, `Mamba2`, …). You may import from FLA or implement your own mechanism from scratch.
- If your attention does not use learned absolute position embeddings, set `self.use_pos_emb = False` in `__init__`; the model then skips adding `wpe` in the forward pass.
- `torch.compile` is disabled for this task because FLA's Triton kernels are not compatible with it.

## Reference baselines
- `gla` — Gated Linear Attention.
- `retnet` — Retentive Network / MultiScaleRetention.
- `deltanet` — DeltaNet (delta-rule linear attention).

## Fixed Pipeline
- The dataset, tokenizer, training schedule, evaluation code, and unrelated objectives are out of scope (fixed by the harness).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/nanoGPT/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `nanoGPT/custom_pretrain.py`
- editable lines **33–70**
- editable lines **88–100**
- editable lines **246–248**




## Readable Context


### `nanoGPT/custom_pretrain.py`  [EDITABLE — lines 33–70, lines 88–100, lines 246–248 only]

```python
     1: """Custom GPT-2 Pretraining Script
     2: Based on Andrej Karpathy's nanoGPT, evaluated on FineWeb dataset.
     3: # flash-linear-attention is available: from fla.layers import GatedLinearAttention, DeltaNet, MultiScaleRetention, etc.
     4: """
     5: 
     6: import math
     7: import inspect
     8: import os
     9: import time
    10: from contextlib import nullcontext
    11: from dataclasses import dataclass
    12: 
    13: import numpy as np
    14: import torch
    15: import torch.nn as nn
    16: from torch.nn import functional as F
    17: 
    18: # ============================================================================
    19: # Model Components
    20: # ============================================================================
    21: 
    22: # ── Normalization ──────────────────────────────────────────────────────────
    23: class LayerNorm(nn.Module):
    24:     """LayerNorm but with an optional bias."""
    25:     def __init__(self, ndim, bias):
    26:         super().__init__()
    27:         self.weight = nn.Parameter(torch.ones(ndim))
    28:         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    29: 
    30:     def forward(self, input):
    31:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    32: 
    33: # ── Self-Attention ─────────────────────────────────────────────────────────
    34: class CausalSelfAttention(nn.Module):
    35:     def __init__(self, config):
    36:         super().__init__()
    37:         assert config.n_embd % config.n_head == 0
    38:         self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    39:         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    40:         self.attn_dropout = nn.Dropout(config.dropout)
    41:         self.resid_dropout = nn.Dropout(config.dropout)
    42:         self.n_head = config.n_head
    43:         self.n_embd = config.n_embd
    44:         self.dropout = config.dropout
    45:         self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    46:         if not self.flash:
    47:             self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
    48:                                         .view(1, 1, config.block_size, config.block_size))
    49:         # Set to False if using custom position encoding (e.g. RoPE)
    50:         self.use_pos_emb = True
    51: 
    52:     def forward(self, x):
    53:         B, T, C = x.size()
    54:         q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    55:         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    56:         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    57:         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    58:         if self.flash:
    59:             y = torch.nn.functional.scaled_dot_product_attention(
    60:                 q, k, v, attn_mask=None,
    61:                 dropout_p=self.dropout if self.training else 0, is_causal=True)
    62:         else:
    63:             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    64:             att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    65:             att = F.softmax(att, dim=-1)
    66:             att = self.attn_dropout(att)
    67:             y = att @ v
    68:         y = y.transpose(1, 2).contiguous().view(B, T, C)
    69:         y = self.resid_dropout(self.c_proj(y))
    70:         return y
    71: 
    72: # ── Feed-Forward Network ──────────────────────────────────────────────────
    73: class MLP(nn.Module):
    74:     def __init__(self, config):
    75:         super().__init__()
    76:         self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
    77:         self.gelu = nn.GELU()
    78:         self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    79:         self.dropout = nn.Dropout(config.dropout)
    80: 
    81:     def forward(self, x):
    82:         x = self.c_fc(x)
    83:         x = self.gelu(x)
    84:         x = self.c_proj(x)
    85:         x = self.dropout(x)
    86:         return x
    87: 
    88: # ── Transformer Block ─────────────────────────────────────────────────────
    89: class Block(nn.Module):
    90:     def __init__(self, config):
    91:         super().__init__()
    92:         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
    93:         self.attn = CausalSelfAttention(config)
    94:         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
    95:         self.mlp = MLP(config)
    96: 
    97:     def forward(self, x):
    98:         x = x + self.attn(self.ln_1(x))
    99:         x = x + self.mlp(self.ln_2(x))
   100:         return x
   101: 
   102: # ============================================================================
   103: # GPT Model
   104: # ============================================================================
   105: 
   106: @dataclass
   107: class GPTConfig:
   108:     block_size: int = 1024
   109:     vocab_size: int = 50304
   110:     n_layer: int = 12
   111:     n_head: int = 12
   112:     n_embd: int = 768
   113:     dropout: float = 0.0
   114:     bias: bool = False
   115: 
   116: class GPT(nn.Module):
   117:     def __init__(self, config):
   118:         super().__init__()
   119:         self.config = config
   120:         self.transformer = nn.ModuleDict(dict(
   121:             wte=nn.Embedding(config.vocab_size, config.n_embd),
   122:             wpe=nn.Embedding(config.block_size, config.n_embd),
   123:             drop=nn.Dropout(config.dropout),
   124:             h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
   125:             ln_f=LayerNorm(config.n_embd, bias=config.bias),
   126:         ))
   127:         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
   128:         self.transformer.wte.weight = self.lm_head.weight
   129:         self.apply(self._init_weights)
   130:         for pn, p in self.named_parameters():
   131:             if pn.endswith('c_proj.weight'):
   132:                 torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
   133:         print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
   134: 
   135:     def get_num_params(self, non_embedding=True):
   136:         n_params = sum(p.numel() for p in self.parameters())
   137:         if non_embedding:
   138:             n_params -= self.transformer.wpe.weight.numel()
   139:         return n_params
   140: 
   141:     def _init_weights(self, module):
   142:         if isinstance(module, nn.Linear):
   143:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   144:             if module.bias is not None:
   145:                 torch.nn.init.zeros_(module.bias)
   146:         elif isinstance(module, nn.Embedding):
   147:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   148: 
   149:     def forward(self, idx, targets=None):
   150:         device = idx.device
   151:         b, t = idx.size()
   152:         assert t <= self.config.block_size
   153:         tok_emb = self.transformer.wte(idx)
   154:         x = self.transformer.drop(tok_emb)
   155:         # Conditionally add learned position embeddings
   156:         use_pos = getattr(self.transformer.h[0].attn, 'use_pos_emb', True)
   157:         if use_pos:
   158:             pos = torch.arange(0, t, dtype=torch.long, device=device)
   159:             x = x + self.transformer.wpe(pos)
   160:         for block in self.transformer.h:
   161:             x = block(x)
   162:         x = self.transformer.ln_f(x)
   163:         if targets is not None:
   164:             logits = self.lm_head(x)
   165:             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
   166:         else:
   167:             logits = self.lm_head(x[:, [-1], :])
   168:             loss = None
   169:         return logits, loss
   170: 
   171:     # ── Optimizer Configuration ────────────────────────────────────────────
   172:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   173:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   174:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   175:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   176:         optim_groups = [
   177:             {'params': decay_params, 'weight_decay': weight_decay},
   178:             {'params': nodecay_params, 'weight_decay': 0.0},
   179:         ]
   180:         num_decay_params = sum(p.numel() for p in decay_params)
   181:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   182:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   183:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   184:         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   185:         use_fused = fused_available and device_type == 'cuda'
   186:         extra_args = dict(fused=True) if use_fused else dict()
   187:         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
   188:         print(f"using fused AdamW: {use_fused}")
   189:         return optimizer
   190: 
   191: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   192: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   193:     """Cosine learning rate schedule with linear warmup."""
   194:     if it < warmup_iters:
   195:         return learning_rate * (it + 1) / (warmup_iters + 1)
   196:     if it > lr_decay_iters:
   197:         return min_lr
   198:     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
   199:     assert 0 <= decay_ratio <= 1
   200:     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
   201:     return min_lr + coeff * (learning_rate - min_lr)
   202: 
   203: # ============================================================================
   204: # Data Loading
   205: # ============================================================================
   206: 
   207: def get_batch(data, batch_size, block_size, device):
   208:     """Get a random batch from a pre-opened memmap (nanoGPT style)."""
   209:     ix = torch.randint(len(data) - block_size, (batch_size,))
   210:     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
   211:     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
   212:     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
   213:     return x, y
   214: 
   215: # ============================================================================
   216: # Training Script
   217: # ============================================================================
   218: 
   219: if __name__ == '__main__':
   220:     # ── Configuration from environment ──
   221:     output_dir = os.environ.get('OUTPUT_DIR', 'out')
   222:     seed = int(os.environ.get('SEED', 1337))
   223:     data_dir = os.environ.get('DATA_DIR', '/data/climbmix')
   224: 
   225:     # Model config from environment
   226:     n_layer = int(os.environ.get('N_LAYER', 12))
   227:     n_head = int(os.environ.get('N_HEAD', 12))
   228:     n_embd = int(os.environ.get('N_EMBD', 768))
   229: 
   230:     # Training hyperparameters (overridable via env for different model sizes)
   231:     max_iters = int(os.environ.get('MAX_ITERS', 5000))
   232:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 500))
   233:     eval_iters = 200
   234:     log_interval = 10
   235:     batch_size = int(os.environ.get('BATCH_SIZE', 12))
   236:     block_size = 1024
   237:     gradient_accumulation_steps = int(os.environ.get('GRAD_ACCUM', 5))
   238:     learning_rate = float(os.environ.get('LEARNING_RATE', 6e-4))
   239:     min_lr = learning_rate / 10
   240:     weight_decay = 1e-1
   241:     beta1 = 0.9
   242:     beta2 = 0.95
   243:     grad_clip = 1.0
   244:     warmup_iters = int(max_iters * 0.04)
   245:     lr_decay_iters = max_iters
   246:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   247:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   248:     CONFIG_OVERRIDES = {}
   249: 
   250:     # Apply per-method hyperparameter overrides
   251:     for _k, _v in CONFIG_OVERRIDES.items():
   252:         if _k == 'learning_rate': learning_rate = _v; min_lr = learning_rate / 10
   253:         elif _k == 'weight_decay': weight_decay = _v
   254:         elif _k == 'warmup_iters': warmup_iters = _v
   255:         elif _k == 'min_lr': min_lr = _v
   256:         elif _k == 'grad_clip': grad_clip = _v
   257: 
   258:     compile_model = True
   259:     dtype = 'bfloat16'
   260: 
   261:     # ── DDP Setup ──
   262:     ddp = int(os.environ.get('RANK', -1)) != -1
   263:     if ddp:
   264:         import torch.distributed as dist
   265:         from torch.nn.parallel import DistributedDataParallel as DDP
   266:         dist.init_process_group(backend='nccl')
   267:         ddp_rank = int(os.environ['RANK'])
   268:         ddp_local_rank = int(os.environ['LOCAL_RANK'])
   269:         ddp_world_size = int(os.environ['WORLD_SIZE'])
   270:         device = f'cuda:{ddp_local_rank}'
   271:         torch.cuda.set_device(device)
   272:         master_process = ddp_rank == 0
   273:         seed_offset = ddp_rank
   274:         assert gradient_accumulation_steps % ddp_world_size == 0
   275:         gradient_accumulation_steps //= ddp_world_size
   276:     else:
   277:         master_process = True
   278:         device = 'cuda'
   279:         seed_offset = 0
   280: 
   281:     # ── Setup ──
   282:     device_type = 'cuda'
   283:     torch.manual_seed(seed + seed_offset)
   284:     torch.backends.cuda.matmul.allow_tf32 = True
   285:     torch.backends.cudnn.allow_tf32 = True
   286:     ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
   287:     ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
   288:     if master_process:
   289:         os.makedirs(output_dir, exist_ok=True)
   290: 
   291:     tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
   292:     if ddp:
   293:         tokens_per_iter *= int(os.environ.get('WORLD_SIZE', 1))
   294:     if master_process:
   295:         print(f"tokens per iteration will be: {tokens_per_iter:,}")
   296: 
   297:     # ── Load Data ──
   298:     train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
   299:     val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
   300:     if master_process:
   301:         print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
   302: 
   303:     # ── Model Init ──
   304:     model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
   305:                       block_size=block_size, bias=False, vocab_size=50304, dropout=0.0)
   306:     gptconf = GPTConfig(**model_args)
   307:     model = GPT(gptconf)
   308:     model.to(device)
   309: 
   310: 
   311:     scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
   312:     optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
   313: 
   314:     if compile_model:
   315:         if master_process:
   316:             print("compiling the model...")
   317:         model = torch.compile(model)
   318: 
   319:     if ddp:
   320:         model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
   321: 
   322:     # ── Evaluation ──
   323:     @torch.no_grad()
   324:     def estimate_loss():
   325:         out = {}
   326:         raw = model.module if ddp else model
   327:         raw.eval()
   328:         for split, data in [('train', train_data), ('val', val_data)]:
   329:             losses = torch.zeros(eval_iters)
   330:             for k in range(eval_iters):
   331:                 X, Y = get_batch(data, batch_size, block_size, device)
   332:                 with ctx:
   333:                     logits, loss = raw(X, Y)
   334:                 losses[k] = loss.item()
   335:             out[split] = losses.mean()
   336:         raw.train()
   337:         return out
   338: 
   339:     # ── Training Loop ──
   340:     t0 = time.time()
   341:     best_val_loss = 1e9
   342: 
   343:     for iter_num in range(max_iters + 1):
   344:         lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
   345:         for param_group in optimizer.param_groups:
   346:             param_group['lr'] = lr
   347: 
   348:         if iter_num % eval_interval == 0 and master_process:
   349:             losses = estimate_loss()
   350:             train_loss = losses['train'].item()
   351:             val_loss = losses['val'].item()
   352:             print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
   353:             print(f"TRAIN_METRICS: step={iter_num}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
   354:             if val_loss < best_val_loss:
   355:                 best_val_loss = val_loss
   356: 
   357:         for micro_step in range(gradient_accumulation_steps):
   358:             if ddp:
   359:                 model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
   360:             with ctx:
   361:                 X, Y = get_batch(train_data, batch_size, block_size, device)
   362:                 logits, loss = model(X, Y)
   363:                 loss = loss / gradient_accumulation_steps
   364:             scaler.scale(loss).backward()
   365: 
   366:         if grad_clip != 0.0:
   367:             scaler.unscale_(optimizer)
   368:             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
   369:         scaler.step(optimizer)
   370:         scaler.update()
   371:         optimizer.zero_grad(set_to_none=True)
   372: 
   373:         t1 = time.time()
   374:         dt = t1 - t0
   375:         t0 = t1
   376:         if iter_num % log_interval == 0 and iter_num > 0 and master_process:
   377:             lossf = loss.item() * gradient_accumulation_steps
   378:             print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
   379: 
   380:     # ── Free training state to reclaim GPU memory ──
   381:     del optimizer, scaler
   382:     import gc; gc.collect()
   383:     torch.cuda.empty_cache()
   384: 
   385:     # ── Final Evaluation ──
   386:     if master_process:
   387:         losses = estimate_loss()
   388:         val_loss = losses['val'].item()
   389:         train_loss = losses['train'].item()
   390:         print(f"Final: train loss {train_loss:.4f}, val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}")
   391: 
   392:         # ── PPL on benchmark datasets ──
   393:         eval_dir = os.environ.get('EVAL_DIR', '/data/eval')
   394:         raw = model.module if ddp else model
   395:         raw.eval()
   396:         eval_datasets = ['wikitext2', 'lambada']
   397:         ppl_results = {}
   398:         for ds_name in eval_datasets:
   399:             ds_path = os.path.join(eval_dir, f'{ds_name}.bin')
   400:             if not os.path.exists(ds_path):
   401:                 print(f"Eval dataset not found: {ds_path}")
   402:                 continue
   403:             data = np.memmap(ds_path, dtype=np.uint16, mode='r')
   404:             n_tokens = len(data)
   405:             # Process in non-overlapping chunks of block_size
   406:             total_loss = 0.0
   407:             n_chunks = 0
   408:             with torch.no_grad():
   409:                 for start in range(0, n_tokens - block_size, block_size):
   410:                     x = torch.from_numpy(data[start:start+block_size].astype(np.int64)).unsqueeze(0).to(device)
   411:                     y = torch.from_numpy(data[start+1:start+1+block_size].astype(np.int64)).unsqueeze(0).to(device)
   412:                     with ctx:
   413:                         _, loss = raw(x, y)
   414:                     total_loss += loss.item()
   415:                     n_chunks += 1
   416:             avg_loss = total_loss / n_chunks
   417:             ppl = math.exp(avg_loss)
   418:             ppl_results[ds_name] = ppl
   419:             print(f"PPL {ds_name}: {ppl:.2f} (avg_loss={avg_loss:.4f}, {n_chunks} chunks)")
   420: 
   421:         ppl_str = ', '.join(f'{k}_ppl={v:.2f}' for k, v in ppl_results.items())
   422:         print(f"TEST_METRICS: val_loss={val_loss:.4f}, {ppl_str}", flush=True)
   423: 
   424:         # ── Save checkpoint for downstream evaluation (lm-eval-harness) ──
   425:         import shutil
   426:         env_label = os.environ.get('ENV', 'model')
   427:         # Unwrap torch.compile to get clean state_dict keys
   428:         save_model = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
   429:         ckpt_data = {'model_state_dict': save_model.state_dict(), 'model_args': model_args}
   430:         ckpt_path = os.path.join(output_dir, f'ckpt_{env_label}.pt')
   431:         torch.save(ckpt_data, ckpt_path)
   432:         print(f"Checkpoint saved to {ckpt_path}")
   433:         src_path = os.path.join(output_dir, f'model_source_{env_label}.py')
   434:         shutil.copy2(os.path.abspath(__file__), src_path)
   435:         print(f"Model source saved to {src_path}")
   436: 
   437:     if ddp:
   438:         dist.destroy_process_group()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `retnet` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 33–49:
    30:     def forward(self, input):
    31:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    32: 
    33: class CausalSelfAttention(nn.Module):
    34:     def __init__(self, config):
    35:         super().__init__()
    36:         from fla.layers import MultiScaleRetention
    37:         self.attn = MultiScaleRetention(
    38:             hidden_size=config.n_embd,
    39:             num_heads=config.n_head,
    40:             expand_k=1.0,
    41:             expand_v=1.0,
    42:             use_output_gate=True,
    43:             gate_fn='swish',
    44:         )
    45:         self.use_pos_emb = False
    46: 
    47:     def forward(self, x):
    48:         o, _, _ = self.attn(x)
    49:         return o
    50: 
    51: # ── Feed-Forward Network ──────────────────────────────────────────────────
    52: class MLP(nn.Module):

Lines 67–78:
    64:         x = self.dropout(x)
    65:         return x
    66: 
    67: class Block(nn.Module):
    68:     def __init__(self, config):
    69:         super().__init__()
    70:         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
    71:         self.attn = CausalSelfAttention(config)
    72:         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
    73:         self.mlp = MLP(config)
    74: 
    75:     def forward(self, x):
    76:         x = x + self.attn(self.ln_1(x))
    77:         x = x + self.mlp(self.ln_2(x))
    78:         return x
    79: 
    80: # ============================================================================
    81: # GPT Model

Lines 224–226:
   221:     grad_clip = 1.0
   222:     warmup_iters = int(max_iters * 0.04)
   223:     lr_decay_iters = max_iters
   224:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   225:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   226:     CONFIG_OVERRIDES = {}
   227: 
   228:     # Apply per-method hyperparameter overrides
   229:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `deltanet` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 33–50:
    30:     def forward(self, input):
    31:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    32: 
    33: class CausalSelfAttention(nn.Module):
    34:     def __init__(self, config):
    35:         super().__init__()
    36:         from fla.layers import DeltaNet
    37:         self.attn = DeltaNet(
    38:             hidden_size=config.n_embd,
    39:             num_heads=config.n_head,
    40:             use_beta=True,
    41:             use_short_conv=True,
    42:             conv_size=4,
    43:             qk_activation='silu',
    44:             qk_norm='l2',
    45:         )
    46:         self.use_pos_emb = False
    47: 
    48:     def forward(self, x):
    49:         o, _, _ = self.attn(x)
    50:         return o
    51: 
    52: # ── Feed-Forward Network ──────────────────────────────────────────────────
    53: class MLP(nn.Module):

Lines 68–79:
    65:         x = self.dropout(x)
    66:         return x
    67: 
    68: class Block(nn.Module):
    69:     def __init__(self, config):
    70:         super().__init__()
    71:         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
    72:         self.attn = CausalSelfAttention(config)
    73:         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
    74:         self.mlp = MLP(config)
    75: 
    76:     def forward(self, x):
    77:         x = x + self.attn(self.ln_1(x))
    78:         x = x + self.mlp(self.ln_2(x))
    79:         return x
    80: 
    81: # ============================================================================
    82: # GPT Model

Lines 225–227:
   222:     grad_clip = 1.0
   223:     warmup_iters = int(max_iters * 0.04)
   224:     lr_decay_iters = max_iters
   225:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   226:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   227:     CONFIG_OVERRIDES = {}
   228: 
   229:     # Apply per-method hyperparameter overrides
   230:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `gla` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 33–54:
    30:     def forward(self, input):
    31:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    32: 
    33: class CausalSelfAttention(nn.Module):
    34:     def __init__(self, config):
    35:         super().__init__()
    36:         from fla.layers import GatedLinearAttention
    37:         self.attn = GatedLinearAttention(
    38:             mode='chunk',
    39:             hidden_size=config.n_embd,
    40:             num_heads=config.n_head,
    41:             expand_k=0.5,
    42:             expand_v=1.0,
    43:             use_output_gate=True,
    44:             gate_fn='swish',
    45:         )
    46:         self.use_pos_emb = False
    47: 
    48:     @torch.compiler.disable
    49:     def _attn_forward(self, x):
    50:         return self.attn(x)
    51: 
    52:     def forward(self, x):
    53:         o, _, _ = self._attn_forward(x)
    54:         return o
    55: 
    56: # ── Feed-Forward Network ──────────────────────────────────────────────────
    57: class MLP(nn.Module):

Lines 72–83:
    69:         x = self.dropout(x)
    70:         return x
    71: 
    72: class Block(nn.Module):
    73:     def __init__(self, config):
    74:         super().__init__()
    75:         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
    76:         self.attn = CausalSelfAttention(config)
    77:         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
    78:         self.mlp = MLP(config)
    79: 
    80:     def forward(self, x):
    81:         x = x + self.attn(self.ln_1(x))
    82:         x = x + self.mlp(self.ln_2(x))
    83:         return x
    84: 
    85: # ============================================================================
    86: # GPT Model

Lines 229–231:
   226:     grad_clip = 1.0
   227:     warmup_iters = int(max_iters * 0.04)
   228:     lr_decay_iters = max_iters
   229:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   230:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   231:     CONFIG_OVERRIDES = {}
   232: 
   233:     # Apply per-method hyperparameter overrides
   234:     for _k, _v in CONFIG_OVERRIDES.items():
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
