# MLS-Bench: llm-pretrain-kernel

# LLM Pretraining: Custom GPU Kernel Optimization

## Research Question

Write a custom GPU kernel (Triton or CUDA via PyTorch) to implement a
fused MLP operation for GPT-2 pretraining. Your kernel should fuse
multiple operations to reduce memory bandwidth and improve throughput
while maintaining or improving model quality.

## What You Can Modify

The `fused_mlp_forward` function in `custom_pretrain.py`:

- The MLP activation function (default: GELU via separate PyTorch ops)
- Kernel fusion strategy (fuse linear + activation, save intermediate
  values)
- Memory optimization (avoid materializing intermediate tensors)
- Custom autograd Functions for efficient backward pass

The function signature `fused_mlp_forward(x, w_fc, w_proj)` must be
preserved.

- `x`: input tensor `(B*T, n_embd)`
- `w_fc`: first linear weight `(4*n_embd, n_embd)`
- `w_proj`: second linear weight `(n_embd, 4*n_embd)`
- Returns: output tensor `(B*T, n_embd)`

The MLP class calls this function and handles dropout separately.

## Evaluation

- Metrics: validation loss (cross-entropy, lower is better) and training
  throughput (elapsed time, lower is better) — kernel optimizations that
  also change the activation function may improve loss
- Model: GPT-2 Medium (24L/16H/1024D, ~355M params)
- Dataset: FineWeb 10B (GPT-2 tokenizer), ~7.1B tokens (D=20N
  Chinchilla-optimal)
- Training: 13535 iterations, BSZ=64, GA=8, 2-GPU DDP


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/nanoGPT/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `nanoGPT/custom_pretrain.py`
- editable lines **33–48**
- editable lines **257–259**


Other files you may **read** for context (do not modify):
- `nanoGPT/model.py`


## Readable Context


### `nanoGPT/custom_pretrain.py`  [EDITABLE — lines 33–48, lines 257–259 only]

```python
     1: """Custom GPT-2 Pretraining Script
     2: Based on Andrej Karpathy's nanoGPT, evaluated on FineWeb dataset.
     3: """
     4: 
     5: import math
     6: import inspect
     7: import os
     8: import time
     9: from contextlib import nullcontext
    10: from dataclasses import dataclass
    11: 
    12: import numpy as np
    13: import torch
    14: import torch.nn as nn
    15: from torch.nn import functional as F
    16: 
    17: # ============================================================================
    18: # Model Components
    19: # ============================================================================
    20: 
    21: # ── Normalization ──────────────────────────────────────────────────────────
    22: class LayerNorm(nn.Module):
    23:     """LayerNorm but with an optional bias."""
    24:     def __init__(self, ndim, bias):
    25:         super().__init__()
    26:         self.weight = nn.Parameter(torch.ones(ndim))
    27:         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    28: 
    29:     def forward(self, input):
    30:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    31: 
    32: # ── Custom Kernel / Fused Operation ───────────────────────────────────────
    33: def fused_mlp_forward(x, w_fc, w_proj):
    34:     """MLP forward pass: linear -> activation -> linear.
    35: 
    36:     Default implementation uses standard PyTorch ops.
    37:     Can be replaced with a fused Triton kernel for better performance.
    38: 
    39:     Args:
    40:         x: input tensor (B*T, n_embd)
    41:         w_fc: first linear weight (4*n_embd, n_embd)
    42:         w_proj: second linear weight (n_embd, 4*n_embd)
    43:     Returns:
    44:         output tensor (B*T, n_embd)
    45:     """
    46:     h = F.gelu(x @ w_fc.t())
    47:     return h @ w_proj.t()
    48: 
    49: # ── Self-Attention ─────────────────────────────────────────────────────────
    50: class CausalSelfAttention(nn.Module):
    51:     def __init__(self, config):
    52:         super().__init__()
    53:         assert config.n_embd % config.n_head == 0
    54:         self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    55:         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    56:         self.attn_dropout = nn.Dropout(config.dropout)
    57:         self.resid_dropout = nn.Dropout(config.dropout)
    58:         self.n_head = config.n_head
    59:         self.n_embd = config.n_embd
    60:         self.dropout = config.dropout
    61:         self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    62:         if not self.flash:
    63:             self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
    64:                                         .view(1, 1, config.block_size, config.block_size))
    65:         self.use_pos_emb = True
    66: 
    67:     def forward(self, x):
    68:         B, T, C = x.size()
    69:         q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    70:         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    71:         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    72:         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    73:         if self.flash:
    74:             y = torch.nn.functional.scaled_dot_product_attention(
    75:                 q, k, v, attn_mask=None,
    76:                 dropout_p=self.dropout if self.training else 0, is_causal=True)
    77:         else:
    78:             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    79:             att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    80:             att = F.softmax(att, dim=-1)
    81:             att = self.attn_dropout(att)
    82:             y = att @ v
    83:         y = y.transpose(1, 2).contiguous().view(B, T, C)
    84:         y = self.resid_dropout(self.c_proj(y))
    85:         return y
    86: 
    87: # ── Feed-Forward Network ──────────────────────────────────────────────────
    88: class MLP(nn.Module):
    89:     def __init__(self, config):
    90:         super().__init__()
    91:         self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
    92:         self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    93:         self.dropout = nn.Dropout(config.dropout)
    94: 
    95:     def forward(self, x):
    96:         B, T, C = x.size()
    97:         out = fused_mlp_forward(x.view(-1, C), self.c_fc.weight, self.c_proj.weight)
    98:         out = self.dropout(out.view(B, T, C))
    99:         return out
   100: 
   101: # ── Transformer Block ─────────────────────────────────────────────────────
   102: class Block(nn.Module):
   103:     def __init__(self, config):
   104:         super().__init__()
   105:         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
   106:         self.attn = CausalSelfAttention(config)
   107:         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
   108:         self.mlp = MLP(config)
   109: 
   110:     def forward(self, x):
   111:         x = x + self.attn(self.ln_1(x))
   112:         x = x + self.mlp(self.ln_2(x))
   113:         return x
   114: 
   115: # ============================================================================
   116: # GPT Model
   117: # ============================================================================
   118: 
   119: @dataclass
   120: class GPTConfig:
   121:     block_size: int = 1024
   122:     vocab_size: int = 50304
   123:     n_layer: int = 12
   124:     n_head: int = 12
   125:     n_embd: int = 768
   126:     dropout: float = 0.0
   127:     bias: bool = False
   128: 
   129: class GPT(nn.Module):
   130:     def __init__(self, config):
   131:         super().__init__()
   132:         self.config = config
   133:         self.transformer = nn.ModuleDict(dict(
   134:             wte=nn.Embedding(config.vocab_size, config.n_embd),
   135:             wpe=nn.Embedding(config.block_size, config.n_embd),
   136:             drop=nn.Dropout(config.dropout),
   137:             h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
   138:             ln_f=LayerNorm(config.n_embd, bias=config.bias),
   139:         ))
   140:         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
   141:         self.transformer.wte.weight = self.lm_head.weight
   142:         self.apply(self._init_weights)
   143:         for pn, p in self.named_parameters():
   144:             if pn.endswith('c_proj.weight'):
   145:                 torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
   146:         print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
   147: 
   148:     def get_num_params(self, non_embedding=True):
   149:         n_params = sum(p.numel() for p in self.parameters())
   150:         if non_embedding:
   151:             n_params -= self.transformer.wpe.weight.numel()
   152:         return n_params
   153: 
   154:     def _init_weights(self, module):
   155:         if isinstance(module, nn.Linear):
   156:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   157:             if module.bias is not None:
   158:                 torch.nn.init.zeros_(module.bias)
   159:         elif isinstance(module, nn.Embedding):
   160:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   161: 
   162:     def forward(self, idx, targets=None):
   163:         device = idx.device
   164:         b, t = idx.size()
   165:         assert t <= self.config.block_size
   166:         tok_emb = self.transformer.wte(idx)
   167:         x = self.transformer.drop(tok_emb)
   168:         use_pos = getattr(self.transformer.h[0].attn, 'use_pos_emb', True)
   169:         if use_pos:
   170:             pos = torch.arange(0, t, dtype=torch.long, device=device)
   171:             x = x + self.transformer.wpe(pos)
   172:         for block in self.transformer.h:
   173:             x = block(x)
   174:         x = self.transformer.ln_f(x)
   175:         if targets is not None:
   176:             logits = self.lm_head(x)
   177:             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
   178:         else:
   179:             logits = self.lm_head(x[:, [-1], :])
   180:             loss = None
   181:         return logits, loss
   182: 
   183:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   184:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   185:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   186:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   187:         optim_groups = [
   188:             {'params': decay_params, 'weight_decay': weight_decay},
   189:             {'params': nodecay_params, 'weight_decay': 0.0},
   190:         ]
   191:         num_decay_params = sum(p.numel() for p in decay_params)
   192:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   193:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   194:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   195:         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   196:         use_fused = fused_available and device_type == 'cuda'
   197:         extra_args = dict(fused=True) if use_fused else dict()
   198:         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
   199:         print(f"using fused AdamW: {use_fused}")
   200:         return optimizer
   201: 
   202: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   203: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   204:     """Cosine learning rate schedule with linear warmup."""
   205:     if it < warmup_iters:
   206:         return learning_rate * (it + 1) / (warmup_iters + 1)
   207:     if it > lr_decay_iters:
   208:         return min_lr
   209:     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
   210:     assert 0 <= decay_ratio <= 1
   211:     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
   212:     return min_lr + coeff * (learning_rate - min_lr)
   213: 
   214: # ============================================================================
   215: # Data Loading
   216: # ============================================================================
   217: 
   218: def get_batch(data, batch_size, block_size, device):
   219:     """Get a random batch from a pre-opened memmap (nanoGPT style)."""
   220:     ix = torch.randint(len(data) - block_size, (batch_size,))
   221:     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
   222:     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
   223:     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
   224:     return x, y
   225: 
   226: # ============================================================================
   227: # Training Script
   228: # ============================================================================
   229: 
   230: if __name__ == '__main__':
   231:     # ── Configuration from environment ──
   232:     output_dir = os.environ.get('OUTPUT_DIR', 'out')
   233:     seed = int(os.environ.get('SEED', 1337))
   234:     data_dir = os.environ.get('DATA_DIR', '/data/climbmix')
   235: 
   236:     # Model config from environment
   237:     n_layer = int(os.environ.get('N_LAYER', 12))
   238:     n_head = int(os.environ.get('N_HEAD', 12))
   239:     n_embd = int(os.environ.get('N_EMBD', 768))
   240: 
   241:     # Training hyperparameters (overridable via env for different model sizes)
   242:     max_iters = int(os.environ.get('MAX_ITERS', 5000))
   243:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 500))
   244:     eval_iters = 200
   245:     log_interval = 10
   246:     batch_size = int(os.environ.get('BATCH_SIZE', 12))
   247:     block_size = 1024
   248:     gradient_accumulation_steps = int(os.environ.get('GRAD_ACCUM', 5))
   249:     learning_rate = float(os.environ.get('LEARNING_RATE', 6e-4))
   250:     min_lr = learning_rate / 10
   251:     weight_decay = 1e-1
   252:     beta1 = 0.9
   253:     beta2 = 0.95
   254:     grad_clip = 1.0
   255:     warmup_iters = int(max_iters * 0.04)
   256:     lr_decay_iters = max_iters
   257:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   258:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   259:     CONFIG_OVERRIDES = {}
   260: 
   261:     # Apply per-method hyperparameter overrides
   262:     for _k, _v in CONFIG_OVERRIDES.items():
   263:         if _k == 'learning_rate': learning_rate = _v; min_lr = learning_rate / 10
   264:         elif _k == 'weight_decay': weight_decay = _v
   265:         elif _k == 'warmup_iters': warmup_iters = _v
   266:         elif _k == 'min_lr': min_lr = _v
   267:         elif _k == 'grad_clip': grad_clip = _v
   268: 
   269:     compile_model = True
   270:     dtype = 'bfloat16'
   271: 
   272:     # ── DDP Setup ──
   273:     ddp = int(os.environ.get('RANK', -1)) != -1
   274:     if ddp:
   275:         import torch.distributed as dist
   276:         from torch.nn.parallel import DistributedDataParallel as DDP
   277:         dist.init_process_group(backend='nccl')
   278:         ddp_rank = int(os.environ['RANK'])
   279:         ddp_local_rank = int(os.environ['LOCAL_RANK'])
   280:         ddp_world_size = int(os.environ['WORLD_SIZE'])
   281:         device = f'cuda:{ddp_local_rank}'
   282:         torch.cuda.set_device(device)
   283:         master_process = ddp_rank == 0
   284:         seed_offset = ddp_rank
   285:         assert gradient_accumulation_steps % ddp_world_size == 0
   286:         gradient_accumulation_steps //= ddp_world_size
   287:     else:
   288:         master_process = True
   289:         device = 'cuda'
   290:         seed_offset = 0
   291: 
   292:     # ── Setup ──
   293:     device_type = 'cuda'
   294:     torch.manual_seed(seed + seed_offset)
   295:     torch.backends.cuda.matmul.allow_tf32 = True
   296:     torch.backends.cudnn.allow_tf32 = True
   297:     ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
   298:     ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
   299:     if master_process:
   300:         os.makedirs(output_dir, exist_ok=True)
   301: 
   302:     tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
   303:     if ddp:
   304:         tokens_per_iter *= int(os.environ.get('WORLD_SIZE', 1))
   305:     if master_process:
   306:         print(f"tokens per iteration will be: {tokens_per_iter:,}")
   307: 
   308:     # ── Load Data ──
   309:     train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
   310:     val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
   311:     if master_process:
   312:         print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
   313: 
   314:     # ── Model Init ──
   315:     model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
   316:                       block_size=block_size, bias=False, vocab_size=50304, dropout=0.0)
   317:     gptconf = GPTConfig(**model_args)
   318:     model = GPT(gptconf)
   319:     model.to(device)
   320: 
   321: 
   322:     scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
   323:     optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
   324: 
   325:     if compile_model:
   326:         if master_process:
   327:             print("compiling the model...")
   328:         model = torch.compile(model)
   329: 
   330:     if ddp:
   331:         model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
   332: 
   333:     # ── Evaluation ──
   334:     @torch.no_grad()
   335:     def estimate_loss():
   336:         out = {}
   337:         raw = model.module if ddp else model
   338:         raw.eval()
   339:         for split, data in [('train', train_data), ('val', val_data)]:
   340:             losses = torch.zeros(eval_iters)
   341:             for k in range(eval_iters):
   342:                 X, Y = get_batch(data, batch_size, block_size, device)
   343:                 with ctx:
   344:                     logits, loss = raw(X, Y)
   345:                 losses[k] = loss.item()
   346:             out[split] = losses.mean()
   347:         raw.train()
   348:         return out
   349: 
   350:     # ── Training Loop ──
   351:     t0 = time.time()
   352:     best_val_loss = 1e9
   353: 
   354:     for iter_num in range(max_iters + 1):
   355:         lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
   356:         for param_group in optimizer.param_groups:
   357:             param_group['lr'] = lr
   358: 
   359:         if iter_num % eval_interval == 0 and master_process:
   360:             losses = estimate_loss()
   361:             train_loss = losses['train'].item()
   362:             val_loss = losses['val'].item()
   363:             print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
   364:             print(f"TRAIN_METRICS: step={iter_num}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
   365:             if val_loss < best_val_loss:
   366:                 best_val_loss = val_loss
   367: 
   368:         for micro_step in range(gradient_accumulation_steps):
   369:             if ddp:
   370:                 model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
   371:             with ctx:
   372:                 X, Y = get_batch(train_data, batch_size, block_size, device)
   373:                 logits, loss = model(X, Y)
   374:                 loss = loss / gradient_accumulation_steps
   375:             scaler.scale(loss).backward()
   376: 
   377:         if grad_clip != 0.0:
   378:             scaler.unscale_(optimizer)
   379:             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
   380:         scaler.step(optimizer)
   381:         scaler.update()
   382:         optimizer.zero_grad(set_to_none=True)
   383: 
   384:         t1 = time.time()
   385:         dt = t1 - t0
   386:         t0 = t1
   387:         if iter_num % log_interval == 0 and iter_num > 0 and master_process:
   388:             lossf = loss.item() * gradient_accumulation_steps
   389:             print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
   390: 
   391:     # ── Free training state to reclaim GPU memory ──
   392:     del optimizer, scaler
   393:     import gc; gc.collect()
   394:     torch.cuda.empty_cache()
   395: 
   396:     # ── Final Evaluation ──
   397:     if master_process:
   398:         losses = estimate_loss()
   399:         val_loss = losses['val'].item()
   400:         train_loss = losses['train'].item()
   401:         print(f"Final: train loss {train_loss:.4f}, val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}")
   402: 
   403:         # ── PPL on benchmark datasets ──
   404:         eval_dir = os.environ.get('EVAL_DIR', '/data/eval')
   405:         raw = model.module if ddp else model
   406:         raw.eval()
   407:         eval_datasets = ['wikitext2', 'lambada']
   408:         ppl_results = {}
   409:         for ds_name in eval_datasets:
   410:             ds_path = os.path.join(eval_dir, f'{ds_name}.bin')
   411:             if not os.path.exists(ds_path):
   412:                 print(f"Eval dataset not found: {ds_path}")
   413:                 continue
   414:             data = np.memmap(ds_path, dtype=np.uint16, mode='r')
   415:             n_tokens = len(data)
   416:             # Process in non-overlapping chunks of block_size
   417:             total_loss = 0.0
   418:             n_chunks = 0
   419:             with torch.no_grad():
   420:                 for start in range(0, n_tokens - block_size, block_size):
   421:                     x = torch.from_numpy(data[start:start+block_size].astype(np.int64)).unsqueeze(0).to(device)
   422:                     y = torch.from_numpy(data[start+1:start+1+block_size].astype(np.int64)).unsqueeze(0).to(device)
   423:                     with ctx:
   424:                         _, loss = raw(x, y)
   425:                     total_loss += loss.item()
   426:                     n_chunks += 1
   427:             avg_loss = total_loss / n_chunks
   428:             ppl = math.exp(avg_loss)
   429:             ppl_results[ds_name] = ppl
   430:             print(f"PPL {ds_name}: {ppl:.2f} (avg_loss={avg_loss:.4f}, {n_chunks} chunks)")
   431: 
   432:         ppl_str = ', '.join(f'{k}_ppl={v:.2f}' for k, v in ppl_results.items())
   433:         print(f"TEST_METRICS: val_loss={val_loss:.4f}, {ppl_str}", flush=True)
   434: 
   435:         # ── Save checkpoint for downstream evaluation (lm-eval-harness) ──
   436:         import shutil
   437:         env_label = os.environ.get('ENV', 'model')
   438:         # Unwrap torch.compile to get clean state_dict keys
   439:         save_model = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
   440:         ckpt_data = {'model_state_dict': save_model.state_dict(), 'model_args': model_args}
   441:         ckpt_path = os.path.join(output_dir, f'ckpt_{env_label}.pt')
   442:         torch.save(ckpt_data, ckpt_path)
   443:         print(f"Checkpoint saved to {ckpt_path}")
   444:         src_path = os.path.join(output_dir, f'model_source_{env_label}.py')
   445:         shutil.copy2(os.path.abspath(__file__), src_path)
   446:         print(f"Model source saved to {src_path}")
   447: 
   448:     if ddp:
   449:         dist.destroy_process_group()
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **gpt-345m** — wall-clock budget `12:00:00`, compute share `4.0`
- **lm-eval-345m** — wall-clock budget `1:00:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.

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


### `relu_sq_torch` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 33–61:
    30:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    31: 
    32: # ── Custom Kernel / Fused Operation ───────────────────────────────────────
    33: def fused_mlp_forward(x, w_fc, w_proj):
    34:     """MLP forward with ReLU^2 activation via custom autograd."""
    35: 
    36:     class ReLUSquaredMLP(torch.autograd.Function):
    37:         @staticmethod
    38:         def forward(ctx, x, w_fc, w_proj):
    39:             h = x @ w_fc.t()
    40:             relu_h = F.relu(h)
    41:             act = relu_h * relu_h  # ReLU^2
    42:             out = act @ w_proj.t()
    43:             ctx.save_for_backward(x, w_fc, w_proj, h, relu_h)
    44:             return out
    45: 
    46:         @staticmethod
    47:         def backward(ctx, grad_output):
    48:             x, w_fc, w_proj, h, relu_h = ctx.saved_tensors
    49:             dtype = grad_output.dtype
    50:             # grad through second linear
    51:             d_act = grad_output @ w_proj.to(dtype)
    52:             # grad through ReLU^2: d/dx[relu(x)^2] = 2*relu(x) * (x > 0)
    53:             d_h = 2 * relu_h.to(dtype) * d_act
    54:             # weight grads
    55:             act_sq = (relu_h * relu_h).to(dtype)
    56:             grad_w_proj = grad_output.t() @ act_sq
    57:             grad_w_fc = d_h.t() @ x.to(dtype)
    58:             grad_x = d_h @ w_fc.to(dtype)
    59:             return grad_x, grad_w_fc, grad_w_proj
    60: 
    61:     return ReLUSquaredMLP.apply(x, w_fc, w_proj)
    62: # ── Self-Attention ─────────────────────────────────────────────────────────
    63: class CausalSelfAttention(nn.Module):
    64:     def __init__(self, config):

Lines 270–272:
   267:     grad_clip = 1.0
   268:     warmup_iters = int(max_iters * 0.04)
   269:     lr_decay_iters = max_iters
   270:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   271:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   272:     CONFIG_OVERRIDES = {}
   273: 
   274:     # Apply per-method hyperparameter overrides
   275:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `triton_gelu` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 33–93:
    30:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    31: 
    32: # ── Custom Kernel / Fused Operation ───────────────────────────────────────
    33: import triton
    34: import triton.language as tl
    35: from triton.language.extra.cuda import libdevice
    36: 
    37: @triton.jit
    38: def _fused_gelu_kernel(
    39:     x_ptr, out_ptr,
    40:     n_elements,
    41:     BLOCK_SIZE: tl.constexpr,
    42: ):
    43:     pid = tl.program_id(0)
    44:     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    45:     mask = offsets < n_elements
    46:     x = tl.load(x_ptr + offsets, mask=mask)
    47:     # Compute entirely in float32 to avoid bfloat16 overflow in x^3
    48:     xf = x.to(tl.float32)
    49:     # tanh-approximation GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    50:     c = 0.7978845608028654  # sqrt(2/pi)
    51:     inner = c * (xf + 0.044715 * xf * xf * xf)
    52:     tanh_val = libdevice.tanh(inner)
    53:     out = xf * 0.5 * (1.0 + tanh_val)
    54:     tl.store(out_ptr + offsets, out.to(x.dtype), mask=mask)
    55: 
    56: class _TritonGELUMLP(torch.autograd.Function):
    57:     @staticmethod
    58:     def forward(ctx, x, w_fc, w_proj):
    59:         h = x @ w_fc.t()
    60:         act = torch.empty_like(h)
    61:         n = h.numel()
    62:         BLOCK = 1024
    63:         grid = ((n + BLOCK - 1) // BLOCK,)
    64:         _fused_gelu_kernel[grid](h, act, n, BLOCK_SIZE=BLOCK)
    65:         out = act @ w_proj.t()
    66:         ctx.save_for_backward(x, w_fc, w_proj, h, act)
    67:         return out
    68: 
    69:     @staticmethod
    70:     def backward(ctx, grad_output):
    71:         x, w_fc, w_proj, h, act = ctx.saved_tensors
    72:         dtype = grad_output.dtype
    73:         d_act = grad_output @ w_proj.to(dtype)
    74:         grad_w_proj = grad_output.reshape(-1, grad_output.shape[-1]).t() @ act.to(dtype).reshape(-1, act.shape[-1])
    75:         # Analytical gradient of tanh-approximation GELU (matches the Triton forward)
    76:         # gelu(x) = 0.5 * x * (1 + tanh(inner)), inner = c * (x + 0.044715 * x^3)
    77:         # d_gelu/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner/dx
    78:         # d_inner/dx = c * (1 + 3 * 0.044715 * x^2)
    79:         h_f = h.float()
    80:         c = 0.7978845608028654
    81:         inner = c * (h_f + 0.044715 * h_f * h_f * h_f)
    82:         tanh_inner = torch.tanh(inner)
    83:         sech2 = 1.0 - tanh_inner * tanh_inner
    84:         d_inner = c * (1.0 + 3.0 * 0.044715 * h_f * h_f)
    85:         gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * h_f * sech2 * d_inner
    86:         d_h = (d_act.float() * gelu_grad).to(dtype)
    87:         grad_x = d_h @ w_fc.to(dtype)
    88:         grad_w_fc = d_h.reshape(-1, d_h.shape[-1]).t() @ x.to(dtype).reshape(-1, x.shape[-1])
    89:         return grad_x, grad_w_fc, grad_w_proj
    90: 
    91: def fused_mlp_forward(x, w_fc, w_proj):
    92:     """MLP forward with Triton fused GELU kernel."""
    93:     return _TritonGELUMLP.apply(x, w_fc, w_proj)
    94: # ── Self-Attention ─────────────────────────────────────────────────────────
    95: class CausalSelfAttention(nn.Module):
    96:     def __init__(self, config):

Lines 302–304:
   299:     grad_clip = 1.0
   300:     warmup_iters = int(max_iters * 0.04)
   301:     lr_decay_iters = max_iters
   302:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   303:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   304:     CONFIG_OVERRIDES = {}
   305: 
   306:     # Apply per-method hyperparameter overrides
   307:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `triton_relu_sq_fused` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 33–110:
    30:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    31: 
    32: # ── Custom Kernel / Fused Operation ───────────────────────────────────────
    33: import triton
    34: import triton.language as tl
    35: 
    36: @triton.jit
    37: def _matmul_relu_sq_kernel(
    38:     a_ptr, b_ptr, c_ptr, pre_ptr,
    39:     M, N, K,
    40:     stride_am, stride_ak,
    41:     stride_bk, stride_bn,
    42:     stride_cm, stride_cn,
    43:     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    44:     SAVE_PRE: tl.constexpr,
    45: ):
    46:     pid_m = tl.program_id(0)
    47:     pid_n = tl.program_id(1)
    48:     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    49:     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    50:     offs_k = tl.arange(0, BLOCK_K)
    51:     a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    52:     b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    53:     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    54:     for k in range(0, K, BLOCK_K):
    55:         a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
    56:         b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
    57:         acc += tl.dot(a, b)
    58:         a_ptrs += BLOCK_K * stride_ak
    59:         b_ptrs += BLOCK_K * stride_bk
    60:         offs_k += BLOCK_K
    61:     pre = acc.to(tl.bfloat16)
    62:     relu_val = tl.maximum(acc, 0.0)
    63:     result = (relu_val * relu_val).to(tl.bfloat16)
    64:     c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    65:     mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    66:     tl.store(c_ptrs, result, mask=mask)
    67:     if SAVE_PRE:
    68:         pre_ptrs = pre_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    69:         tl.store(pre_ptrs, pre, mask=mask)
    70: 
    71: class _FusedLinearReLUSquare(torch.autograd.Function):
    72:     @staticmethod
    73:     def forward(ctx, x, w_fc, w_proj):
    74:         M, K = x.shape
    75:         N = w_fc.shape[0]
    76:         post = torch.empty((M, N), device=x.device, dtype=x.dtype)
    77:         pre = torch.empty((M, N), device=x.device, dtype=x.dtype)
    78:         grid = lambda meta: (
    79:             triton.cdiv(M, meta['BLOCK_M']),
    80:             triton.cdiv(N, meta['BLOCK_N']),
    81:         )
    82:         b = w_fc.t().contiguous()
    83:         _matmul_relu_sq_kernel[grid](
    84:             x, b, post, pre,
    85:             M, N, K,
    86:             x.stride(0), x.stride(1),
    87:             b.stride(0), b.stride(1),
    88:             post.stride(0), post.stride(1),
    89:             BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    90:             SAVE_PRE=True,
    91:         )
    92:         out = post @ w_proj.t()
    93:         ctx.save_for_backward(x, w_fc, w_proj, pre)
    94:         return out
    95: 
    96:     @staticmethod
    97:     def backward(ctx, grad_output):
    98:         x, w_fc, w_proj, pre = ctx.saved_tensors
    99:         dtype = grad_output.dtype
   100:         d_post = grad_output @ w_proj.to(dtype)
   101:         grad_w_proj = grad_output.reshape(-1, grad_output.shape[-1]).t() @ \
   102:                       F.relu(pre).pow(2).to(dtype).reshape(-1, pre.shape[-1])
   103:         d_pre = 2 * F.relu(pre).to(dtype) * d_post
   104:         grad_x = d_pre @ w_fc.to(dtype)
   105:         grad_w_fc = d_pre.reshape(-1, d_pre.shape[-1]).t() @ x.to(dtype).reshape(-1, x.shape[-1])
   106:         return grad_x, grad_w_fc, grad_w_proj
   107: 
   108: def fused_mlp_forward(x, w_fc, w_proj):
   109:     """MLP forward with Triton fused linear+ReLU^2 kernel."""
   110:     return _FusedLinearReLUSquare.apply(x, w_fc, w_proj)
   111: # ── Self-Attention ─────────────────────────────────────────────────────────
   112: class CausalSelfAttention(nn.Module):
   113:     def __init__(self, config):

Lines 319–321:
   316:     grad_clip = 1.0
   317:     warmup_iters = int(max_iters * 0.04)
   318:     lr_decay_iters = max_iters
   319:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   320:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   321:     CONFIG_OVERRIDES = {}
   322: 
   323:     # Apply per-method hyperparameter overrides
   324:     for _k, _v in CONFIG_OVERRIDES.items():
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
