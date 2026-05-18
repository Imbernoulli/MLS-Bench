# MLS-Bench: llm-pretrain-optimizer

# LLM Pretraining: Optimizer & Learning Rate Schedule Optimization

## Research Question
Design an improved optimizer and / or learning-rate schedule for GPT-style language model pretraining. The change should reduce validation loss compared to AdamW + cosine annealing under the same model and data budget.

## Background
The default optimizer is AdamW (fused) with weight decay only on 2D parameters and cosine LR decay with linear warmup. Studied alternatives at this layer:

- **Lion** — Chen et al., "Symbolic Discovery of Optimization Algorithms", NeurIPS 2023, arXiv:2302.06675. Sign-momentum optimizer found via program search; tracks only momentum, applies a uniform-magnitude `sign(...)` update; typically uses LR ≈ 0.1× AdamW LR and stronger weight decay.
- **Muon** — Keller Jordan et al. (2024), "Muon: An optimizer for hidden layers in neural networks" (https://kellerjordan.github.io/posts/muon/). Applies SGD-momentum, then orthogonalizes the resulting matrix update via a 5-step Newton–Schulz iteration; intended for 2D hidden-layer matrices, with AdamW kept for embeddings / `lm_head` / 1D parameters. ~35% training-speed improvement reported on the NanoGPT speedrun versus AdamW.
- **AdamW + Nesterov momentum** — straightforward variant adding Nesterov-style lookahead to Adam's first moment.

## What you can modify
Two regions in `nanoGPT/custom_pretrain.py`:

1. **`configure_optimizers` method** — optimizer creation and parameter grouping.
2. **`get_lr` function** — learning-rate schedule.

You may modify:
- The optimization algorithm (default: AdamW fused).
- Parameter grouping strategy (default: weight decay for 2D params, none for 1D params).
- LR schedule shape (default: cosine with linear warmup).
- Any optimizer hyperparameters (betas, eps, weight decay, etc.).

### Interface contract
- `get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr)` — keep this signature.
- The optimizer returned by `configure_optimizers` must support `.zero_grad()`, `.step()`, and `.param_groups`.
- Architecture, tokenizer, dataset, batch construction, and evaluation are fixed.

## Reference baselines
- `lion` — Lion optimizer with cosine schedule.
- `muon` — Muon for 2D hidden weights + AdamW for the rest.
- `adamw_nesterov` — AdamW with Nesterov momentum.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens.
- **Training**: 12,030 iterations, micro-batch 96, gradient accumulation 6, 2-GPU DDP.

## Evaluation
- **Validation loss** — cross-entropy on FineWeb (lower is better, primary).
- **Perplexity** — WikiText-2, LAMBADA (lower is better).
- **Downstream accuracy** — ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/nanoGPT/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `nanoGPT/custom_pretrain.py`
- editable lines **171–189**
- editable lines **245–247**


Other files you may **read** for context (do not modify):
- `nanoGPT/model.py`
- `nanoGPT/train.py`


## Readable Context


### `nanoGPT/custom_pretrain.py`  [EDITABLE — lines 171–189, lines 245–247 only]

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
    32: # ── Self-Attention ─────────────────────────────────────────────────────────
    33: class CausalSelfAttention(nn.Module):
    34:     def __init__(self, config):
    35:         super().__init__()
    36:         assert config.n_embd % config.n_head == 0
    37:         self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
    38:         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    39:         self.attn_dropout = nn.Dropout(config.dropout)
    40:         self.resid_dropout = nn.Dropout(config.dropout)
    41:         self.n_head = config.n_head
    42:         self.n_embd = config.n_embd
    43:         self.dropout = config.dropout
    44:         self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    45:         if not self.flash:
    46:             self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
    47:                                         .view(1, 1, config.block_size, config.block_size))
    48:         # Set to False if using custom position encoding (e.g. RoPE)
    49:         self.use_pos_emb = True
    50: 
    51:     def forward(self, x):
    52:         B, T, C = x.size()
    53:         q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    54:         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    55:         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    56:         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    57:         if self.flash:
    58:             y = torch.nn.functional.scaled_dot_product_attention(
    59:                 q, k, v, attn_mask=None,
    60:                 dropout_p=self.dropout if self.training else 0, is_causal=True)
    61:         else:
    62:             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    63:             att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    64:             att = F.softmax(att, dim=-1)
    65:             att = self.attn_dropout(att)
    66:             y = att @ v
    67:         y = y.transpose(1, 2).contiguous().view(B, T, C)
    68:         y = self.resid_dropout(self.c_proj(y))
    69:         return y
    70: 
    71: # ── Feed-Forward Network ──────────────────────────────────────────────────
    72: class MLP(nn.Module):
    73:     def __init__(self, config):
    74:         super().__init__()
    75:         self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
    76:         self.gelu = nn.GELU()
    77:         self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    78:         self.dropout = nn.Dropout(config.dropout)
    79: 
    80:     def forward(self, x):
    81:         x = self.c_fc(x)
    82:         x = self.gelu(x)
    83:         x = self.c_proj(x)
    84:         x = self.dropout(x)
    85:         return x
    86: 
    87: # ── Transformer Block ─────────────────────────────────────────────────────
    88: class Block(nn.Module):
    89:     def __init__(self, config):
    90:         super().__init__()
    91:         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
    92:         self.attn = CausalSelfAttention(config)
    93:         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
    94:         self.mlp = MLP(config)
    95: 
    96:     def forward(self, x):
    97:         x = x + self.attn(self.ln_1(x))
    98:         x = x + self.mlp(self.ln_2(x))
    99:         return x
   100: 
   101: # ============================================================================
   102: # GPT Model
   103: # ============================================================================
   104: 
   105: @dataclass
   106: class GPTConfig:
   107:     block_size: int = 1024
   108:     vocab_size: int = 50304
   109:     n_layer: int = 12
   110:     n_head: int = 12
   111:     n_embd: int = 768
   112:     dropout: float = 0.0
   113:     bias: bool = False
   114: 
   115: class GPT(nn.Module):
   116:     def __init__(self, config):
   117:         super().__init__()
   118:         self.config = config
   119:         self.transformer = nn.ModuleDict(dict(
   120:             wte=nn.Embedding(config.vocab_size, config.n_embd),
   121:             wpe=nn.Embedding(config.block_size, config.n_embd),
   122:             drop=nn.Dropout(config.dropout),
   123:             h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
   124:             ln_f=LayerNorm(config.n_embd, bias=config.bias),
   125:         ))
   126:         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
   127:         self.transformer.wte.weight = self.lm_head.weight
   128:         self.apply(self._init_weights)
   129:         for pn, p in self.named_parameters():
   130:             if pn.endswith('c_proj.weight'):
   131:                 torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
   132:         print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
   133: 
   134:     def get_num_params(self, non_embedding=True):
   135:         n_params = sum(p.numel() for p in self.parameters())
   136:         if non_embedding:
   137:             n_params -= self.transformer.wpe.weight.numel()
   138:         return n_params
   139: 
   140:     def _init_weights(self, module):
   141:         if isinstance(module, nn.Linear):
   142:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   143:             if module.bias is not None:
   144:                 torch.nn.init.zeros_(module.bias)
   145:         elif isinstance(module, nn.Embedding):
   146:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   147: 
   148:     def forward(self, idx, targets=None):
   149:         device = idx.device
   150:         b, t = idx.size()
   151:         assert t <= self.config.block_size
   152:         tok_emb = self.transformer.wte(idx)
   153:         x = self.transformer.drop(tok_emb)
   154:         # Conditionally add learned position embeddings
   155:         use_pos = getattr(self.transformer.h[0].attn, 'use_pos_emb', True)
   156:         if use_pos:
   157:             pos = torch.arange(0, t, dtype=torch.long, device=device)
   158:             x = x + self.transformer.wpe(pos)
   159:         for block in self.transformer.h:
   160:             x = block(x)
   161:         x = self.transformer.ln_f(x)
   162:         if targets is not None:
   163:             logits = self.lm_head(x)
   164:             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
   165:         else:
   166:             logits = self.lm_head(x[:, [-1], :])
   167:             loss = None
   168:         return logits, loss
   169: 
   170:     # ── Optimizer Configuration ────────────────────────────────────────────
   171:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   172:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   173:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   174:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   175:         optim_groups = [
   176:             {'params': decay_params, 'weight_decay': weight_decay},
   177:             {'params': nodecay_params, 'weight_decay': 0.0},
   178:         ]
   179:         num_decay_params = sum(p.numel() for p in decay_params)
   180:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   181:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   182:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   183:         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   184:         use_fused = fused_available and device_type == 'cuda'
   185:         extra_args = dict(fused=True) if use_fused else dict()
   186:         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
   187:         print(f"using fused AdamW: {use_fused}")
   188:         return optimizer
   189: 
   190: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   191: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   192:     """Cosine learning rate schedule with linear warmup."""
   193:     if it < warmup_iters:
   194:         return learning_rate * (it + 1) / (warmup_iters + 1)
   195:     if it > lr_decay_iters:
   196:         return min_lr
   197:     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
   198:     assert 0 <= decay_ratio <= 1
   199:     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
   200:     return min_lr + coeff * (learning_rate - min_lr)
   201: 
   202: # ============================================================================
   203: # Data Loading
   204: # ============================================================================
   205: 
   206: def get_batch(data, batch_size, block_size, device):
   207:     """Get a random batch from a pre-opened memmap (nanoGPT style)."""
   208:     ix = torch.randint(len(data) - block_size, (batch_size,))
   209:     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
   210:     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
   211:     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
   212:     return x, y
   213: 
   214: # ============================================================================
   215: # Training Script
   216: # ============================================================================
   217: 
   218: if __name__ == '__main__':
   219:     # ── Configuration from environment ──
   220:     output_dir = os.environ.get('OUTPUT_DIR', 'out')
   221:     seed = int(os.environ.get('SEED', 1337))
   222:     data_dir = os.environ.get('DATA_DIR', '/data/climbmix')
   223: 
   224:     # Model config from environment
   225:     n_layer = int(os.environ.get('N_LAYER', 12))
   226:     n_head = int(os.environ.get('N_HEAD', 12))
   227:     n_embd = int(os.environ.get('N_EMBD', 768))
   228: 
   229:     # Training hyperparameters (overridable via env for different model sizes)
   230:     max_iters = int(os.environ.get('MAX_ITERS', 5000))
   231:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 500))
   232:     eval_iters = 200
   233:     log_interval = 10
   234:     batch_size = int(os.environ.get('BATCH_SIZE', 12))
   235:     block_size = 1024
   236:     gradient_accumulation_steps = int(os.environ.get('GRAD_ACCUM', 5))
   237:     learning_rate = float(os.environ.get('LEARNING_RATE', 6e-4))
   238:     min_lr = learning_rate / 10
   239:     weight_decay = 1e-1
   240:     beta1 = 0.9
   241:     beta2 = 0.95
   242:     grad_clip = 1.0
   243:     warmup_iters = int(max_iters * 0.04)
   244:     lr_decay_iters = max_iters
   245:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   246:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   247:     CONFIG_OVERRIDES = {}
   248: 
   249:     # Apply per-method hyperparameter overrides
   250:     for _k, _v in CONFIG_OVERRIDES.items():
   251:         if _k == 'learning_rate': learning_rate = _v; min_lr = learning_rate / 10
   252:         elif _k == 'weight_decay': weight_decay = _v
   253:         elif _k == 'warmup_iters': warmup_iters = _v
   254:         elif _k == 'min_lr': min_lr = _v
   255:         elif _k == 'grad_clip': grad_clip = _v
   256: 
   257:     compile_model = True
   258:     dtype = 'bfloat16'
   259: 
   260:     # ── DDP Setup ──
   261:     ddp = int(os.environ.get('RANK', -1)) != -1
   262:     if ddp:
   263:         import torch.distributed as dist
   264:         from torch.nn.parallel import DistributedDataParallel as DDP
   265:         dist.init_process_group(backend='nccl')
   266:         ddp_rank = int(os.environ['RANK'])
   267:         ddp_local_rank = int(os.environ['LOCAL_RANK'])
   268:         ddp_world_size = int(os.environ['WORLD_SIZE'])
   269:         device = f'cuda:{ddp_local_rank}'
   270:         torch.cuda.set_device(device)
   271:         master_process = ddp_rank == 0
   272:         seed_offset = ddp_rank
   273:         assert gradient_accumulation_steps % ddp_world_size == 0
   274:         gradient_accumulation_steps //= ddp_world_size
   275:     else:
   276:         master_process = True
   277:         device = 'cuda'
   278:         seed_offset = 0
   279: 
   280:     # ── Setup ──
   281:     device_type = 'cuda'
   282:     torch.manual_seed(seed + seed_offset)
   283:     torch.backends.cuda.matmul.allow_tf32 = True
   284:     torch.backends.cudnn.allow_tf32 = True
   285:     ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
   286:     ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
   287:     if master_process:
   288:         os.makedirs(output_dir, exist_ok=True)
   289: 
   290:     tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
   291:     if ddp:
   292:         tokens_per_iter *= int(os.environ.get('WORLD_SIZE', 1))
   293:     if master_process:
   294:         print(f"tokens per iteration will be: {tokens_per_iter:,}")
   295: 
   296:     # ── Load Data ──
   297:     train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
   298:     val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
   299:     if master_process:
   300:         print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
   301: 
   302:     # ── Model Init ──
   303:     model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
   304:                       block_size=block_size, bias=False, vocab_size=50304, dropout=0.0)
   305:     gptconf = GPTConfig(**model_args)
   306:     model = GPT(gptconf)
   307:     model.to(device)
   308: 
   309: 
   310:     scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
   311:     optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
   312: 
   313:     if compile_model:
   314:         if master_process:
   315:             print("compiling the model...")
   316:         model = torch.compile(model)
   317: 
   318:     if ddp:
   319:         model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
   320: 
   321:     # ── Evaluation ──
   322:     @torch.no_grad()
   323:     def estimate_loss():
   324:         out = {}
   325:         raw = model.module if ddp else model
   326:         raw.eval()
   327:         for split, data in [('train', train_data), ('val', val_data)]:
   328:             losses = torch.zeros(eval_iters)
   329:             for k in range(eval_iters):
   330:                 X, Y = get_batch(data, batch_size, block_size, device)
   331:                 with ctx:
   332:                     logits, loss = raw(X, Y)
   333:                 losses[k] = loss.item()
   334:             out[split] = losses.mean()
   335:         raw.train()
   336:         return out
   337: 
   338:     # ── Training Loop ──
   339:     t0 = time.time()
   340:     best_val_loss = 1e9
   341: 
   342:     for iter_num in range(max_iters + 1):
   343:         lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
   344:         for param_group in optimizer.param_groups:
   345:             param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
   346: 
   347:         if iter_num % eval_interval == 0 and master_process:
   348:             losses = estimate_loss()
   349:             train_loss = losses['train'].item()
   350:             val_loss = losses['val'].item()
   351:             print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
   352:             print(f"TRAIN_METRICS: step={iter_num}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
   353:             if val_loss < best_val_loss:
   354:                 best_val_loss = val_loss
   355: 
   356:         for micro_step in range(gradient_accumulation_steps):
   357:             if ddp:
   358:                 model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
   359:             with ctx:
   360:                 X, Y = get_batch(train_data, batch_size, block_size, device)
   361:                 logits, loss = model(X, Y)
   362:                 loss = loss / gradient_accumulation_steps
   363:             scaler.scale(loss).backward()
   364: 
   365:         if grad_clip != 0.0:
   366:             scaler.unscale_(optimizer)
   367:             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
   368:         scaler.step(optimizer)
   369:         scaler.update()
   370:         optimizer.zero_grad(set_to_none=True)
   371: 
   372:         t1 = time.time()
   373:         dt = t1 - t0
   374:         t0 = t1
   375:         if iter_num % log_interval == 0 and iter_num > 0 and master_process:
   376:             lossf = loss.item() * gradient_accumulation_steps
   377:             print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
   378: 
   379:     # ── Free training state to reclaim GPU memory ──
   380:     del optimizer, scaler
   381:     import gc; gc.collect()
   382:     torch.cuda.empty_cache()
   383: 
   384:     # ── Final Evaluation ──
   385:     if master_process:
   386:         losses = estimate_loss()
   387:         val_loss = losses['val'].item()
   388:         train_loss = losses['train'].item()
   389:         print(f"Final: train loss {train_loss:.4f}, val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}")
   390: 
   391:         # ── PPL on benchmark datasets ──
   392:         eval_dir = os.environ.get('EVAL_DIR', '/data/eval')
   393:         raw = model.module if ddp else model
   394:         raw.eval()
   395:         eval_datasets = ['wikitext2', 'lambada']
   396:         ppl_results = {}
   397:         for ds_name in eval_datasets:
   398:             ds_path = os.path.join(eval_dir, f'{ds_name}.bin')
   399:             if not os.path.exists(ds_path):
   400:                 print(f"Eval dataset not found: {ds_path}")
   401:                 continue
   402:             data = np.memmap(ds_path, dtype=np.uint16, mode='r')
   403:             n_tokens = len(data)
   404:             # Process in non-overlapping chunks of block_size
   405:             total_loss = 0.0
   406:             n_chunks = 0
   407:             with torch.no_grad():
   408:                 for start in range(0, n_tokens - block_size, block_size):
   409:                     x = torch.from_numpy(data[start:start+block_size].astype(np.int64)).unsqueeze(0).to(device)
   410:                     y = torch.from_numpy(data[start+1:start+1+block_size].astype(np.int64)).unsqueeze(0).to(device)
   411:                     with ctx:
   412:                         _, loss = raw(x, y)
   413:                     total_loss += loss.item()
   414:                     n_chunks += 1
   415:             avg_loss = total_loss / n_chunks
   416:             ppl = math.exp(avg_loss)
   417:             ppl_results[ds_name] = ppl
   418:             print(f"PPL {ds_name}: {ppl:.2f} (avg_loss={avg_loss:.4f}, {n_chunks} chunks)")
   419: 
   420:         ppl_str = ', '.join(f'{k}_ppl={v:.2f}' for k, v in ppl_results.items())
   421:         print(f"TEST_METRICS: val_loss={val_loss:.4f}, {ppl_str}", flush=True)
   422: 
   423:         # ── Save checkpoint for downstream evaluation (lm-eval-harness) ──
   424:         import shutil
   425:         env_label = os.environ.get('ENV', 'model')
   426:         # Unwrap torch.compile to get clean state_dict keys
   427:         save_model = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
   428:         ckpt_data = {'model_state_dict': save_model.state_dict(), 'model_args': model_args}
   429:         ckpt_path = os.path.join(output_dir, f'ckpt_{env_label}.pt')
   430:         torch.save(ckpt_data, ckpt_path)
   431:         print(f"Checkpoint saved to {ckpt_path}")
   432:         src_path = os.path.join(output_dir, f'model_source_{env_label}.py')
   433:         shutil.copy2(os.path.abspath(__file__), src_path)
   434:         print(f"Model source saved to {src_path}")
   435: 
   436:     if ddp:
   437:         dist.destroy_process_group()
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **gpt-345m** — wall-clock budget `12:00:00`, compute share `4.0`
- **lm-eval-345m** — wall-clock budget `1:00:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `adamw_nesterov` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 171–186:
   168:         return logits, loss
   169: 
   170:     # ── Optimizer Configuration ────────────────────────────────────────────
   171:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   172:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   173:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   174:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   175:         optim_groups = [
   176:             {'params': decay_params, 'weight_decay': weight_decay},
   177:             {'params': nodecay_params, 'weight_decay': 0.0},
   178:         ]
   179:         num_decay_params = sum(p.numel() for p in decay_params)
   180:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   181:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   182:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   183:         optimizer = torch.optim.NAdam(optim_groups, lr=learning_rate, betas=betas,
   184:                                       decoupled_weight_decay=True)
   185:         print("using NAdam optimizer")
   186:         return optimizer
   187: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   188: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   189:     """Cosine learning rate schedule with linear warmup."""

Lines 242–244:
   239:     grad_clip = 1.0
   240:     warmup_iters = int(max_iters * 0.04)
   241:     lr_decay_iters = max_iters
   242:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   243:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   244:     CONFIG_OVERRIDES = {}
   245: 
   246:     # Apply per-method hyperparameter overrides
   247:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `lion` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 171–210:
   168:         return logits, loss
   169: 
   170:     # ── Optimizer Configuration ────────────────────────────────────────────
   171:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   172:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   173:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   174:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   175:         num_decay_params = sum(p.numel() for p in decay_params)
   176:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   177:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   178:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   179: 
   180:         class Lion(torch.optim.Optimizer):
   181:             """Lion optimizer — sign-based updates with EMA momentum."""
   182:             def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
   183:                 defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
   184:                 super().__init__(params, defaults)
   185:             @torch.no_grad()
   186:             def step(self):
   187:                 for group in self.param_groups:
   188:                     for p in group['params']:
   189:                         if p.grad is None:
   190:                             continue
   191:                         grad = p.grad
   192:                         state = self.state[p]
   193:                         if len(state) == 0:
   194:                             state['exp_avg'] = torch.zeros_like(p)
   195:                         exp_avg = state['exp_avg']
   196:                         beta1, beta2 = group['betas']
   197:                         # Weight decay first (decoupled, before update)
   198:                         if group['weight_decay'] != 0:
   199:                             p.mul_(1 - group['lr'] * group['weight_decay'])
   200:                         update = exp_avg * beta1 + grad * (1 - beta1)
   201:                         p.add_(torch.sign(update), alpha=-group['lr'])
   202:                         exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
   203: 
   204:         optim_groups = [
   205:             {'params': decay_params, 'weight_decay': weight_decay},
   206:             {'params': nodecay_params, 'weight_decay': 0.0},
   207:         ]
   208:         optimizer = Lion(optim_groups, lr=learning_rate * 0.3, betas=betas)
   209:         print("using Lion optimizer")
   210:         return optimizer
   211: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   212: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   213:     """Cosine learning rate schedule with linear warmup."""

Lines 266–268:
   263:     grad_clip = 1.0
   264:     warmup_iters = int(max_iters * 0.04)
   265:     lr_decay_iters = max_iters
   266:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   267:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   268:     CONFIG_OVERRIDES = {}
   269: 
   270:     # Apply per-method hyperparameter overrides
   271:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `muon` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 171–288:
   168:         return logits, loss
   169: 
   170:     # ── Optimizer Configuration ────────────────────────────────────────────
   171:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   172:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   173:         decay_params = [(n, p) for n, p in param_dict.items() if p.dim() >= 2]
   174:         nodecay_params = [(n, p) for n, p in param_dict.items() if p.dim() < 2]
   175:         num_decay_params = sum(p.numel() for _, p in decay_params)
   176:         num_nodecay_params = sum(p.numel() for _, p in nodecay_params)
   177:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   178:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   179: 
   180:         # Separate 2D projection weights (for Muon) from rest (for AdamW)
   181:         muon_params = [p for n, p in decay_params
   182:                        if 'wte' not in n and 'wpe' not in n and 'lm_head' not in n]
   183:         adam_decay_params = [p for n, p in decay_params
   184:                             if 'wte' in n or 'wpe' in n or 'lm_head' in n]
   185:         adam_nodecay_params = [p for _, p in nodecay_params]
   186: 
   187:         class Muon(torch.optim.Optimizer):
   188:             """Muon — MomentUm Orthogonalized by Newton-schulz.
   189:             Uses Newton-Schulz iteration to orthogonalize momentum-accumulated
   190:             gradients for 2D weight matrices. Based on modded-nanogpt.
   191:             """
   192:             def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
   193:                 defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
   194:                 super().__init__(params, defaults)
   195: 
   196:             @staticmethod
   197:             def zeroth_power_via_newtonschulz5(G, steps=5):
   198:                 """Approximate G @ (G^T G)^{-1/2} via 5 Newton-Schulz iterations."""
   199:                 assert G.ndim == 2
   200:                 a, b, c = (3.4445, -4.7750, 2.0315)
   201:                 X = G.bfloat16()
   202:                 X = X / (X.norm() + 1e-7)
   203:                 if G.size(0) > G.size(1):
   204:                     X = X.T
   205:                 for _ in range(steps):
   206:                     A = X @ X.T
   207:                     X = a * X + b * (A @ X) + c * (A @ (A @ X))
   208:                 if G.size(0) > G.size(1):
   209:                     X = X.T
   210:                 return X
   211: 
   212:             @torch.no_grad()
   213:             def step(self):
   214:                 for group in self.param_groups:
   215:                     lr = group['lr']
   216:                     momentum = group['momentum']
   217:                     wd = group.get('weight_decay', 0.0)
   218:                     for p in group['params']:
   219:                         if p.grad is None:
   220:                             continue
   221:                         # Decoupled weight decay (before update)
   222:                         if wd > 0:
   223:                             p.mul_(1 - lr * wd)
   224:                         g = p.grad
   225:                         state = self.state[p]
   226:                         if len(state) == 0:
   227:                             state['momentum_buffer'] = torch.zeros_like(g)
   228:                         buf = state['momentum_buffer']
   229:                         # EMA momentum: buf = (1-beta)*grad + beta*buf
   230:                         buf.lerp_(g, 1.0 - momentum)
   231:                         # Nesterov: update = (1-beta)*grad + beta*buf
   232:                         nesterov_g = g.lerp(buf, momentum)
   233:                         if nesterov_g.dim() == 2:
   234:                             orig_shape = nesterov_g.shape
   235:                             # Split fused QKV (c_attn: 3*n_embd x n_embd) into 3 parts
   236:                             if orig_shape[0] == 3 * orig_shape[1]:
   237:                                 parts = nesterov_g.split(orig_shape[1])
   238:                                 update = torch.cat([
   239:                                     self.zeroth_power_via_newtonschulz5(part, steps=group['ns_steps'])
   240:                                     for part in parts
   241:                                 ])
   242:                                 scale = max(1, orig_shape[0] // orig_shape[1]) ** 0.5
   243:                             else:
   244:                                 update = self.zeroth_power_via_newtonschulz5(nesterov_g, steps=group['ns_steps'])
   245:                                 scale = max(1, orig_shape[0] / orig_shape[1]) ** 0.5
   246:                             p.data.add_(update.to(p.dtype), alpha=-lr * scale)
   247:                         else:
   248:                             # Fallback: plain SGD with momentum for non-2D
   249:                             p.add_(buf, alpha=-lr)
   250: 
   251:         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   252:         use_fused = fused_available and device_type == 'cuda'
   253:         extra_args = dict(fused=True) if use_fused else dict()
   254: 
   255:         # Muon LR is typically ~0.02, much higher than Adam LR (~6e-4).
   256:         # Use lr_scale so the training loop's LR schedule scales Muon proportionally.
   257:         muon_base_lr = 0.02
   258:         muon_lr_scale = muon_base_lr / learning_rate
   259:         # Weight decay 0.1 follows the Muon-family task default.
   260:         # "Muon is Scalable" (arXiv:2502.16982) identifies weight decay as
   261:         # crucial for scaling Muon beyond 124M models.
   262:         muon_opt = Muon([{'params': muon_params, 'lr_scale': muon_lr_scale}],
   263:                         lr=muon_base_lr, momentum=0.95, weight_decay=0.1)
   264:         # AdamW for embeddings/norms uses the base config weight_decay (0.1).
   265:         adam_groups = [
   266:             {'params': adam_decay_params, 'weight_decay': weight_decay},
   267:             {'params': adam_nodecay_params, 'weight_decay': 0.0},
   268:         ]
   269:         adam_opt = torch.optim.AdamW(adam_groups, lr=learning_rate, betas=betas, **extra_args)
   270: 
   271:         class CombinedOptimizer:
   272:             """Combines Muon (for projections) with AdamW (for embeddings/norms)."""
   273:             def __init__(self, optimizers):
   274:                 self.optimizers = optimizers
   275:                 self.param_groups = []
   276:                 for opt in optimizers:
   277:                     self.param_groups.extend(opt.param_groups)
   278:             def zero_grad(self, set_to_none=True):
   279:                 for opt in self.optimizers:
   280:                     opt.zero_grad(set_to_none=set_to_none)
   281:             def step(self):
   282:                 for opt in self.optimizers:
   283:                     opt.step()
   284:             def state_dict(self):
   285:                 return [opt.state_dict() for opt in self.optimizers]
   286: 
   287:         print(f"using Muon (lr={muon_base_lr}, scale={muon_lr_scale:.1f}) + AdamW combined optimizer")
   288:         return CombinedOptimizer([muon_opt, adam_opt])
   289: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   290: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   291:     """Cosine learning rate schedule with linear warmup."""

Lines 344–346:
   341:     grad_clip = 1.0
   342:     warmup_iters = int(max_iters * 0.04)
   343:     lr_decay_iters = max_iters
   344:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   345:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   346:     CONFIG_OVERRIDES = {'learning_rate': 1e-3}
   347: 
   348:     # Apply per-method hyperparameter overrides
   349:     for _k, _v in CONFIG_OVERRIDES.items():
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
