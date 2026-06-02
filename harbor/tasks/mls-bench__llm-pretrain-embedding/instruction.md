# MLS-Bench: llm-pretrain-embedding

# LLM Pretraining: Embedding Strategy Optimization

## Research Question
Design an improved embedding strategy for GPT-style language model pretraining. The change should reduce validation loss compared to standard learned token + position embeddings with weight tying, while remaining a modular embedding-level intervention.

## Background
The default scheme uses:
- Learned token embedding (`wte`) of shape `(vocab_size, n_embd)`.
- Learned absolute position embedding (`wpe`) of shape `(block_size, n_embd)`.
- **Tied weights** between the input token embedding and the output `lm_head` projection (Press & Wolf, "Using the Output Embedding to Improve Language Models", 2016/2017, arXiv:1608.05859).

Common alternatives studied at this layer:
- Untied input/output embeddings.
- Hash-based / bigram / n-gram embeddings to inject sub-token co-occurrence statistics.
- **Value embeddings** (popularized in the modded-nanogpt speedrun, originally inspired by Zhou et al., 2024): a separate embedding table whose output is added to the *value* projections inside attention layers — typically gated and inserted at a few specific layers.

## What you can modify
The `TokenEmbedding` class in `nanoGPT/custom_pretrain.py`:
- Token embedding representation (default: learned token + position embeddings).
- Weight-tying strategy (default: input embedding shares weights with output `lm_head`).
- Additional embedding sources (e.g., n-gram, hash-based).
- Per-layer value embeddings injected via `get_value_embed(layer_idx)`.

### Interface
Your `TokenEmbedding` class must implement:
- `forward(idx) -> x` — takes token indices `(B, T)`, returns embeddings `(B, T, n_embd)`.
- `get_lm_head_weight() -> Tensor` — returns the weight tensor used for the output projection.
- `get_num_pos_params() -> int` — returns the count of position parameters (excluded from the reported parameter count).
- `get_value_embed(layer_idx) -> Optional[Tensor]` — optional per-layer value-embedding residual `(B, T, n_embd)` or `None`.

## Reference baselines
- `untied` — break weight tying between input embedding and `lm_head`.
- `bigram_hash` — hash-based bigram embeddings additive to the token embedding.
- `value_embed` — value-style per-layer embedding injection.

## Fixed Pipeline
- The corpus, tokenizer, training loop, optimizer, and unrelated transformer blocks are fixed.
- The benchmark's parameter accounting excludes `get_num_pos_params()` from the reported count, so simply scaling capacity through positional parameters is not a valid escape.

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/nanoGPT/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `nanoGPT/custom_pretrain.py`
- editable lines **115–140**
- editable lines **265–267**


Other files you may **read** for context (do not modify):
- `nanoGPT/model.py`


## Readable Context


### `nanoGPT/custom_pretrain.py`  [EDITABLE — lines 115–140, lines 265–267 only]

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
    48:         self.use_pos_emb = True
    49: 
    50:     def forward(self, x):
    51:         B, T, C = x.size()
    52:         q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    53:         k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    54:         q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    55:         v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    56:         if self.flash:
    57:             y = torch.nn.functional.scaled_dot_product_attention(
    58:                 q, k, v, attn_mask=None,
    59:                 dropout_p=self.dropout if self.training else 0, is_causal=True)
    60:         else:
    61:             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    62:             att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    63:             att = F.softmax(att, dim=-1)
    64:             att = self.attn_dropout(att)
    65:             y = att @ v
    66:         y = y.transpose(1, 2).contiguous().view(B, T, C)
    67:         y = self.resid_dropout(self.c_proj(y))
    68:         return y
    69: 
    70: # ── Feed-Forward Network ──────────────────────────────────────────────────
    71: class MLP(nn.Module):
    72:     def __init__(self, config):
    73:         super().__init__()
    74:         self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
    75:         self.gelu = nn.GELU()
    76:         self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    77:         self.dropout = nn.Dropout(config.dropout)
    78: 
    79:     def forward(self, x):
    80:         x = self.c_fc(x)
    81:         x = self.gelu(x)
    82:         x = self.c_proj(x)
    83:         x = self.dropout(x)
    84:         return x
    85: 
    86: # ── Transformer Block ─────────────────────────────────────────────────────
    87: class Block(nn.Module):
    88:     def __init__(self, config):
    89:         super().__init__()
    90:         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
    91:         self.attn = CausalSelfAttention(config)
    92:         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
    93:         self.mlp = MLP(config)
    94: 
    95:     def forward(self, x):
    96:         x = x + self.attn(self.ln_1(x))
    97:         x = x + self.mlp(self.ln_2(x))
    98:         return x
    99: 
   100: # ============================================================================
   101: # GPT Model
   102: # ============================================================================
   103: 
   104: @dataclass
   105: class GPTConfig:
   106:     block_size: int = 1024
   107:     vocab_size: int = 50304
   108:     n_layer: int = 12
   109:     n_head: int = 12
   110:     n_embd: int = 768
   111:     dropout: float = 0.0
   112:     bias: bool = False
   113: 
   114: # ── Embedding Strategy ────────────────────────────────────────────────────
   115: class TokenEmbedding(nn.Module):
   116:     """Token + position embedding with weight tying to lm_head."""
   117:     def __init__(self, config):
   118:         super().__init__()
   119:         self.wte = nn.Embedding(config.vocab_size, config.n_embd)
   120:         self.wpe = nn.Embedding(config.block_size, config.n_embd)
   121:         self.drop = nn.Dropout(config.dropout)
   122:         self.block_size = config.block_size
   123:         self.n_embd = config.n_embd
   124:         self.vocab_size = config.vocab_size
   125: 
   126:     def forward(self, idx):
   127:         b, t = idx.size()
   128:         tok_emb = self.wte(idx)
   129:         pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
   130:         pos_emb = self.wpe(pos)
   131:         return self.drop(tok_emb + pos_emb)
   132: 
   133:     def get_lm_head_weight(self):
   134:         """Return weight for the language model head (tied by default)."""
   135:         return self.wte.weight
   136: 
   137:     def get_num_pos_params(self):
   138:         """Return number of position embedding parameters (excluded from param count)."""
   139:         return self.wpe.weight.numel()
   140: 
   141: class GPT(nn.Module):
   142:     def __init__(self, config):
   143:         super().__init__()
   144:         self.config = config
   145:         self.embedding = TokenEmbedding(config)
   146:         self.transformer = nn.ModuleDict(dict(
   147:             h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
   148:             ln_f=LayerNorm(config.n_embd, bias=config.bias),
   149:         ))
   150:         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
   151:         self.lm_head.weight = self.embedding.get_lm_head_weight()
   152:         self.apply(self._init_weights)
   153:         for pn, p in self.named_parameters():
   154:             if pn.endswith('c_proj.weight'):
   155:                 torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
   156:         print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
   157: 
   158:     def get_num_params(self, non_embedding=True):
   159:         n_params = sum(p.numel() for p in self.parameters())
   160:         if non_embedding:
   161:             n_params -= self.embedding.get_num_pos_params()
   162:         return n_params
   163: 
   164:     def _init_weights(self, module):
   165:         if isinstance(module, nn.Linear):
   166:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   167:             if module.bias is not None:
   168:                 torch.nn.init.zeros_(module.bias)
   169:         elif isinstance(module, nn.Embedding):
   170:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   171: 
   172:     def forward(self, idx, targets=None):
   173:         b, t = idx.size()
   174:         assert t <= self.config.block_size
   175:         x = self.embedding(idx)
   176:         for i, block in enumerate(self.transformer.h):
   177:             # Support optional value embedding injection from TokenEmbedding
   178:             ve = getattr(self.embedding, 'get_value_embed', lambda _: None)(i)
   179:             if ve is not None:
   180:                 x = x + ve
   181:             x = block(x)
   182:         x = self.transformer.ln_f(x)
   183:         if targets is not None:
   184:             logits = self.lm_head(x)
   185:             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
   186:         else:
   187:             logits = self.lm_head(x[:, [-1], :])
   188:             loss = None
   189:         return logits, loss
   190: 
   191:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   192:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   193:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   194:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   195:         optim_groups = [
   196:             {'params': decay_params, 'weight_decay': weight_decay},
   197:             {'params': nodecay_params, 'weight_decay': 0.0},
   198:         ]
   199:         num_decay_params = sum(p.numel() for p in decay_params)
   200:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   201:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   202:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   203:         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   204:         use_fused = fused_available and device_type == 'cuda'
   205:         extra_args = dict(fused=True) if use_fused else dict()
   206:         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
   207:         print(f"using fused AdamW: {use_fused}")
   208:         return optimizer
   209: 
   210: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   211: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   212:     """Cosine learning rate schedule with linear warmup."""
   213:     if it < warmup_iters:
   214:         return learning_rate * (it + 1) / (warmup_iters + 1)
   215:     if it > lr_decay_iters:
   216:         return min_lr
   217:     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
   218:     assert 0 <= decay_ratio <= 1
   219:     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
   220:     return min_lr + coeff * (learning_rate - min_lr)
   221: 
   222: # ============================================================================
   223: # Data Loading
   224: # ============================================================================
   225: 
   226: def get_batch(data, batch_size, block_size, device):
   227:     """Get a random batch from a pre-opened memmap (nanoGPT style)."""
   228:     ix = torch.randint(len(data) - block_size, (batch_size,))
   229:     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
   230:     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
   231:     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
   232:     return x, y
   233: 
   234: # ============================================================================
   235: # Training Script
   236: # ============================================================================
   237: 
   238: if __name__ == '__main__':
   239:     # ── Configuration from environment ──
   240:     output_dir = os.environ.get('OUTPUT_DIR', 'out')
   241:     seed = int(os.environ.get('SEED', 1337))
   242:     data_dir = os.environ.get('DATA_DIR', '/data/climbmix')
   243: 
   244:     # Model config from environment
   245:     n_layer = int(os.environ.get('N_LAYER', 12))
   246:     n_head = int(os.environ.get('N_HEAD', 12))
   247:     n_embd = int(os.environ.get('N_EMBD', 768))
   248: 
   249:     # Training hyperparameters (overridable via env for different model sizes)
   250:     max_iters = int(os.environ.get('MAX_ITERS', 5000))
   251:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 500))
   252:     eval_iters = 200
   253:     log_interval = 10
   254:     batch_size = int(os.environ.get('BATCH_SIZE', 12))
   255:     block_size = 1024
   256:     gradient_accumulation_steps = int(os.environ.get('GRAD_ACCUM', 5))
   257:     learning_rate = float(os.environ.get('LEARNING_RATE', 6e-4))
   258:     min_lr = learning_rate / 10
   259:     weight_decay = 1e-1
   260:     beta1 = 0.9
   261:     beta2 = 0.95
   262:     grad_clip = 1.0
   263:     warmup_iters = int(max_iters * 0.04)
   264:     lr_decay_iters = max_iters
   265:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   266:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   267:     CONFIG_OVERRIDES = {}
   268: 
   269:     # Apply per-method hyperparameter overrides
   270:     for _k, _v in CONFIG_OVERRIDES.items():
   271:         if _k == 'learning_rate': learning_rate = _v; min_lr = learning_rate / 10
   272:         elif _k == 'weight_decay': weight_decay = _v
   273:         elif _k == 'warmup_iters': warmup_iters = _v
   274:         elif _k == 'min_lr': min_lr = _v
   275:         elif _k == 'grad_clip': grad_clip = _v
   276: 
   277:     compile_model = True
   278:     dtype = 'bfloat16'
   279: 
   280:     # ── DDP Setup ──
   281:     ddp = int(os.environ.get('RANK', -1)) != -1
   282:     if ddp:
   283:         import torch.distributed as dist
   284:         from torch.nn.parallel import DistributedDataParallel as DDP
   285:         dist.init_process_group(backend='nccl')
   286:         ddp_rank = int(os.environ['RANK'])
   287:         ddp_local_rank = int(os.environ['LOCAL_RANK'])
   288:         ddp_world_size = int(os.environ['WORLD_SIZE'])
   289:         device = f'cuda:{ddp_local_rank}'
   290:         torch.cuda.set_device(device)
   291:         master_process = ddp_rank == 0
   292:         seed_offset = ddp_rank
   293:         assert gradient_accumulation_steps % ddp_world_size == 0
   294:         gradient_accumulation_steps //= ddp_world_size
   295:     else:
   296:         master_process = True
   297:         device = 'cuda'
   298:         seed_offset = 0
   299: 
   300:     # ── Setup ──
   301:     device_type = 'cuda'
   302:     torch.manual_seed(seed + seed_offset)
   303:     torch.backends.cuda.matmul.allow_tf32 = True
   304:     torch.backends.cudnn.allow_tf32 = True
   305:     ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
   306:     ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
   307:     if master_process:
   308:         os.makedirs(output_dir, exist_ok=True)
   309: 
   310:     tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
   311:     if ddp:
   312:         tokens_per_iter *= int(os.environ.get('WORLD_SIZE', 1))
   313:     if master_process:
   314:         print(f"tokens per iteration will be: {tokens_per_iter:,}")
   315: 
   316:     # ── Load Data ──
   317:     train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
   318:     val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
   319:     if master_process:
   320:         print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
   321: 
   322:     # ── Model Init ──
   323:     model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
   324:                       block_size=block_size, bias=False, vocab_size=50304, dropout=0.0)
   325:     gptconf = GPTConfig(**model_args)
   326:     model = GPT(gptconf)
   327:     model.to(device)
   328: 
   329:     # ── Parameter Budget Check ──
   330:     # Largest baselines (bigram_hash, value_embed) each add ~5*vocab_size*n_embd
   331:     # embedding parameters on top of the base model. Budget = 1.05x that max.
   332:     _base_params = (
   333:         # Transformer blocks: each has attn(4*E*E) + mlp(8*E*E) + 2*norms(2*E)
   334:         n_layer * (4 * n_embd * n_embd + 8 * n_embd * n_embd + 2 * n_embd)
   335:         # Token embedding (tied with lm_head)
   336:         + gptconf.vocab_size * n_embd
   337:         # Position embedding
   338:         + block_size * n_embd
   339:     )
   340:     _max_baseline_extra = 5 * gptconf.vocab_size * n_embd + n_layer + 10  # bigram_hash / value_embed
   341:     _param_budget = int((_base_params + _max_baseline_extra) * 1.05)
   342:     _total_params = sum(p.numel() for p in model.parameters())
   343:     if master_process:
   344:         print(f"Parameter budget check: {_total_params:,} params (budget: {_param_budget:,})")
   345: 
   346:     scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
   347:     optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
   348: 
   349:     if compile_model:
   350:         if master_process:
   351:             print("compiling the model...")
   352:         model = torch.compile(model)
   353: 
   354:     if ddp:
   355:         model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
   356: 
   357:     # ── Evaluation ──
   358:     @torch.no_grad()
   359:     def estimate_loss():
   360:         out = {}
   361:         raw = model.module if ddp else model
   362:         raw.eval()
   363:         for split, data in [('train', train_data), ('val', val_data)]:
   364:             losses = torch.zeros(eval_iters)
   365:             for k in range(eval_iters):
   366:                 X, Y = get_batch(data, batch_size, block_size, device)
   367:                 with ctx:
   368:                     logits, loss = raw(X, Y)
   369:                 losses[k] = loss.item()
   370:             out[split] = losses.mean()
   371:         raw.train()
   372:         return out
   373: 
   374:     # ── Training Loop ──
   375:     t0 = time.time()
   376:     best_val_loss = 1e9
   377: 
   378:     for iter_num in range(max_iters + 1):
   379:         lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
   380:         for param_group in optimizer.param_groups:
   381:             param_group['lr'] = lr
   382: 
   383:         if iter_num % eval_interval == 0 and master_process:
   384:             losses = estimate_loss()
   385:             train_loss = losses['train'].item()
   386:             val_loss = losses['val'].item()
   387:             print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
   388:             print(f"TRAIN_METRICS: step={iter_num}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
   389:             if val_loss < best_val_loss:
   390:                 best_val_loss = val_loss
   391: 
   392:         for micro_step in range(gradient_accumulation_steps):
   393:             if ddp:
   394:                 model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
   395:             with ctx:
   396:                 X, Y = get_batch(train_data, batch_size, block_size, device)
   397:                 logits, loss = model(X, Y)
   398:                 loss = loss / gradient_accumulation_steps
   399:             scaler.scale(loss).backward()
   400: 
   401:         if grad_clip != 0.0:
   402:             scaler.unscale_(optimizer)
   403:             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
   404:         scaler.step(optimizer)
   405:         scaler.update()
   406:         optimizer.zero_grad(set_to_none=True)
   407: 
   408:         t1 = time.time()
   409:         dt = t1 - t0
   410:         t0 = t1
   411:         if iter_num % log_interval == 0 and iter_num > 0 and master_process:
   412:             lossf = loss.item() * gradient_accumulation_steps
   413:             print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
   414: 
   415:     # ── Free training state to reclaim GPU memory ──
   416:     del optimizer, scaler
   417:     import gc; gc.collect()
   418:     torch.cuda.empty_cache()
   419: 
   420:     # ── Final Evaluation ──
   421:     if master_process:
   422:         losses = estimate_loss()
   423:         val_loss = losses['val'].item()
   424:         train_loss = losses['train'].item()
   425:         print(f"Final: train loss {train_loss:.4f}, val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}")
   426: 
   427:         # ── PPL on benchmark datasets ──
   428:         eval_dir = os.environ.get('EVAL_DIR', '/data/eval')
   429:         raw = model.module if ddp else model
   430:         raw.eval()
   431:         eval_datasets = ['wikitext2', 'lambada']
   432:         ppl_results = {}
   433:         for ds_name in eval_datasets:
   434:             ds_path = os.path.join(eval_dir, f'{ds_name}.bin')
   435:             if not os.path.exists(ds_path):
   436:                 print(f"Eval dataset not found: {ds_path}")
   437:                 continue
   438:             data = np.memmap(ds_path, dtype=np.uint16, mode='r')
   439:             n_tokens = len(data)
   440:             # Process in non-overlapping chunks of block_size
   441:             total_loss = 0.0
   442:             n_chunks = 0
   443:             with torch.no_grad():
   444:                 for start in range(0, n_tokens - block_size, block_size):
   445:                     x = torch.from_numpy(data[start:start+block_size].astype(np.int64)).unsqueeze(0).to(device)
   446:                     y = torch.from_numpy(data[start+1:start+1+block_size].astype(np.int64)).unsqueeze(0).to(device)
   447:                     with ctx:
   448:                         _, loss = raw(x, y)
   449:                     total_loss += loss.item()
   450:                     n_chunks += 1
   451:             avg_loss = total_loss / n_chunks
   452:             ppl = math.exp(avg_loss)
   453:             ppl_results[ds_name] = ppl
   454:             print(f"PPL {ds_name}: {ppl:.2f} (avg_loss={avg_loss:.4f}, {n_chunks} chunks)")
   455: 
   456:         ppl_str = ', '.join(f'{k}_ppl={v:.2f}' for k, v in ppl_results.items())
   457:         print(f"TEST_METRICS: val_loss={val_loss:.4f}, {ppl_str}", flush=True)
   458: 
   459:         # ── Save checkpoint for downstream evaluation (lm-eval-harness) ──
   460:         import shutil
   461:         env_label = os.environ.get('ENV', 'model')
   462:         # Unwrap torch.compile to get clean state_dict keys
   463:         save_model = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
   464:         ckpt_data = {'model_state_dict': save_model.state_dict(), 'model_args': model_args}
   465:         ckpt_path = os.path.join(output_dir, f'ckpt_{env_label}.pt')
   466:         torch.save(ckpt_data, ckpt_path)
   467:         print(f"Checkpoint saved to {ckpt_path}")
   468:         src_path = os.path.join(output_dir, f'model_source_{env_label}.py')
   469:         shutil.copy2(os.path.abspath(__file__), src_path)
   470:         print(f"Model source saved to {src_path}")
   471: 
   472:     if ddp:
   473:         dist.destroy_process_group()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `untied` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 115–140:
   112:     bias: bool = False
   113: 
   114: # ── Embedding Strategy ────────────────────────────────────────────────────
   115: class TokenEmbedding(nn.Module):
   116:     """Token + position embedding with UNTIED lm_head weight."""
   117:     def __init__(self, config):
   118:         super().__init__()
   119:         self.wte = nn.Embedding(config.vocab_size, config.n_embd)
   120:         self.wpe = nn.Embedding(config.block_size, config.n_embd)
   121:         self.drop = nn.Dropout(config.dropout)
   122:         self.block_size = config.block_size
   123:         self.n_embd = config.n_embd
   124:         self.vocab_size = config.vocab_size
   125:         # Separate output projection weight (not tied to wte)
   126:         self._lm_head_weight = nn.Parameter(torch.empty(config.vocab_size, config.n_embd))
   127:         nn.init.zeros_(self._lm_head_weight)
   128: 
   129:     def forward(self, idx):
   130:         b, t = idx.size()
   131:         tok_emb = self.wte(idx)
   132:         pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
   133:         pos_emb = self.wpe(pos)
   134:         return self.drop(tok_emb + pos_emb)
   135: 
   136:     def get_lm_head_weight(self):
   137:         return self._lm_head_weight
   138: 
   139:     def get_num_pos_params(self):
   140:         return self.wpe.weight.numel()
   141: class GPT(nn.Module):
   142:     def __init__(self, config):
   143:         super().__init__()

Lines 265–267:
   262:     grad_clip = 1.0
   263:     warmup_iters = int(max_iters * 0.04)
   264:     lr_decay_iters = max_iters
   265:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   266:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   267:     CONFIG_OVERRIDES = {}
   268: 
   269:     # Apply per-method hyperparameter overrides
   270:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `value_embed` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 115–164:
   112:     bias: bool = False
   113: 
   114: # ── Embedding Strategy ────────────────────────────────────────────────────
   115: class TokenEmbedding(nn.Module):
   116:     """Token + position embedding with value embeddings for selected layers."""
   117:     def __init__(self, config):
   118:         super().__init__()
   119:         self.wte = nn.Embedding(config.vocab_size, config.n_embd)
   120:         self.wpe = nn.Embedding(config.block_size, config.n_embd)
   121:         self.drop = nn.Dropout(config.dropout)
   122:         self.block_size = config.block_size
   123:         self.n_embd = config.n_embd
   124:         self.vocab_size = config.vocab_size
   125:         self.n_layer = config.n_layer
   126:         # Value embeddings: 5 tables injected into selected layers (like modded-nanogpt)
   127:         self.n_ve = 5
   128:         self.vte = nn.Embedding(config.vocab_size * self.n_ve, config.n_embd)
   129:         nn.init.normal_(self.vte.weight, mean=0.0, std=0.01)
   130:         # Per-VE learnable blending coefficient (lambda)
   131:         self.ve_lambda = nn.Parameter(torch.full((self.n_ve,), 0.5))
   132:         # Injection layers: layer 1, 2, and last 3 layers
   133:         self._ve_layers = None
   134:         self._cached_ve = None
   135: 
   136:     def forward(self, idx):
   137:         b, t = idx.size()
   138:         tok_emb = self.wte(idx)
   139:         pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
   140:         pos_emb = self.wpe(pos)
   141:         # Compute injection layer indices: layer 1, 2, and last 3 layers
   142:         if self._ve_layers is None:
   143:             self._ve_layers = [1, 2, self.n_layer - 3, self.n_layer - 2, self.n_layer - 1]
   144:         # Cache per-VE value embeddings (5 separate lookups into partitioned table)
   145:         vs = self.vocab_size
   146:         self._cached_ve = {}
   147:         for i, layer_idx in enumerate(self._ve_layers):
   148:             offset_idx = idx + i * vs  # offset into partition i
   149:             self._cached_ve[layer_idx] = self.vte(offset_idx)
   150:         return self.drop(tok_emb + pos_emb)
   151: 
   152:     def get_value_embed(self, layer_idx):
   153:         """Get value embedding residual for a given layer, or None."""
   154:         if self._cached_ve is None or layer_idx not in self._cached_ve:
   155:             return None
   156:         ve_idx = self._ve_layers.index(layer_idx)
   157:         lamb = self.ve_lambda[ve_idx]
   158:         return lamb * self._cached_ve[layer_idx]
   159: 
   160:     def get_lm_head_weight(self):
   161:         return self.wte.weight
   162: 
   163:     def get_num_pos_params(self):
   164:         return self.wpe.weight.numel()
   165: class GPT(nn.Module):
   166:     def __init__(self, config):
   167:         super().__init__()

Lines 289–291:
   286:     grad_clip = 1.0
   287:     warmup_iters = int(max_iters * 0.04)
   288:     lr_decay_iters = max_iters
   289:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   290:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   291:     CONFIG_OVERRIDES = {}
   292: 
   293:     # Apply per-method hyperparameter overrides
   294:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `bigram_hash` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 115–168:
   112:     bias: bool = False
   113: 
   114: # ── Embedding Strategy ────────────────────────────────────────────────────
   115: class TokenEmbedding(nn.Module):
   116:     """Token + position + bigram hash embedding."""
   117:     def __init__(self, config):
   118:         super().__init__()
   119:         self.wte = nn.Embedding(config.vocab_size, config.n_embd)
   120:         self.wpe = nn.Embedding(config.block_size, config.n_embd)
   121:         self.drop = nn.Dropout(config.dropout)
   122:         self.block_size = config.block_size
   123:         self.n_embd = config.n_embd
   124:         self.vocab_size = config.vocab_size
   125:         # Bigram hash embedding: 5x vocab for hash collision reduction
   126:         self.bigram_vocab_size = config.vocab_size * 5
   127:         self.bigram_embed = nn.Embedding(self.bigram_vocab_size, config.n_embd)
   128:         nn.init.zeros_(self.bigram_embed.weight)
   129:         self.n_layer = config.n_layer
   130:         # Per-layer learnable scaling for bigram embedding injection
   131:         self.bigram_lambdas = nn.Parameter(torch.full((config.n_layer,), 0.1))
   132:         self._cached_bigram = None
   133: 
   134:     def _bigram_hash(self, idx):
   135:         """Compute bigram hash indices from consecutive token pairs."""
   136:         rand_int_1 = 36313
   137:         rand_int_2 = 27191
   138:         mod = self.bigram_vocab_size - 1
   139:         x = idx.to(torch.int32)
   140:         out = torch.zeros_like(x)
   141:         # Position 0: no previous token, use reserved index
   142:         out[:, 0] = mod
   143:         # Positions 1+: XOR hash of (current, previous) token pair
   144:         out[:, 1:] = torch.bitwise_xor(
   145:             rand_int_1 * x[:, 1:],
   146:             rand_int_2 * x[:, :-1]
   147:         ) % mod
   148:         return out.long()
   149: 
   150:     def forward(self, idx):
   151:         b, t = idx.size()
   152:         tok_emb = self.wte(idx)
   153:         pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
   154:         pos_emb = self.wpe(pos)
   155:         self._cached_bigram = self.bigram_embed(self._bigram_hash(idx))
   156:         return self.drop(tok_emb + pos_emb)
   157: 
   158:     def get_value_embed(self, layer_idx):
   159:         """Inject bigram embedding at every layer with learnable scaling."""
   160:         if self._cached_bigram is None or layer_idx >= self.n_layer:
   161:             return None
   162:         return self.bigram_lambdas[layer_idx] * self._cached_bigram
   163: 
   164:     def get_lm_head_weight(self):
   165:         return self.wte.weight
   166: 
   167:     def get_num_pos_params(self):
   168:         return self.wpe.weight.numel()
   169: class GPT(nn.Module):
   170:     def __init__(self, config):
   171:         super().__init__()

Lines 293–295:
   290:     grad_clip = 1.0
   291:     warmup_iters = int(max_iters * 0.04)
   292:     lr_decay_iters = max_iters
   293:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   294:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   295:     CONFIG_OVERRIDES = {}
   296: 
   297:     # Apply per-method hyperparameter overrides
   298:     for _k, _v in CONFIG_OVERRIDES.items():
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
