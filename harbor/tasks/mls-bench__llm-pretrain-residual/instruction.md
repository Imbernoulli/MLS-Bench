# MLS-Bench: llm-pretrain-residual

# LLM Pretraining: Residual Connection Strategy

## Research Question
Improve the residual-connection strategy of a GPT-style language model. The default uses standard Pre-LN additive residuals (`x = x + sublayer(LN(x))`) in each transformer block. The goal is to redesign how information flows through the residual stream across layers to lower validation loss.

## Background

### Standard residual stream
The default GPT-2 block:
```python
x = x + self.attn(self.ln_1(x))   # attention sublayer
x = x + self.mlp(self.ln_2(x))    # MLP sublayer
```
The residual stream is the model's information highway; its design affects gradient flow, feature reuse, and training stability.

### Research directions
- **Per-layer residual scaling** — learnable scalars that gate the contribution of each sublayer. Examples: ReZero (Bachlechner et al., "ReZero is All You Need: Fast Convergence at Large Depth", 2020, arXiv:2003.04887), SkipInit, and the per-layer scalar gates used in modded-nanogpt.
- **Initial-embedding (x0) blending** — re-blend the token embedding back into the residual at each layer to preserve token identity (used in modded-nanogpt and related speedrun work).
- **Hyper-Connections** — Zhu et al. (ByteDance), "Hyper-Connections", ICLR 2025, arXiv:2409.19606. Maintain `m` parallel residual streams with learned transition matrices, addressing the gradient-vanishing / representation-collapse seesaw of vanilla residuals.
- **Attention-over-layers residuals** — softmax attention over all previous layer outputs to dynamically pick which past representations to combine, a recurring idea in recent open-source LM work.

## What you can modify
In `nanoGPT/custom_pretrain.py`:

- **`Block` class** — per-block residual behavior; how attention and MLP outputs are combined with the residual stream within each block.
- **`GPT.__init__`** — additional parameters for your residual strategy (per-layer scalars, transition matrices, query vectors, etc.).
- **The block loop in `GPT.forward`** — how blocks are called and how their outputs are accumulated (e.g., multi-stream processing, attention over layer outputs).
- **`configure_optimizers`** — assign new parameters to optimizer groups with appropriate LR / weight decay.
- **`CONFIG_OVERRIDES`** dict — adjust LR / weight decay if your design needs it.

### Interface contract
- `CausalSelfAttention`, `MLP`, `LayerNorm`, and `GPTConfig` are fixed.
- `Block.forward` must accept `x` and return a tensor of the same shape.
- `GPT.forward` must accept `(idx, targets=None)` and return `(logits, loss)`.

## Reference baselines
- `vanilla` — standard additive Pre-LN residuals (the default).
- `learned_scaling` — ReZero-style per-layer learnable scalar on each sublayer.
- `prores` — initial-embedding blending into the residual stream.
- `full_attnres` — attention over all previous layer outputs.

## Fixed Pipeline
- **Model**: a GPT-2-style decoder-only transformer.
- **Training**: standard warmup + cosine LR decay.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/nanoGPT/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `nanoGPT/custom_pretrain.py`
- editable lines **88–99**
- editable lines **128–130**
- editable lines **162–164**
- editable lines **175–192**
- editable lines **251–251**


Other files you may **read** for context (do not modify):
- `nanoGPT/model.py`


## Readable Context


### `nanoGPT/custom_pretrain.py`  [EDITABLE — lines 88–99, lines 128–130, lines 162–164, lines 175–192, lines 251–251 only]

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
   128:         # ── Residual stream parameters ──
   129:         # (default: none — vanilla residual x + sublayer(x) is in Block.forward)
   130:         # Add custom residual parameters here if needed.
   131:         self.apply(self._init_weights)
   132:         for pn, p in self.named_parameters():
   133:             if pn.endswith('c_proj.weight'):
   134:                 torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
   135:         print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
   136: 
   137:     def get_num_params(self, non_embedding=True):
   138:         n_params = sum(p.numel() for p in self.parameters())
   139:         if non_embedding:
   140:             n_params -= self.transformer.wpe.weight.numel()
   141:         return n_params
   142: 
   143:     def _init_weights(self, module):
   144:         if isinstance(module, nn.Linear):
   145:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   146:             if module.bias is not None:
   147:                 torch.nn.init.zeros_(module.bias)
   148:         elif isinstance(module, nn.Embedding):
   149:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   150: 
   151:     def forward(self, idx, targets=None):
   152:         device = idx.device
   153:         b, t = idx.size()
   154:         assert t <= self.config.block_size
   155:         tok_emb = self.transformer.wte(idx)
   156:         x = self.transformer.drop(tok_emb)
   157:         # Conditionally add learned position embeddings
   158:         use_pos = getattr(self.transformer.h[0].attn, 'use_pos_emb', True)
   159:         if use_pos:
   160:             pos = torch.arange(0, t, dtype=torch.long, device=device)
   161:             x = x + self.transformer.wpe(pos)
   162:         # ── Residual stream: iterate through transformer blocks ──
   163:         for block in self.transformer.h:
   164:             x = block(x)
   165:         x = self.transformer.ln_f(x)
   166:         if targets is not None:
   167:             logits = self.lm_head(x)
   168:             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
   169:         else:
   170:             logits = self.lm_head(x[:, [-1], :])
   171:             loss = None
   172:         return logits, loss
   173: 
   174:     # ── Optimizer Configuration ────────────────────────────────────────────
   175:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   176:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   177:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   178:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   179:         optim_groups = [
   180:             {'params': decay_params, 'weight_decay': weight_decay},
   181:             {'params': nodecay_params, 'weight_decay': 0.0},
   182:         ]
   183:         num_decay_params = sum(p.numel() for p in decay_params)
   184:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   185:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   186:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   187:         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   188:         use_fused = fused_available and device_type == 'cuda'
   189:         extra_args = dict(fused=True) if use_fused else dict()
   190:         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
   191:         print(f"using fused AdamW: {use_fused}")
   192:         return optimizer
   193: 
   194: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   195: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   196:     """Cosine learning rate schedule with linear warmup."""
   197:     if it < warmup_iters:
   198:         return learning_rate * (it + 1) / (warmup_iters + 1)
   199:     if it > lr_decay_iters:
   200:         return min_lr
   201:     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
   202:     assert 0 <= decay_ratio <= 1
   203:     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
   204:     return min_lr + coeff * (learning_rate - min_lr)
   205: 
   206: # ============================================================================
   207: # Data Loading
   208: # ============================================================================
   209: 
   210: def get_batch(data, batch_size, block_size, device):
   211:     """Get a random batch from a pre-opened memmap (nanoGPT style)."""
   212:     ix = torch.randint(len(data) - block_size, (batch_size,))
   213:     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
   214:     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
   215:     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
   216:     return x, y
   217: 
   218: # ============================================================================
   219: # Training Script
   220: # ============================================================================
   221: 
   222: if __name__ == '__main__':
   223:     # ── Configuration from environment ──
   224:     output_dir = os.environ.get('OUTPUT_DIR', 'out')
   225:     seed = int(os.environ.get('SEED', 1337))
   226:     data_dir = os.environ.get('DATA_DIR', '/data/climbmix')
   227: 
   228:     # Model config from environment
   229:     n_layer = int(os.environ.get('N_LAYER', 12))
   230:     n_head = int(os.environ.get('N_HEAD', 12))
   231:     n_embd = int(os.environ.get('N_EMBD', 768))
   232: 
   233:     # Training hyperparameters (overridable via env for different model sizes)
   234:     max_iters = int(os.environ.get('MAX_ITERS', 5000))
   235:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 500))
   236:     eval_iters = 200
   237:     log_interval = 10
   238:     batch_size = int(os.environ.get('BATCH_SIZE', 12))
   239:     block_size = 1024
   240:     gradient_accumulation_steps = int(os.environ.get('GRAD_ACCUM', 5))
   241:     learning_rate = float(os.environ.get('LEARNING_RATE', 6e-4))
   242:     min_lr = learning_rate / 10
   243:     weight_decay = 1e-1
   244:     beta1 = 0.9
   245:     beta2 = 0.95
   246:     grad_clip = 1.0
   247:     warmup_iters = int(max_iters * 0.04)
   248:     lr_decay_iters = max_iters
   249:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   250:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   251:     CONFIG_OVERRIDES = {}
   252: 
   253:     # Apply per-method hyperparameter overrides
   254:     for _k, _v in CONFIG_OVERRIDES.items():
   255:         if _k == 'learning_rate': learning_rate = _v; min_lr = learning_rate / 10
   256:         elif _k == 'weight_decay': weight_decay = _v
   257:         elif _k == 'warmup_iters': warmup_iters = _v
   258:         elif _k == 'min_lr': min_lr = _v
   259:         elif _k == 'grad_clip': grad_clip = _v
   260: 
   261:     compile_model = True
   262:     dtype = 'bfloat16'
   263: 
   264:     # ── DDP Setup ──
   265:     ddp = int(os.environ.get('RANK', -1)) != -1
   266:     if ddp:
   267:         import torch.distributed as dist
   268:         from torch.nn.parallel import DistributedDataParallel as DDP
   269:         dist.init_process_group(backend='nccl')
   270:         ddp_rank = int(os.environ['RANK'])
   271:         ddp_local_rank = int(os.environ['LOCAL_RANK'])
   272:         ddp_world_size = int(os.environ['WORLD_SIZE'])
   273:         device = f'cuda:{ddp_local_rank}'
   274:         torch.cuda.set_device(device)
   275:         master_process = ddp_rank == 0
   276:         seed_offset = ddp_rank
   277:         assert gradient_accumulation_steps % ddp_world_size == 0
   278:         gradient_accumulation_steps //= ddp_world_size
   279:     else:
   280:         master_process = True
   281:         device = 'cuda'
   282:         seed_offset = 0
   283: 
   284:     # ── Setup ──
   285:     device_type = 'cuda'
   286:     torch.manual_seed(seed + seed_offset)
   287:     torch.backends.cuda.matmul.allow_tf32 = True
   288:     torch.backends.cudnn.allow_tf32 = True
   289:     ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
   290:     ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
   291:     if master_process:
   292:         os.makedirs(output_dir, exist_ok=True)
   293: 
   294:     tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
   295:     if ddp:
   296:         tokens_per_iter *= int(os.environ.get('WORLD_SIZE', 1))
   297:     if master_process:
   298:         print(f"tokens per iteration will be: {tokens_per_iter:,}")
   299: 
   300:     # ── Load Data ──
   301:     train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
   302:     val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
   303:     if master_process:
   304:         print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
   305: 
   306:     # ── Model Init ──
   307:     model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
   308:                       block_size=block_size, bias=False, vocab_size=50304, dropout=0.0)
   309:     gptconf = GPTConfig(**model_args)
   310:     model = GPT(gptconf)
   311:     model.to(device)
   312: 
   313: 
   314:     scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
   315:     optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
   316: 
   317:     if compile_model:
   318:         if master_process:
   319:             print("compiling the model...")
   320:         model = torch.compile(model)
   321: 
   322:     if ddp:
   323:         model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
   324: 
   325:     # ── Evaluation ──
   326:     @torch.no_grad()
   327:     def estimate_loss():
   328:         out = {}
   329:         raw = model.module if ddp else model
   330:         raw.eval()
   331:         for split, data in [('train', train_data), ('val', val_data)]:
   332:             losses = torch.zeros(eval_iters)
   333:             for k in range(eval_iters):
   334:                 X, Y = get_batch(data, batch_size, block_size, device)
   335:                 with ctx:
   336:                     logits, loss = raw(X, Y)
   337:                 losses[k] = loss.item()
   338:             out[split] = losses.mean()
   339:         raw.train()
   340:         return out
   341: 
   342:     # ── Training Loop ──
   343:     t0 = time.time()
   344:     best_val_loss = 1e9
   345: 
   346:     for iter_num in range(max_iters + 1):
   347:         lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
   348:         for param_group in optimizer.param_groups:
   349:             param_group['lr'] = lr
   350: 
   351:         if iter_num % eval_interval == 0 and master_process:
   352:             losses = estimate_loss()
   353:             train_loss = losses['train'].item()
   354:             val_loss = losses['val'].item()
   355:             print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
   356:             print(f"TRAIN_METRICS: step={iter_num}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
   357:             if val_loss < best_val_loss:
   358:                 best_val_loss = val_loss
   359: 
   360:         for micro_step in range(gradient_accumulation_steps):
   361:             if ddp:
   362:                 model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
   363:             with ctx:
   364:                 X, Y = get_batch(train_data, batch_size, block_size, device)
   365:                 logits, loss = model(X, Y)
   366:                 loss = loss / gradient_accumulation_steps
   367:             scaler.scale(loss).backward()
   368: 
   369:         if grad_clip != 0.0:
   370:             scaler.unscale_(optimizer)
   371:             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
   372:         scaler.step(optimizer)
   373:         scaler.update()
   374:         optimizer.zero_grad(set_to_none=True)
   375: 
   376:         t1 = time.time()
   377:         dt = t1 - t0
   378:         t0 = t1
   379:         if iter_num % log_interval == 0 and iter_num > 0 and master_process:
   380:             lossf = loss.item() * gradient_accumulation_steps
   381:             print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
   382: 
   383:     # ── Free training state to reclaim GPU memory ──
   384:     del optimizer, scaler
   385:     import gc; gc.collect()
   386:     torch.cuda.empty_cache()
   387: 
   388:     # ── Final Evaluation ──
   389:     if master_process:
   390:         losses = estimate_loss()
   391:         val_loss = losses['val'].item()
   392:         train_loss = losses['train'].item()
   393:         print(f"Final: train loss {train_loss:.4f}, val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}")
   394: 
   395:         # ── PPL on benchmark datasets ──
   396:         eval_dir = os.environ.get('EVAL_DIR', '/data/eval')
   397:         raw = model.module if ddp else model
   398:         raw.eval()
   399:         eval_datasets = ['wikitext2', 'lambada']
   400:         ppl_results = {}
   401:         for ds_name in eval_datasets:
   402:             ds_path = os.path.join(eval_dir, f'{ds_name}.bin')
   403:             if not os.path.exists(ds_path):
   404:                 print(f"Eval dataset not found: {ds_path}")
   405:                 continue
   406:             data = np.memmap(ds_path, dtype=np.uint16, mode='r')
   407:             n_tokens = len(data)
   408:             # Process in non-overlapping chunks of block_size
   409:             total_loss = 0.0
   410:             n_chunks = 0
   411:             with torch.no_grad():
   412:                 for start in range(0, n_tokens - block_size, block_size):
   413:                     x = torch.from_numpy(data[start:start+block_size].astype(np.int64)).unsqueeze(0).to(device)
   414:                     y = torch.from_numpy(data[start+1:start+1+block_size].astype(np.int64)).unsqueeze(0).to(device)
   415:                     with ctx:
   416:                         _, loss = raw(x, y)
   417:                     total_loss += loss.item()
   418:                     n_chunks += 1
   419:             avg_loss = total_loss / n_chunks
   420:             ppl = math.exp(avg_loss)
   421:             ppl_results[ds_name] = ppl
   422:             print(f"PPL {ds_name}: {ppl:.2f} (avg_loss={avg_loss:.4f}, {n_chunks} chunks)")
   423: 
   424:         ppl_str = ', '.join(f'{k}_ppl={v:.2f}' for k, v in ppl_results.items())
   425:         print(f"TEST_METRICS: val_loss={val_loss:.4f}, {ppl_str}", flush=True)
   426: 
   427:         # ── Save checkpoint for downstream evaluation (lm-eval-harness) ──
   428:         import shutil
   429:         env_label = os.environ.get('ENV', 'model')
   430:         # Unwrap torch.compile to get clean state_dict keys
   431:         save_model = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
   432:         ckpt_data = {'model_state_dict': save_model.state_dict(), 'model_args': model_args}
   433:         ckpt_path = os.path.join(output_dir, f'ckpt_{env_label}.pt')
   434:         torch.save(ckpt_data, ckpt_path)
   435:         print(f"Checkpoint saved to {ckpt_path}")
   436:         src_path = os.path.join(output_dir, f'model_source_{env_label}.py')
   437:         shutil.copy2(os.path.abspath(__file__), src_path)
   438:         print(f"Model source saved to {src_path}")
   439: 
   440:     if ddp:
   441:         dist.destroy_process_group()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `prores` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 88–99:
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

Lines 128–132:
   125:         ))
   126:         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
   127:         self.transformer.wte.weight = self.lm_head.weight
   128:         # ── ProRes: progressive residual warmup ──
   129:         # T controls the warmup period; deeper layers take T*layer_idx steps
   130:         # to reach full contribution.  step counter is a non-parameter buffer.
   131:         self.prores_T = 1000
   132:         self.register_buffer('_prores_step', torch.zeros(1, dtype=torch.long))
   133:         self.apply(self._init_weights)
   134:         for pn, p in self.named_parameters():
   135:             if pn.endswith('c_proj.weight'):

Lines 164–179:
   161:         if use_pos:
   162:             pos = torch.arange(0, t, dtype=torch.long, device=device)
   163:             x = x + self.transformer.wpe(pos)
   164:         # ── ProRes: progressive residual warmup per block ──
   165:         # Increment step counter once per forward (training only).
   166:         if self.training:
   167:             self._prores_step += 1
   168:         step = self._prores_step.item()
   169:         T = self.prores_T
   170:         for i, block in enumerate(self.transformer.h):
   171:             block_out = block(x)
   172:             if self.training and step < T * (i + 1):
   173:                 # alpha ramps from 0 to 1 over T * layer_idx steps
   174:                 layer_idx = i + 1
   175:                 alpha = min(step / (T * layer_idx), 1.0)
   176:                 # block_out = x + delta (Pre-LN residual), so delta = block_out - x
   177:                 x = x + alpha * (block_out - x)
   178:             else:
   179:                 x = block_out
   180:         x = self.transformer.ln_f(x)
   181:         if targets is not None:
   182:             logits = self.lm_head(x)

Lines 190–207:
   187:         return logits, loss
   188: 
   189:     # ── Optimizer Configuration ────────────────────────────────────────────
   190:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   191:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   192:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   193:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   194:         optim_groups = [
   195:             {'params': decay_params, 'weight_decay': weight_decay},
   196:             {'params': nodecay_params, 'weight_decay': 0.0},
   197:         ]
   198:         num_decay_params = sum(p.numel() for p in decay_params)
   199:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   200:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   201:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   202:         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   203:         use_fused = fused_available and device_type == 'cuda'
   204:         extra_args = dict(fused=True) if use_fused else dict()
   205:         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
   206:         print(f"using fused AdamW: {use_fused}")
   207:         return optimizer
   208: 
   209: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   210: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):

Lines 266–266:
   263:     lr_decay_iters = max_iters
   264:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   265:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   266:     CONFIG_OVERRIDES = {}
   267: 
   268:     # Apply per-method hyperparameter overrides
   269:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `learned_scaling` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 88–99:
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

Lines 128–132:
   125:         ))
   126:         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
   127:         self.transformer.wte.weight = self.lm_head.weight
   128:         # ── Learnable residual scaling + x0 injection ──
   129:         # resid_lambdas[i]: scales the incoming residual stream (init 1.0 = vanilla)
   130:         # x0_lambdas[i]:    scales the embedding injection (init 0.0 = no injection)
   131:         self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
   132:         self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
   133:         self.apply(self._init_weights)
   134:         for pn, p in self.named_parameters():
   135:             if pn.endswith('c_proj.weight'):

Lines 164–170:
   161:         if use_pos:
   162:             pos = torch.arange(0, t, dtype=torch.long, device=device)
   163:             x = x + self.transformer.wpe(pos)
   164:         # ── Learnable residual scaling + x0 injection ──
   165:         # x0 = embedding output; provides gradient highway to every depth.
   166:         x0 = x
   167:         for i, block in enumerate(self.transformer.h):
   168:             block_out = block(x)
   169:             delta = block_out - x
   170:             x = self.resid_lambdas[i] * x + delta + self.x0_lambdas[i] * x0
   171:         x = self.transformer.ln_f(x)
   172:         if targets is not None:
   173:             logits = self.lm_head(x)

Lines 181–204:
   178:         return logits, loss
   179: 
   180:     # ── Optimizer Configuration ────────────────────────────────────────────
   181:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   182:         # Route residual scaling params to no-decay group
   183:         scaling_ids = {id(self.resid_lambdas), id(self.x0_lambdas)}
   184:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   185:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and id(p) not in scaling_ids]
   186:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and id(p) not in scaling_ids]
   187:         scaling_params = [p for n, p in param_dict.items() if id(p) in scaling_ids]
   188:         optim_groups = [
   189:             {'params': decay_params, 'weight_decay': weight_decay},
   190:             {'params': nodecay_params, 'weight_decay': 0.0},
   191:             {'params': scaling_params, 'weight_decay': 0.0},
   192:         ]
   193:         num_decay_params = sum(p.numel() for p in decay_params)
   194:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   195:         num_scaling_params = sum(p.numel() for p in scaling_params)
   196:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   197:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   198:         print(f"num scaling parameter tensors: {len(scaling_params)}, with {num_scaling_params:,} parameters")
   199:         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   200:         use_fused = fused_available and device_type == 'cuda'
   201:         extra_args = dict(fused=True) if use_fused else dict()
   202:         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
   203:         print(f"using fused AdamW: {use_fused}")
   204:         return optimizer
   205: 
   206: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   207: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):

Lines 263–263:
   260:     lr_decay_iters = max_iters
   261:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   262:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   263:     CONFIG_OVERRIDES = {}
   264: 
   265:     # Apply per-method hyperparameter overrides
   266:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `full_attnres` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 88–99:
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

Lines 128–134:
   125:         ))
   126:         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
   127:         self.transformer.wte.weight = self.lm_head.weight
   128:         # ── Block Attention Residuals: partition layers into blocks ──
   129:         # 24 layers / 4 = 6 blocks; attention at 5 boundaries + 1 output query
   130:         self.attnres_block_size = 4  # layers per block
   131:         n_blocks = config.n_layer // self.attnres_block_size
   132:         # n_blocks-1 boundary queries (first block gets embedding directly)
   133:         self.attnres_queries = nn.Parameter(torch.zeros(n_blocks - 1, config.n_embd))
   134:         self.attnres_query_out = nn.Parameter(torch.zeros(config.n_embd))
   135:         self.apply(self._init_weights)
   136:         for pn, p in self.named_parameters():
   137:             if pn.endswith('c_proj.weight'):

Lines 166–190:
   163:         if use_pos:
   164:             pos = torch.arange(0, t, dtype=torch.long, device=device)
   165:             x = x + self.transformer.wpe(pos)
   166:         # ── Block Attention Residuals: standard residual within blocks,
   167:         #    attention aggregation at block boundaries ──
   168:         block_size_layers = self.attnres_block_size
   169:         n_blocks = len(self.transformer.h) // block_size_layers
   170:         block_outputs = [x]  # initial embedding is first source
   171:         for blk_idx in range(n_blocks):
   172:             # At block boundary (except first): attend over previous block outputs
   173:             if blk_idx > 0:
   174:                 stacked = torch.stack(block_outputs, dim=0)  # (num_sources, B, T, D)
   175:                 keys_normed = F.rms_norm(stacked, (stacked.size(-1),))
   176:                 logits = torch.einsum('d, n b t d -> n b t', self.attnres_queries[blk_idx - 1], keys_normed)
   177:                 weights = logits.softmax(dim=0)  # (num_sources, B, T)
   178:                 x = torch.einsum('n b t, n b t d -> b t d', weights, stacked)
   179:             # Run layers within this block with standard residual connections
   180:             start = blk_idx * block_size_layers
   181:             end = start + block_size_layers
   182:             for layer_idx in range(start, end):
   183:                 x = self.transformer.h[layer_idx](x)
   184:             block_outputs.append(x)
   185:         # Final output: attend over all block outputs with dedicated query
   186:         stacked = torch.stack(block_outputs, dim=0)
   187:         keys_normed = F.rms_norm(stacked, (stacked.size(-1),))
   188:         logits = torch.einsum('d, n b t d -> n b t', self.attnres_query_out, keys_normed)
   189:         weights = logits.softmax(dim=0)
   190:         x = torch.einsum('n b t, n b t d -> b t d', weights, stacked)
   191:         x = self.transformer.ln_f(x)
   192:         if targets is not None:
   193:             logits = self.lm_head(x)

Lines 201–224:
   198:         return logits, loss
   199: 
   200:     # ── Optimizer Configuration ────────────────────────────────────────────
   201:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   202:         # Separate AttnRes query params from main model params
   203:         attnres_params = [self.attnres_queries, self.attnres_query_out]
   204:         attnres_ids = {id(p) for p in attnres_params}
   205:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   206:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and id(p) not in attnres_ids]
   207:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and id(p) not in attnres_ids]
   208:         optim_groups = [
   209:             {'params': decay_params, 'weight_decay': weight_decay},
   210:             {'params': nodecay_params, 'weight_decay': 0.0},
   211:             {'params': attnres_params, 'lr': learning_rate * 0.1, 'weight_decay': 0.0},
   212:         ]
   213:         num_decay_params = sum(p.numel() for p in decay_params)
   214:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   215:         num_attnres_params = sum(p.numel() for p in attnres_params)
   216:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   217:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   218:         print(f"num AttnRes parameter tensors: {len(attnres_params)}, with {num_attnres_params:,} parameters")
   219:         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   220:         use_fused = fused_available and device_type == 'cuda'
   221:         extra_args = dict(fused=True) if use_fused else dict()
   222:         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
   223:         print(f"using fused AdamW: {use_fused}")
   224:         return optimizer
   225: 
   226: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   227: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):

Lines 283–283:
   280:     lr_decay_iters = max_iters
   281:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   282:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   283:     CONFIG_OVERRIDES = {}
   284: 
   285:     # Apply per-method hyperparameter overrides
   286:     for _k, _v in CONFIG_OVERRIDES.items():
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
