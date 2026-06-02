# MLS-Bench: llm-pretrain-loss

# LLM Pretraining: Loss Function Optimization

## Research Question
Design an improved loss function for GPT-2 next-token language model pretraining. The change should reduce validation loss and improve downstream language ability under the same architecture, data, and optimization budget, compared to standard cross-entropy.

## Background
The default objective is plain next-token cross-entropy. Several modifications have been studied as drop-in replacements at this layer:

- **Label smoothing** — Szegedy et al., "Rethinking the Inception Architecture for Computer Vision", 2015, arXiv:1512.00567 (Section 7). Replaces hard one-hot targets with `(1-eps) * onehot + eps / V` (default eps ≈ 0.1).
- **Logit z-loss** — auxiliary penalty `lambda * (logsumexp(logits))^2` to keep logit magnitudes small. Originally from the Mesh-TensorFlow softmax z-loss; popularized by ST-MoE / PaLM (Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", 2022, arXiv:2202.08906). Typical coefficient `lambda ≈ 1e-4`.
- **Logit soft-capping** — `softcap * tanh(logits / softcap)` applied before the softmax, used in Gemma 2 (Gemma Team, "Gemma 2: Improving Open Language Models at a Practical Size", 2024, arXiv:2408.00118), with attention logits capped at 50.0 and final logits capped at 30.0.

## What you can modify
The `compute_loss` function in `nanoGPT/custom_pretrain.py`:
- Loss formulation (default: standard cross-entropy).
- Logit processing (e.g., soft-capping, temperature scaling).
- Regularization terms (e.g., z-loss, entropy penalties).
- Label-distribution modifications (e.g., label smoothing).

### Interface contract
- Signature must remain `compute_loss(logits, targets)`.
- `logits` shape `(B, T, V)`; `targets` shape `(B, T)`.
- The function is called inside the model's forward pass during training.
- Stable throughout training; do not lower reported loss by distorting probabilities (e.g., via temperature) without improving the actual modeling distribution.

## Reference baselines
- `label_smoothing` — eps=0.1.
- `z_loss` — lambda=1e-4.
- `softcap_ce` — Gemma-2-style final-logit soft-cap at 30.0.

## Fixed Pipeline
- **Model**: GPT-2 Medium (24 layers, 16 heads, d=1024, ~355M params).
- **Dataset**: FineWeb 10B (HuggingFace `HuggingFaceFW/fineweb` `sample-10BT`), GPT-2 tokenizer, ~7.1B training tokens.
- **Training**: 13,535 iterations, micro-batch 64, gradient accumulation 8, 2-GPU DDP.
- Architecture, tokenizer, dataset, training loop, and evaluation pipeline are fixed.

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
or deleting existing ones — will cause your submission to be invalid.

- `nanoGPT/custom_pretrain.py`
- editable lines **188–191**
- editable lines **247–249**


Other files you may **read** for context (do not modify):
- `nanoGPT/model.py`


## Readable Context


### `nanoGPT/custom_pretrain.py`  [EDITABLE — lines 188–191, lines 247–249 only]

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
   114: class GPT(nn.Module):
   115:     def __init__(self, config):
   116:         super().__init__()
   117:         self.config = config
   118:         self.transformer = nn.ModuleDict(dict(
   119:             wte=nn.Embedding(config.vocab_size, config.n_embd),
   120:             wpe=nn.Embedding(config.block_size, config.n_embd),
   121:             drop=nn.Dropout(config.dropout),
   122:             h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
   123:             ln_f=LayerNorm(config.n_embd, bias=config.bias),
   124:         ))
   125:         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
   126:         self.transformer.wte.weight = self.lm_head.weight
   127:         self.apply(self._init_weights)
   128:         for pn, p in self.named_parameters():
   129:             if pn.endswith('c_proj.weight'):
   130:                 torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
   131:         print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
   132: 
   133:     def get_num_params(self, non_embedding=True):
   134:         n_params = sum(p.numel() for p in self.parameters())
   135:         if non_embedding:
   136:             n_params -= self.transformer.wpe.weight.numel()
   137:         return n_params
   138: 
   139:     def _init_weights(self, module):
   140:         if isinstance(module, nn.Linear):
   141:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   142:             if module.bias is not None:
   143:                 torch.nn.init.zeros_(module.bias)
   144:         elif isinstance(module, nn.Embedding):
   145:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   146: 
   147:     def forward(self, idx, targets=None):
   148:         device = idx.device
   149:         b, t = idx.size()
   150:         assert t <= self.config.block_size
   151:         tok_emb = self.transformer.wte(idx)
   152:         x = self.transformer.drop(tok_emb)
   153:         use_pos = getattr(self.transformer.h[0].attn, 'use_pos_emb', True)
   154:         if use_pos:
   155:             pos = torch.arange(0, t, dtype=torch.long, device=device)
   156:             x = x + self.transformer.wpe(pos)
   157:         for block in self.transformer.h:
   158:             x = block(x)
   159:         x = self.transformer.ln_f(x)
   160:         if targets is not None:
   161:             logits = self.lm_head(x)
   162:             loss = compute_loss(logits, targets)
   163:         else:
   164:             logits = self.lm_head(x[:, [-1], :])
   165:             loss = None
   166:         return logits, loss
   167: 
   168:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   169:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   170:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   171:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   172:         optim_groups = [
   173:             {'params': decay_params, 'weight_decay': weight_decay},
   174:             {'params': nodecay_params, 'weight_decay': 0.0},
   175:         ]
   176:         num_decay_params = sum(p.numel() for p in decay_params)
   177:         num_nodecay_params = sum(p.numel() for p in nodecay_params)
   178:         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
   179:         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
   180:         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
   181:         use_fused = fused_available and device_type == 'cuda'
   182:         extra_args = dict(fused=True) if use_fused else dict()
   183:         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
   184:         print(f"using fused AdamW: {use_fused}")
   185:         return optimizer
   186: 
   187: # ── Loss Computation ───────────────────────────────────────────────────────
   188: def compute_loss(logits, targets):
   189:     """Compute language modeling loss from logits and targets."""
   190:     return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
   191: 
   192: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   193: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   194:     """Cosine learning rate schedule with linear warmup."""
   195:     if it < warmup_iters:
   196:         return learning_rate * (it + 1) / (warmup_iters + 1)
   197:     if it > lr_decay_iters:
   198:         return min_lr
   199:     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
   200:     assert 0 <= decay_ratio <= 1
   201:     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
   202:     return min_lr + coeff * (learning_rate - min_lr)
   203: 
   204: # ============================================================================
   205: # Data Loading
   206: # ============================================================================
   207: 
   208: def get_batch(data, batch_size, block_size, device):
   209:     """Get a random batch from a pre-opened memmap (nanoGPT style)."""
   210:     ix = torch.randint(len(data) - block_size, (batch_size,))
   211:     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
   212:     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
   213:     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
   214:     return x, y
   215: 
   216: # ============================================================================
   217: # Training Script
   218: # ============================================================================
   219: 
   220: if __name__ == '__main__':
   221:     # ── Configuration from environment ──
   222:     output_dir = os.environ.get('OUTPUT_DIR', 'out')
   223:     seed = int(os.environ.get('SEED', 1337))
   224:     data_dir = os.environ.get('DATA_DIR', '/data/climbmix')
   225: 
   226:     # Model config from environment
   227:     n_layer = int(os.environ.get('N_LAYER', 12))
   228:     n_head = int(os.environ.get('N_HEAD', 12))
   229:     n_embd = int(os.environ.get('N_EMBD', 768))
   230: 
   231:     # Training hyperparameters (overridable via env for different model sizes)
   232:     max_iters = int(os.environ.get('MAX_ITERS', 5000))
   233:     eval_interval = int(os.environ.get('EVAL_INTERVAL', 500))
   234:     eval_iters = 200
   235:     log_interval = 10
   236:     batch_size = int(os.environ.get('BATCH_SIZE', 12))
   237:     block_size = 1024
   238:     gradient_accumulation_steps = int(os.environ.get('GRAD_ACCUM', 5))
   239:     learning_rate = float(os.environ.get('LEARNING_RATE', 6e-4))
   240:     min_lr = learning_rate / 10
   241:     weight_decay = 1e-1
   242:     beta1 = 0.9
   243:     beta2 = 0.95
   244:     grad_clip = 1.0
   245:     warmup_iters = int(max_iters * 0.04)
   246:     lr_decay_iters = max_iters
   247:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   248:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   249:     CONFIG_OVERRIDES = {}
   250: 
   251:     # Apply per-method hyperparameter overrides
   252:     for _k, _v in CONFIG_OVERRIDES.items():
   253:         if _k == 'learning_rate': learning_rate = _v; min_lr = learning_rate / 10
   254:         elif _k == 'weight_decay': weight_decay = _v
   255:         elif _k == 'warmup_iters': warmup_iters = _v
   256:         elif _k == 'min_lr': min_lr = _v
   257:         elif _k == 'grad_clip': grad_clip = _v
   258: 
   259:     compile_model = True
   260:     dtype = 'bfloat16'
   261: 
   262:     # ── DDP Setup ──
   263:     ddp = int(os.environ.get('RANK', -1)) != -1
   264:     if ddp:
   265:         import torch.distributed as dist
   266:         from torch.nn.parallel import DistributedDataParallel as DDP
   267:         dist.init_process_group(backend='nccl')
   268:         ddp_rank = int(os.environ['RANK'])
   269:         ddp_local_rank = int(os.environ['LOCAL_RANK'])
   270:         ddp_world_size = int(os.environ['WORLD_SIZE'])
   271:         device = f'cuda:{ddp_local_rank}'
   272:         torch.cuda.set_device(device)
   273:         master_process = ddp_rank == 0
   274:         seed_offset = ddp_rank
   275:         assert gradient_accumulation_steps % ddp_world_size == 0
   276:         gradient_accumulation_steps //= ddp_world_size
   277:     else:
   278:         master_process = True
   279:         device = 'cuda'
   280:         seed_offset = 0
   281: 
   282:     # ── Setup ──
   283:     device_type = 'cuda'
   284:     torch.manual_seed(seed + seed_offset)
   285:     torch.backends.cuda.matmul.allow_tf32 = True
   286:     torch.backends.cudnn.allow_tf32 = True
   287:     ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
   288:     ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
   289:     if master_process:
   290:         os.makedirs(output_dir, exist_ok=True)
   291: 
   292:     tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
   293:     if ddp:
   294:         tokens_per_iter *= int(os.environ.get('WORLD_SIZE', 1))
   295:     if master_process:
   296:         print(f"tokens per iteration will be: {tokens_per_iter:,}")
   297: 
   298:     # ── Load Data ──
   299:     train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
   300:     val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
   301:     if master_process:
   302:         print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
   303: 
   304:     # ── Model Init ──
   305:     model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
   306:                       block_size=block_size, bias=False, vocab_size=50304, dropout=0.0)
   307:     gptconf = GPTConfig(**model_args)
   308:     model = GPT(gptconf)
   309:     model.to(device)
   310: 
   311: 
   312:     scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
   313:     optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
   314: 
   315:     if compile_model:
   316:         if master_process:
   317:             print("compiling the model...")
   318:         model = torch.compile(model)
   319: 
   320:     if ddp:
   321:         model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
   322: 
   323:     # ── Evaluation ──
   324:     @torch.no_grad()
   325:     def estimate_loss():
   326:         out = {}
   327:         raw = model.module if ddp else model
   328:         raw.eval()
   329:         for split, data in [('train', train_data), ('val', val_data)]:
   330:             losses = torch.zeros(eval_iters)
   331:             for k in range(eval_iters):
   332:                 X, Y = get_batch(data, batch_size, block_size, device)
   333:                 with ctx:
   334:                     logits, loss = raw(X, Y)
   335:                 losses[k] = loss.item()
   336:             out[split] = losses.mean()
   337:         raw.train()
   338:         return out
   339: 
   340:     # ── Training Loop ──
   341:     t0 = time.time()
   342:     best_val_loss = 1e9
   343: 
   344:     for iter_num in range(max_iters + 1):
   345:         lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
   346:         for param_group in optimizer.param_groups:
   347:             param_group['lr'] = lr
   348: 
   349:         if iter_num % eval_interval == 0 and master_process:
   350:             losses = estimate_loss()
   351:             train_loss = losses['train'].item()
   352:             val_loss = losses['val'].item()
   353:             print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
   354:             print(f"TRAIN_METRICS: step={iter_num}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
   355:             if val_loss < best_val_loss:
   356:                 best_val_loss = val_loss
   357: 
   358:         for micro_step in range(gradient_accumulation_steps):
   359:             if ddp:
   360:                 model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
   361:             with ctx:
   362:                 X, Y = get_batch(train_data, batch_size, block_size, device)
   363:                 logits, loss = model(X, Y)
   364:                 loss = loss / gradient_accumulation_steps
   365:             scaler.scale(loss).backward()
   366: 
   367:         if grad_clip != 0.0:
   368:             scaler.unscale_(optimizer)
   369:             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
   370:         scaler.step(optimizer)
   371:         scaler.update()
   372:         optimizer.zero_grad(set_to_none=True)
   373: 
   374:         t1 = time.time()
   375:         dt = t1 - t0
   376:         t0 = t1
   377:         if iter_num % log_interval == 0 and iter_num > 0 and master_process:
   378:             lossf = loss.item() * gradient_accumulation_steps
   379:             print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}")
   380: 
   381:     # ── Free training state to reclaim GPU memory ──
   382:     del optimizer, scaler
   383:     import gc; gc.collect()
   384:     torch.cuda.empty_cache()
   385: 
   386:     # ── Final Evaluation ──
   387:     if master_process:
   388:         losses = estimate_loss()
   389:         val_loss = losses['val'].item()
   390:         train_loss = losses['train'].item()
   391:         print(f"Final: train loss {train_loss:.4f}, val loss {val_loss:.4f}, best val loss {best_val_loss:.4f}")
   392: 
   393:         # ── PPL on benchmark datasets ──
   394:         eval_dir = os.environ.get('EVAL_DIR', '/data/eval')
   395:         raw = model.module if ddp else model
   396:         raw.eval()
   397:         eval_datasets = ['wikitext2', 'lambada']
   398:         ppl_results = {}
   399:         for ds_name in eval_datasets:
   400:             ds_path = os.path.join(eval_dir, f'{ds_name}.bin')
   401:             if not os.path.exists(ds_path):
   402:                 print(f"Eval dataset not found: {ds_path}")
   403:                 continue
   404:             data = np.memmap(ds_path, dtype=np.uint16, mode='r')
   405:             n_tokens = len(data)
   406:             # Process in non-overlapping chunks of block_size
   407:             total_loss = 0.0
   408:             n_chunks = 0
   409:             with torch.no_grad():
   410:                 for start in range(0, n_tokens - block_size, block_size):
   411:                     x = torch.from_numpy(data[start:start+block_size].astype(np.int64)).unsqueeze(0).to(device)
   412:                     y = torch.from_numpy(data[start+1:start+1+block_size].astype(np.int64)).unsqueeze(0).to(device)
   413:                     with ctx:
   414:                         _, loss = raw(x, y)
   415:                     total_loss += loss.item()
   416:                     n_chunks += 1
   417:             avg_loss = total_loss / n_chunks
   418:             ppl = math.exp(avg_loss)
   419:             ppl_results[ds_name] = ppl
   420:             print(f"PPL {ds_name}: {ppl:.2f} (avg_loss={avg_loss:.4f}, {n_chunks} chunks)")
   421: 
   422:         ppl_str = ', '.join(f'{k}_ppl={v:.2f}' for k, v in ppl_results.items())
   423:         print(f"TEST_METRICS: val_loss={val_loss:.4f}, {ppl_str}", flush=True)
   424: 
   425:         # ── Save checkpoint for downstream evaluation (lm-eval-harness) ──
   426:         import shutil
   427:         env_label = os.environ.get('ENV', 'model')
   428:         # Unwrap torch.compile to get clean state_dict keys
   429:         save_model = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
   430:         ckpt_data = {'model_state_dict': save_model.state_dict(), 'model_args': model_args}
   431:         ckpt_path = os.path.join(output_dir, f'ckpt_{env_label}.pt')
   432:         torch.save(ckpt_data, ckpt_path)
   433:         print(f"Checkpoint saved to {ckpt_path}")
   434:         src_path = os.path.join(output_dir, f'model_source_{env_label}.py')
   435:         shutil.copy2(os.path.abspath(__file__), src_path)
   436:         print(f"Model source saved to {src_path}")
   437: 
   438:     if ddp:
   439:         dist.destroy_process_group()
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


### `label_smoothing` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 188–199:
   185:         return optimizer
   186: 
   187: # ── Loss Computation ───────────────────────────────────────────────────────
   188: def compute_loss(logits, targets):
   189:     """Cross-entropy with label smoothing (eps=0.05) during training only.
   190: 
   191:     Label smoothing is applied only when gradients are enabled (training).
   192:     During evaluation (@torch.no_grad()), standard cross-entropy is used
   193:     so that val_loss remains comparable across methods.
   194:     """
   195:     smoothing = 0.05 if torch.is_grad_enabled() else 0.0
   196:     return F.cross_entropy(
   197:         logits.view(-1, logits.size(-1)), targets.view(-1),
   198:         ignore_index=-1, label_smoothing=smoothing
   199:     )
   200: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   201: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   202:     """Cosine learning rate schedule with linear warmup."""

Lines 255–257:
   252:     grad_clip = 1.0
   253:     warmup_iters = int(max_iters * 0.04)
   254:     lr_decay_iters = max_iters
   255:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   256:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   257:     CONFIG_OVERRIDES = {}
   258: 
   259:     # Apply per-method hyperparameter overrides
   260:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `softcap_ce` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 188–198:
   185:         return optimizer
   186: 
   187: # ── Loss Computation ───────────────────────────────────────────────────────
   188: def compute_loss(logits, targets):
   189:     """Cross-entropy with logit softcapping via sigmoid."""
   190:     # Softcap: maps logits through A * sigmoid((logits + B) / C)
   191:     # Prevents extreme logit magnitudes while preserving ranking
   192:     # Constants from modded-nanogpt PR #199
   193:     A, B, C = 23.0, 5.0, 7.5
   194:     capped_logits = A * torch.sigmoid((logits.float() + B) / C)
   195:     return F.cross_entropy(
   196:         capped_logits.view(-1, capped_logits.size(-1)), targets.view(-1),
   197:         ignore_index=-1
   198:     )
   199: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   200: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   201:     """Cosine learning rate schedule with linear warmup."""

Lines 254–256:
   251:     grad_clip = 1.0
   252:     warmup_iters = int(max_iters * 0.04)
   253:     lr_decay_iters = max_iters
   254:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   255:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   256:     CONFIG_OVERRIDES = {}
   257: 
   258:     # Apply per-method hyperparameter overrides
   259:     for _k, _v in CONFIG_OVERRIDES.items():
```

### `z_loss` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 188–198:
   185:         return optimizer
   186: 
   187: # ── Loss Computation ───────────────────────────────────────────────────────
   188: def compute_loss(logits, targets):
   189:     """Cross-entropy with z-loss regularization."""
   190:     flat_logits = logits.view(-1, logits.size(-1))
   191:     flat_targets = targets.view(-1)
   192:     ce_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)
   193:     # Z-loss: penalize large log-partition values
   194:     log_z = torch.logsumexp(flat_logits, dim=-1)
   195:     # Only compute z-loss for non-ignored positions
   196:     mask = flat_targets != -1
   197:     z_loss = (log_z[mask] ** 2).mean()
   198:     return ce_loss + 1e-4 * z_loss
   199: # ── Learning Rate Schedule ─────────────────────────────────────────────────
   200: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   201:     """Cosine learning rate schedule with linear warmup."""

Lines 254–256:
   251:     grad_clip = 1.0
   252:     warmup_iters = int(max_iters * 0.04)
   253:     lr_decay_iters = max_iters
   254:     # CONFIG_OVERRIDES: override training hyperparameters for your method.
   255:     # Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
   256:     CONFIG_OVERRIDES = {}
   257: 
   258:     # Apply per-method hyperparameter overrides
   259:     for _k, _v in CONFIG_OVERRIDES.items():
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
