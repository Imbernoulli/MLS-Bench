# MLS-Bench: llm-kv-structural-reduction

# LLM Pretraining: KV-Structural Reduction

## Research Question

Design a more KV-efficient causal attention structure for GPT-style
pretraining, with the primary focus on the tradeoff between KV head
sharing and latent KV compression:

- how much language-model quality can be preserved by reducing the
  realized KV state
- whether grouped/shared KV heads or latent KV bottlenecks give the better
  quality-memory tradeoff under a fixed small-scale pretraining budget

## Background

Multi-Head Attention (MHA) materializes one (K, V) pair per query head,
which dominates KV memory at long context. Multi-Query Attention (MQA) and
Grouped-Query Attention (GQA) reduce that by sharing a small number of K/V
heads across many query heads. Multi-head Latent Attention (MLA), proposed
in DeepSeek-V2 (Liu et al., 2024; arXiv:2405.04434) and analyzed further
in TransMLA (Meng et al., 2025; arXiv:2502.07864), instead compresses K/V
into a low-rank latent vector that is decompressed on the fly, decoupling
realized KV bytes from query-head count. This task isolates that design
space inside one fixed nanoGPT-style pretraining loop.

## What You Can Modify

One editable region in `custom_pretrain.py`:

1. Attention-structure region (between read-only
   `# BEGIN/END KV EDITABLE REGION` markers — do NOT delete or replace the
   marker lines), including:
   - `build_kv_heads(...)`: how many KV heads are materialized relative to
     query heads
   - `cross_layer_share(...)`: optional structural sharing hook inside the
     attention stack
   - `latent_kv_project(...)`: whether K/V are compressed into a
     lower-rank latent space
   - `CausalSelfAttention`: how the above choices are instantiated inside
     the attention block, including the internal query/KV projection and
     attention mixing path

## Intended Task Boundary

- This task studies KV-state reduction inside the attention block.
- The main comparison axes are dense MHA vs grouped/shared KV heads, and
  grouped/shared KV heads vs latent KV compression.
- `cross_layer_share(...)` remains available as an auxiliary structural
  hook inside the same block.
- The evaluator enforces the top-level boundary of this region with an AST
  validator: only the allowed helper functions plus `CausalSelfAttention`
  may appear in the editable span. That keeps edits inside the attention
  block, even though the internal contents of `CausalSelfAttention` remain
  flexible.

## Baselines

The visible baseline chain is `MHA -> MQA -> GQA -> MLA`:

- `MHA`: dense unreduced control with one KV head per query head.
- `MQA`: simplest structural anchor with one shared KV head reused across
  all query heads.
- `GQA`: keeps full query heads but reduces the number of materialized KV
  heads.
- `MLA`: latent-KV bottleneck adapted from the DeepSeek-V2
  (arXiv:2405.04434) / TransMLA (arXiv:2502.07864) family into the fixed
  nanoGPT substrate. A proper MLA implementation has
  `kv_lora_rank < head_dim`, reducing the realized KV state.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/nanoGPT/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `nanoGPT/custom_pretrain.py`
- editable lines **36–155**




## Readable Context


### `nanoGPT/custom_pretrain.py`  [EDITABLE — lines 36–155 only]

```python
     1: """Custom GPT-2 pretraining script for KV-structural reduction tasks.
     2: 
     3: Based on Andrej Karpathy's nanoGPT, with a narrow editable region for KV
     4: structure changes such as grouped KV heads and latent KV compression.
     5: """
     6: 
     7: import ast
     8: import inspect
     9: import json
    10: import math
    11: import os
    12: import time
    13: from contextlib import nullcontext
    14: from dataclasses import dataclass
    15: 
    16: import numpy as np
    17: import torch
    18: import torch.nn as nn
    19: from torch.nn import functional as F
    20: 
    21: 
    22: class LayerNorm(nn.Module):
    23:     """LayerNorm but with an optional bias."""
    24: 
    25:     def __init__(self, ndim, bias):
    26:         super().__init__()
    27:         self.weight = nn.Parameter(torch.ones(ndim))
    28:         self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    29: 
    30:     def forward(self, input):
    31:         return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    32: 
    33: 
    34: # ── Editable region: KV structure design ──────────────────────────────────
    35: # BEGIN KV EDITABLE REGION
    36: def build_kv_heads(config):
    37:     """Return the number of KV heads and per-head dimension."""
    38: 
    39:     n_kv_head = config.n_head
    40:     head_dim = config.n_embd // config.n_head
    41:     return n_kv_head, head_dim
    42: 
    43: 
    44: def cross_layer_share(layer_idx, config):
    45:     """Optionally reuse KV structure across layers.
    46: 
    47:     Default: no cross-layer KV sharing.
    48:     """
    49: 
    50:     return False
    51: 
    52: 
    53: def latent_kv_project(k, v, config):
    54:     """Optional latent KV bottleneck.
    55: 
    56:     Default: identity projection.
    57:     """
    58: 
    59:     return k, v, 1.0
    60: 
    61: 
    62: def expand_kv_to_q_heads(tensor, target_heads):
    63:     """Expand KV heads to query heads while remaining safe for any head count."""
    64: 
    65:     current_heads = tensor.size(1)
    66:     if current_heads == target_heads:
    67:         return tensor
    68:     full_repeats = target_heads // current_heads
    69:     remainder = target_heads % current_heads
    70:     parts = []
    71:     if full_repeats > 0:
    72:         parts.append(tensor.repeat_interleave(full_repeats, dim=1))
    73:     if remainder > 0:
    74:         parts.append(tensor[:, :remainder, :, :])
    75:     return torch.cat(parts, dim=1)
    76: 
    77: 
    78: class CausalSelfAttention(nn.Module):
    79:     _shared_kv_cache = {}
    80: 
    81:     def __init__(self, config, layer_idx=0):
    82:         super().__init__()
    83:         assert config.n_embd % config.n_head == 0
    84:         self.n_head = config.n_head
    85:         self.n_embd = config.n_embd
    86:         self.dropout = config.dropout
    87:         self.layer_idx = layer_idx
    88:         self.n_kv_head, self.head_dim = build_kv_heads(config)
    89:         self.share_across_layers = cross_layer_share(layer_idx, config)
    90: 
    91:         q_dim = config.n_embd
    92:         kv_dim = 2 * self.n_kv_head * self.head_dim
    93:         self.c_attn = nn.Linear(config.n_embd, q_dim + kv_dim, bias=config.bias)
    94:         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    95:         self.attn_dropout = nn.Dropout(config.dropout)
    96:         self.resid_dropout = nn.Dropout(config.dropout)
    97:         self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    98:         if not self.flash:
    99:             self.register_buffer(
   100:                 "bias",
   101:                 torch.tril(torch.ones(config.block_size, config.block_size)).view(
   102:                     1, 1, config.block_size, config.block_size
   103:                 ),
   104:             )
   105:         self.use_pos_emb = True
   106:         self.head_sharing_ratio = self.n_head / max(self.n_kv_head, 1)
   107: 
   108:     def forward(self, x):
   109:         bsz, seq_len, channels = x.size()
   110:         qkv = self.c_attn(x)
   111:         q, kv = qkv.split(
   112:             [self.n_embd, 2 * self.n_kv_head * self.head_dim],
   113:             dim=2,
   114:         )
   115:         k, v = kv.chunk(2, dim=2)
   116: 
   117:         q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
   118:         k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
   119:         v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
   120: 
   121:         reused_previous = False
   122:         if self.share_across_layers and (self.layer_idx - 1) in self._shared_kv_cache:
   123:             k, v = self._shared_kv_cache[self.layer_idx - 1]
   124:             reused_previous = True
   125:         else:
   126:             self._shared_kv_cache[self.layer_idx] = (k.detach(), v.detach())
   127: 
   128:         if self.n_kv_head != self.n_head:
   129:             k = expand_kv_to_q_heads(k, self.n_head)
   130:             v = expand_kv_to_q_heads(v, self.n_head)
   131: 
   132:         k, v, latent_ratio = latent_kv_project(k, v, self)
   133:         self._last_latent_rank_ratio = float(latent_ratio)
   134:         self._last_kv_storage_ratio = 0.0 if reused_previous else float(latent_ratio)
   135:         self._uses_latent_compression = bool(latent_ratio < 0.999)
   136: 
   137:         if self.flash:
   138:             y = torch.nn.functional.scaled_dot_product_attention(
   139:                 q,
   140:                 k,
   141:                 v,
   142:                 attn_mask=None,
   143:                 dropout_p=self.dropout if self.training else 0.0,
   144:                 is_causal=True,
   145:             )
   146:         else:
   147:             att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
   148:             att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
   149:             att = F.softmax(att, dim=-1)
   150:             att = self.attn_dropout(att)
   151:             y = att @ v
   152: 
   153:         y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
   154:         y = self.resid_dropout(self.c_proj(y))
   155:         return y
   156: # END KV EDITABLE REGION
   157: 
   158: 
   159: def _validate_kv_editable_region():
   160:     allowed_names = {
   161:         "build_kv_heads",
   162:         "cross_layer_share",
   163:         "latent_kv_project",
   164:         "expand_kv_to_q_heads",
   165:         "MLARMSNorm",
   166:         "rotate_half",
   167:         "build_rotary_cache",
   168:         "apply_rotary_pos_emb_interleave",
   169:         "CausalSelfAttention",
   170:     }
   171:     required_names = {
   172:         "build_kv_heads",
   173:         "cross_layer_share",
   174:         "latent_kv_project",
   175:         "CausalSelfAttention",
   176:     }
   177:     with open(__file__, 'r') as _f:
   178:         source = _f.read()
   179:     start_marker = "# BEGIN KV EDITABLE REGION"
   180:     end_marker = "# END KV EDITABLE REGION"
   181:     start = source.index(start_marker) + len(start_marker)
   182:     end = source.index(end_marker)
   183:     snippet = source[start:end]
   184:     parsed = ast.parse(snippet)
   185:     seen_names = set()
   186:     for node in parsed.body:
   187:         if isinstance(node, ast.FunctionDef):
   188:             seen_names.add(node.name)
   189:             if node.name not in allowed_names:
   190:                 raise RuntimeError(f"Forbidden top-level function in KV region: {node.name}")
   191:         elif isinstance(node, ast.ClassDef):
   192:             seen_names.add(node.name)
   193:             if node.name not in allowed_names:
   194:                 raise RuntimeError(f"Forbidden top-level class in KV region: {node.name}")
   195:         else:
   196:             raise RuntimeError(
   197:                 "KV editable region may only contain top-level function/class definitions"
   198:             )
   199:     missing = required_names - seen_names
   200:     if missing:
   201:         raise RuntimeError(
   202:             f"KV editable region is missing required definitions: {sorted(missing)}"
   203:         )
   204: 
   205: 
   206: _validate_kv_editable_region()
   207: 
   208: 
   209: class MLP(nn.Module):
   210:     def __init__(self, config):
   211:         super().__init__()
   212:         self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
   213:         self.gelu = nn.GELU()
   214:         self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
   215:         self.dropout = nn.Dropout(config.dropout)
   216: 
   217:     def forward(self, x):
   218:         x = self.c_fc(x)
   219:         x = self.gelu(x)
   220:         x = self.c_proj(x)
   221:         x = self.dropout(x)
   222:         return x
   223: 
   224: 
   225: class Block(nn.Module):
   226:     def __init__(self, config, layer_idx):
   227:         super().__init__()
   228:         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
   229:         self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
   230:         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
   231:         self.mlp = MLP(config)
   232: 
   233:     def forward(self, x):
   234:         x = x + self.attn(self.ln_1(x))
   235:         x = x + self.mlp(self.ln_2(x))
   236:         return x
   237: 
   238: 
   239: @dataclass
   240: class GPTConfig:
   241:     block_size: int = 1024
   242:     vocab_size: int = 50304
   243:     n_layer: int = 12
   244:     n_head: int = 12
   245:     n_embd: int = 768
   246:     dropout: float = 0.0
   247:     bias: bool = False
   248: 
   249: 
   250: class GPT(nn.Module):
   251:     def __init__(self, config):
   252:         super().__init__()
   253:         self.config = config
   254:         self.transformer = nn.ModuleDict(
   255:             dict(
   256:                 wte=nn.Embedding(config.vocab_size, config.n_embd),
   257:                 wpe=nn.Embedding(config.block_size, config.n_embd),
   258:                 drop=nn.Dropout(config.dropout),
   259:                 h=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
   260:                 ln_f=LayerNorm(config.n_embd, bias=config.bias),
   261:             )
   262:         )
   263:         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
   264:         self.transformer.wte.weight = self.lm_head.weight
   265:         self.apply(self._init_weights)
   266:         for pn, p in self.named_parameters():
   267:             if pn.endswith("c_proj.weight"):
   268:                 torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
   269:         print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
   270: 
   271:     def get_num_params(self, non_embedding=True):
   272:         n_params = sum(p.numel() for p in self.parameters())
   273:         if non_embedding:
   274:             n_params -= self.transformer.wpe.weight.numel()
   275:         return n_params
   276: 
   277:     def _init_weights(self, module):
   278:         if isinstance(module, nn.Linear):
   279:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   280:             if module.bias is not None:
   281:                 torch.nn.init.zeros_(module.bias)
   282:         elif isinstance(module, nn.Embedding):
   283:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   284: 
   285:     def structural_metrics(self):
   286:         head_sharing = []
   287:         latent_rank = []
   288:         kv_bytes = []
   289:         for block in self.transformer.h:
   290:             attn = block.attn
   291:             n_head = int(getattr(attn, "n_head", self.config.n_head))
   292:             n_kv_head = int(getattr(attn, "n_kv_head", n_head))
   293:             head_dim = int(getattr(attn, "head_dim", self.config.n_embd // self.config.n_head))
   294:             head_sharing.append(n_head / max(n_kv_head, 1))
   295: 
   296:             if getattr(attn, "share_across_layers", False):
   297:                 # This layer borrows KV from the previous layer; no new KV storage needed.
   298:                 latent_rank.append(0.0)
   299:                 kv_bytes.append(0.0)
   300:             elif hasattr(attn, "kv_a_proj_with_mqa") and hasattr(attn, "kv_b_proj"):
   301:                 if hasattr(attn, "kv_a_layernorm"):
   302:                     kv_lora_rank = int(attn.kv_a_layernorm.weight.numel())
   303:                 else:
   304:                     kv_lora_rank = int(getattr(attn, "kv_lora_rank", head_dim))
   305:                 qk_rope_head_dim = int(attn.kv_a_proj_with_mqa.out_features - kv_lora_rank)
   306:                 qk_head_dim = int(getattr(attn, "qk_head_dim", head_dim + qk_rope_head_dim))
   307:                 latent_rank.append(kv_lora_rank / max(qk_head_dim, 1))
   308:                 kv_bytes.append(float(2 * (kv_lora_rank + qk_rope_head_dim)))
   309:             else:
   310:                 latent_rank.append(1.0)
   311:                 kv_bytes.append(float(2 * n_kv_head * head_dim * 2))
   312:         return {
   313:             "head_sharing_ratio": sum(head_sharing) / len(head_sharing),
   314:             "latent_rank_ratio": sum(latent_rank) / len(latent_rank),
   315:             "kv_bytes_per_token": sum(kv_bytes) / len(kv_bytes),
   316:         }
   317: 
   318:     def forward(self, idx, targets=None):
   319:         device = idx.device
   320:         _, t = idx.size()
   321:         assert t <= self.config.block_size
   322:         tok_emb = self.transformer.wte(idx)
   323:         x = self.transformer.drop(tok_emb)
   324:         use_pos = getattr(self.transformer.h[0].attn, "use_pos_emb", True)
   325:         if use_pos:
   326:             pos = torch.arange(0, t, dtype=torch.long, device=device)
   327:             x = x + self.transformer.wpe(pos)
   328:         for block in self.transformer.h:
   329:             x = block(x)
   330:         x = self.transformer.ln_f(x)
   331:         if targets is not None:
   332:             logits = self.lm_head(x)
   333:             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
   334:         else:
   335:             logits = self.lm_head(x[:, [-1], :])
   336:             loss = None
   337:         return logits, loss
   338: 
   339:     @torch.no_grad()
   340:     def generate(self, idx, max_new_tokens):
   341:         for _ in range(max_new_tokens):
   342:             idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
   343:             logits, _ = self(idx_cond)
   344:             next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
   345:             idx = torch.cat((idx, next_token), dim=1)
   346:         return idx
   347: 
   348:     def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
   349:         param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
   350:         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
   351:         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
   352:         optim_groups = [
   353:             {"params": decay_params, "weight_decay": weight_decay},
   354:             {"params": nodecay_params, "weight_decay": 0.0},
   355:         ]
   356:         fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
   357:         use_fused = fused_available and device_type == "cuda"
   358:         extra_args = dict(fused=True) if use_fused else dict()
   359:         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
   360:         print(f"using fused AdamW: {use_fused}")
   361:         return optimizer
   362: 
   363: 
   364: def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
   365:     if it < warmup_iters:
   366:         return learning_rate * (it + 1) / (warmup_iters + 1)
   367:     if it > lr_decay_iters:
   368:         return min_lr
   369:     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
   370:     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
   371:     return min_lr + coeff * (learning_rate - min_lr)
   372: 
   373: 
   374: def get_batch(split, data_dir, batch_size, block_size, device):
   375:     data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
   376:     ix = torch.randint(len(data) - block_size, (batch_size,))
   377:     x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
   378:     y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
   379:     if "cuda" in str(device):
   380:         x = x.pin_memory().to(device, non_blocking=True)
   381:         y = y.pin_memory().to(device, non_blocking=True)
   382:     else:
   383:         x = x.to(device)
   384:         y = y.to(device)
   385:     return x, y
   386: 
   387: 
   388: def get_named_eval_batch(dataset_path, batch_size, block_size, device):
   389:     data = np.memmap(dataset_path, dtype=np.uint16, mode="r")
   390:     ix = torch.randint(len(data) - block_size - 1, (batch_size,))
   391:     x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
   392:     y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
   393:     if "cuda" in str(device):
   394:         x = x.pin_memory().to(device, non_blocking=True)
   395:         y = y.pin_memory().to(device, non_blocking=True)
   396:     else:
   397:         x = x.to(device)
   398:         y = y.to(device)
   399:     return x, y
   400: 
   401: 
   402: if __name__ == "__main__":
   403:     output_dir = os.environ.get("OUTPUT_DIR", "out")
   404:     seed = int(os.environ.get("SEED", 1337))
   405:     _DATA_ROOT = os.environ.get("DATA_ROOT", "/data")
   406:     data_dir = os.environ.get("DATA_DIR", os.path.join(_DATA_ROOT, "climbmix"))
   407:     eval_dir = os.environ.get("EVAL_DIR", os.path.join(_DATA_ROOT, "eval"))
   408:     n_layer = int(os.environ.get("N_LAYER", 12))
   409:     n_head = int(os.environ.get("N_HEAD", 12))
   410:     n_embd = int(os.environ.get("N_EMBD", 768))
   411:     max_iters = int(os.environ.get("MAX_ITERS", 5000))
   412:     eval_interval = int(os.environ.get("EVAL_INTERVAL", 500))
   413:     eval_iters = 200
   414:     log_interval = 10
   415:     batch_size = int(os.environ.get("BATCH_SIZE", 12))
   416:     block_size = int(os.environ.get("BLOCK_SIZE", 1024))
   417:     gradient_accumulation_steps = int(os.environ.get("GRAD_ACCUM", 5))
   418:     run_aux_eval = os.environ.get("RUN_AUX_EVAL", "0") == "1"
   419:     aux_eval_datasets = [
   420:         item.strip() for item in os.environ.get("AUX_EVAL_DATASETS", "wikitext2").split(",") if item.strip()
   421:     ]
   422:     aux_eval_iters = int(os.environ.get("AUX_EVAL_ITERS", 64))
   423:     aux_eval_batch_size = int(os.environ.get("AUX_EVAL_BATCH_SIZE", 4))
   424:     learning_rate = float(os.environ.get("LEARNING_RATE", 6e-4))
   425:     min_lr = learning_rate / 10
   426:     weight_decay = 1e-1
   427:     beta1 = 0.9
   428:     beta2 = 0.95
   429:     grad_clip = 1.0
   430:     warmup_iters = int(max_iters * 0.04)
   431:     lr_decay_iters = max_iters
   432:     # torch.compile on MLA-style dynamic split/reshape (qk_nope+qk_rope,
   433:     # kv_a_proj split, broadcast-expand across heads) generated more graph
   434:     # breaks than speedup (6s/iter compiled vs 3.4s eager on H200). Keep
   435:     # compile off here; sibling llm-pretrain-attention etc. are free to flip.
   436:     compile_model = False
   437:     dtype = "bfloat16"
   438: 
   439:     ddp = int(os.environ.get("RANK", -1)) != -1
   440:     if ddp:
   441:         import torch.distributed as dist
   442:         from torch.nn.parallel import DistributedDataParallel as DDP
   443: 
   444:         dist.init_process_group(backend="nccl")
   445:         ddp_rank = int(os.environ["RANK"])
   446:         ddp_local_rank = int(os.environ["LOCAL_RANK"])
   447:         ddp_world_size = int(os.environ["WORLD_SIZE"])
   448:         device = f"cuda:{ddp_local_rank}"
   449:         torch.cuda.set_device(device)
   450:         master_process = ddp_rank == 0
   451:         seed_offset = ddp_rank
   452:     else:
   453:         master_process = True
   454:         seed_offset = 0
   455:         ddp_world_size = 1
   456:         device = "cuda" if torch.cuda.is_available() else "cpu"
   457: 
   458:     assert gradient_accumulation_steps % ddp_world_size == 0
   459:     gradient_accumulation_steps //= ddp_world_size
   460: 
   461:     os.makedirs(output_dir, exist_ok=True)
   462:     torch.manual_seed(seed + seed_offset)
   463:     torch.backends.cuda.matmul.allow_tf32 = True
   464:     torch.backends.cudnn.allow_tf32 = True
   465:     device_type = "cuda" if "cuda" in device else "cpu"
   466:     ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
   467:     ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
   468: 
   469:     tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
   470:     if master_process:
   471:         print(f"tokens per iteration will be: {tokens_per_iter:,}")
   472: 
   473:     model_args = dict(
   474:         n_layer=n_layer,
   475:         n_head=n_head,
   476:         n_embd=n_embd,
   477:         block_size=block_size,
   478:         bias=False,
   479:         vocab_size=50304,
   480:         dropout=0.0,
   481:     )
   482:     gptconf = GPTConfig(**model_args)
   483:     model = GPT(gptconf)
   484:     model.to(device)
   485:     scaler = torch.cuda.amp.GradScaler(enabled=False)
   486:     optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
   487:     if ddp:
   488:         model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
   489:     raw_model = model.module if ddp else model
   490: 
   491:     @torch.no_grad()
   492:     def estimate_loss():
   493:         out = {}
   494:         model.eval()
   495:         for split in ["train", "val"]:
   496:             losses = torch.zeros(eval_iters)
   497:             for k in range(eval_iters):
   498:                 x, y = get_batch(split, data_dir, batch_size, block_size, device)
   499:                 with ctx:
   500:                     _, loss = model(x, y)

[truncated: showing at most 500 lines / 60000 bytes from nanoGPT/custom_pretrain.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `mha` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 36–111:
    33: 
    34: # ── Editable region: KV structure design ──────────────────────────────────
    35: # BEGIN KV EDITABLE REGION
    36: def build_kv_heads(config):
    37:     """Dense control: one KV head per query head."""
    38: 
    39:     n_kv_head = config.n_head
    40:     head_dim = config.n_embd // config.n_head
    41:     return n_kv_head, head_dim
    42: 
    43: 
    44: def cross_layer_share(layer_idx, config):
    45:     return False
    46: 
    47: 
    48: def latent_kv_project(k, v, config):
    49:     return k, v, 1.0
    50: 
    51: 
    52: def expand_kv_to_q_heads(tensor, target_heads):
    53:     current_heads = tensor.size(1)
    54:     if current_heads == target_heads:
    55:         return tensor
    56:     full_repeats = target_heads // current_heads
    57:     remainder = target_heads % current_heads
    58:     parts = []
    59:     if full_repeats > 0:
    60:         parts.append(tensor.repeat_interleave(full_repeats, dim=1))
    61:     if remainder > 0:
    62:         parts.append(tensor[:, :remainder, :, :])
    63:     return torch.cat(parts, dim=1)
    64: 
    65: 
    66: class CausalSelfAttention(nn.Module):
    67:     def __init__(self, config, layer_idx=0):
    68:         super().__init__()
    69:         assert config.n_embd % config.n_head == 0
    70:         self.n_head = config.n_head
    71:         self.n_embd = config.n_embd
    72:         self.dropout = config.dropout
    73:         self.layer_idx = layer_idx
    74:         self.n_kv_head, self.head_dim = build_kv_heads(config)
    75:         self.share_across_layers = False
    76: 
    77:         q_dim = config.n_embd
    78:         kv_dim = 2 * self.n_kv_head * self.head_dim
    79:         self.c_attn = nn.Linear(config.n_embd, q_dim + kv_dim, bias=config.bias)
    80:         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    81:         self.attn_dropout = nn.Dropout(config.dropout)
    82:         self.resid_dropout = nn.Dropout(config.dropout)
    83:         self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    84:         if not self.flash:
    85:             self.register_buffer(
    86:                 "bias",
    87:                 torch.tril(torch.ones(config.block_size, config.block_size)).view(
    88:                     1, 1, config.block_size, config.block_size
    89:                 ),
    90:             )
    91:         self.use_pos_emb = True
    92:         self.head_sharing_ratio = 1.0
    93: 
    94:     def forward(self, x):
    95:         bsz, seq_len, channels = x.size()
    96:         qkv = self.c_attn(x)
    97:         q, kv = qkv.split([self.n_embd, 2 * self.n_kv_head * self.head_dim], dim=2)
    98:         k, v = kv.chunk(2, dim=2)
    99:         q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
   100:         k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
   101:         v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
   102:         k, v, latent_ratio = latent_kv_project(k, v, self)
   103:         self._last_latent_rank_ratio = float(latent_ratio)
   104:         self._last_kv_storage_ratio = 1.0
   105:         self._uses_latent_compression = False
   106:         y = torch.nn.functional.scaled_dot_product_attention(
   107:             q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
   108:         )
   109:         y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
   110:         y = self.resid_dropout(self.c_proj(y))
   111:         return y
   112: # END KV EDITABLE REGION
   113: 
   114: 
```

### `mqa` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 36–115:
    33: 
    34: # ── Editable region: KV structure design ──────────────────────────────────
    35: # BEGIN KV EDITABLE REGION
    36: def build_kv_heads(config):
    37:     """Use one shared KV head for all query heads."""
    38: 
    39:     n_kv_head = 1
    40:     head_dim = config.n_embd // config.n_head
    41:     return n_kv_head, head_dim
    42: 
    43: 
    44: def cross_layer_share(layer_idx, config):
    45:     return False
    46: 
    47: 
    48: def latent_kv_project(k, v, config):
    49:     return k, v, 1.0
    50: 
    51: 
    52: def expand_kv_to_q_heads(tensor, target_heads):
    53:     current_heads = tensor.size(1)
    54:     if current_heads == target_heads:
    55:         return tensor
    56:     full_repeats = target_heads // current_heads
    57:     remainder = target_heads % current_heads
    58:     parts = []
    59:     if full_repeats > 0:
    60:         parts.append(tensor.repeat_interleave(full_repeats, dim=1))
    61:     if remainder > 0:
    62:         parts.append(tensor[:, :remainder, :, :])
    63:     return torch.cat(parts, dim=1)
    64: 
    65: 
    66: class CausalSelfAttention(nn.Module):
    67:     def __init__(self, config, layer_idx=0):
    68:         super().__init__()
    69:         assert config.n_embd % config.n_head == 0
    70:         self.n_head = config.n_head
    71:         self.n_embd = config.n_embd
    72:         self.dropout = config.dropout
    73:         self.layer_idx = layer_idx
    74:         self.n_kv_head, self.head_dim = build_kv_heads(config)
    75:         self.share_across_layers = False
    76: 
    77:         q_dim = config.n_embd
    78:         kv_dim = 2 * self.n_kv_head * self.head_dim
    79:         self.c_attn = nn.Linear(config.n_embd, q_dim + kv_dim, bias=config.bias)
    80:         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    81:         self.attn_dropout = nn.Dropout(config.dropout)
    82:         self.resid_dropout = nn.Dropout(config.dropout)
    83:         self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    84:         if not self.flash:
    85:             self.register_buffer(
    86:                 "bias",
    87:                 torch.tril(torch.ones(config.block_size, config.block_size)).view(
    88:                     1, 1, config.block_size, config.block_size
    89:                 ),
    90:             )
    91:         self.use_pos_emb = True
    92:         self.head_sharing_ratio = float(self.n_head)
    93: 
    94:     def forward(self, x):
    95:         bsz, seq_len, channels = x.size()
    96:         qkv = self.c_attn(x)
    97:         q, kv = qkv.split(
    98:             [self.n_embd, 2 * self.n_kv_head * self.head_dim],
    99:             dim=2,
   100:         )
   101:         k, v = kv.chunk(2, dim=2)
   102:         q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
   103:         k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
   104:         v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
   105:         k = expand_kv_to_q_heads(k, self.n_head)
   106:         v = expand_kv_to_q_heads(v, self.n_head)
   107:         self._last_latent_rank_ratio = 1.0
   108:         self._last_kv_storage_ratio = 1.0
   109:         self._uses_latent_compression = False
   110:         y = torch.nn.functional.scaled_dot_product_attention(
   111:             q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
   112:         )
   113:         y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
   114:         y = self.resid_dropout(self.c_proj(y))
   115:         return y
   116: # END KV EDITABLE REGION
   117: 
   118: 
```

### `gqa` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 36–104:
    33: 
    34: # ── Editable region: KV structure design ──────────────────────────────────
    35: # BEGIN KV EDITABLE REGION
    36: def build_kv_heads(config):
    37:     """Use fewer KV heads than query heads, preserving query expressivity."""
    38: 
    39:     n_kv_head = max(1, config.n_head // 4)
    40:     while config.n_head % n_kv_head != 0:
    41:         n_kv_head -= 1
    42:     head_dim = config.n_embd // config.n_head
    43:     return n_kv_head, head_dim
    44: 
    45: 
    46: def cross_layer_share(layer_idx, config):
    47:     return False
    48: 
    49: 
    50: def latent_kv_project(k, v, config):
    51:     return k, v, 1.0
    52: 
    53: 
    54: class CausalSelfAttention(nn.Module):
    55:     def __init__(self, config, layer_idx=0):
    56:         super().__init__()
    57:         assert config.n_embd % config.n_head == 0
    58:         self.n_head = config.n_head
    59:         self.n_embd = config.n_embd
    60:         self.dropout = config.dropout
    61:         self.layer_idx = layer_idx
    62:         self.n_kv_head, self.head_dim = build_kv_heads(config)
    63:         self.share_across_layers = False
    64: 
    65:         q_dim = config.n_embd
    66:         kv_dim = 2 * self.n_kv_head * self.head_dim
    67:         self.c_attn = nn.Linear(config.n_embd, q_dim + kv_dim, bias=config.bias)
    68:         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    69:         self.attn_dropout = nn.Dropout(config.dropout)
    70:         self.resid_dropout = nn.Dropout(config.dropout)
    71:         self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    72:         if not self.flash:
    73:             self.register_buffer(
    74:                 "bias",
    75:                 torch.tril(torch.ones(config.block_size, config.block_size)).view(
    76:                     1, 1, config.block_size, config.block_size
    77:                 ),
    78:             )
    79:         self.use_pos_emb = True
    80:         self.head_sharing_ratio = self.n_head / max(self.n_kv_head, 1)
    81: 
    82:     def forward(self, x):
    83:         bsz, seq_len, channels = x.size()
    84:         qkv = self.c_attn(x)
    85:         q, kv = qkv.split(
    86:             [self.n_embd, 2 * self.n_kv_head * self.head_dim],
    87:             dim=2,
    88:         )
    89:         k, v = kv.chunk(2, dim=2)
    90:         q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
    91:         k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
    92:         v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
    93:         repeat_factor = self.n_head // self.n_kv_head
    94:         k = k.repeat_interleave(repeat_factor, dim=1)
    95:         v = v.repeat_interleave(repeat_factor, dim=1)
    96:         self._last_latent_rank_ratio = 1.0
    97:         self._last_kv_storage_ratio = 1.0
    98:         self._uses_latent_compression = False
    99:         y = torch.nn.functional.scaled_dot_product_attention(
   100:             q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
   101:         )
   102:         y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
   103:         y = self.resid_dropout(self.c_proj(y))
   104:         return y
   105: # END KV EDITABLE REGION
   106: 
   107: 
```

### `mla` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 36–216:
    33: 
    34: # ── Editable region: KV structure design ──────────────────────────────────
    35: # BEGIN KV EDITABLE REGION
    36: def build_kv_heads(config):
    37:     head_dim = config.n_embd // config.n_head
    38:     return 1, head_dim
    39: 
    40: 
    41: def cross_layer_share(layer_idx, config):
    42:     return False
    43: 
    44: 
    45: def latent_kv_project(k, v, config):
    46:     return k, v, 1.0
    47: 
    48: 
    49: class MLARMSNorm(nn.Module):
    50:     def __init__(self, hidden_size, eps=1e-6):
    51:         super().__init__()
    52:         self.weight = nn.Parameter(torch.ones(hidden_size))
    53:         self.eps = eps
    54: 
    55:     def forward(self, x):
    56:         input_dtype = x.dtype
    57:         x = x.to(torch.float32)
    58:         variance = x.pow(2).mean(-1, keepdim=True)
    59:         x = x * torch.rsqrt(variance + self.eps)
    60:         return self.weight * x.to(input_dtype)
    61: 
    62: 
    63: def rotate_half(x):
    64:     x1 = x[..., : x.shape[-1] // 2]
    65:     x2 = x[..., x.shape[-1] // 2 :]
    66:     return torch.cat((-x2, x1), dim=-1)
    67: 
    68: 
    69: def build_rotary_cache(seq_len, dim, device, dtype, theta=10000.0):
    70:     inv_freq = 1.0 / (
    71:         theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    72:     )
    73:     positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    74:     freqs = torch.outer(positions, inv_freq)
    75:     emb = torch.cat((freqs, freqs), dim=-1)
    76:     cos = emb.cos().to(dtype).view(1, 1, seq_len, dim)
    77:     sin = emb.sin().to(dtype).view(1, 1, seq_len, dim)
    78:     return cos, sin
    79: 
    80: 
    81: def apply_rotary_pos_emb_interleave(q, k, cos, sin):
    82:     # build_rotary_cache uses the half-split convention (cat((freqs, freqs), -1)),
    83:     # so rotate_half + the *cos/+sin formula below is already the correct form.
    84:     # The original view->transpose(4,3)->reshape re-interleave was needed only when
    85:     # loading DeepSeek-V2 pretrained weights in interleaved layout; for a from-scratch
    86:     # nanoGPT this permutation just adds a per-forward materialization per Q and K
    87:     # (~640MB total activation across 24 layers at B=32 T=1024). Drop it.
    88:     q_embed = (q * cos) + (rotate_half(q) * sin)
    89:     k_embed = (k * cos) + (rotate_half(k) * sin)
    90:     return q_embed, k_embed
    91: 
    92: 
    93: class CausalSelfAttention(nn.Module):
    94:     def __init__(self, config, layer_idx=0):
    95:         super().__init__()
    96:         assert config.n_embd % config.n_head == 0
    97:         self.n_head = config.n_head
    98:         self.n_embd = config.n_embd
    99:         self.dropout = config.dropout
   100:         self.layer_idx = layer_idx
   101:         self.n_kv_head, self.head_dim = build_kv_heads(config)
   102:         self.share_across_layers = False
   103: 
   104:         # DeepSeek/TransMLA treat qk_nope as the original dense head dimension
   105:         # and add a separate rotary slice on top, rather than partitioning the
   106:         # original head dim into two halves.
   107:         self.qk_rope_head_dim = min(64, self.head_dim)
   108:         self.qk_rope_head_dim = max(16, self.qk_rope_head_dim)
   109:         if self.qk_rope_head_dim % 2 != 0:
   110:             self.qk_rope_head_dim -= 1
   111:         self.qk_nope_head_dim = self.head_dim
   112:         self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
   113:         self.v_head_dim = self.head_dim
   114:         # Preserve the relative rank schedule used in DeepSeek-V2 style MLA
   115:         # while capping by the tiny nanoGPT hidden size.
   116:         self.q_lora_rank = min(self.n_embd, 12 * self.head_dim)
   117:         self.kv_lora_rank = max(16, self.head_dim // 2)
   118: 
   119:         self.q_a_proj = nn.Linear(config.n_embd, self.q_lora_rank, bias=False)
   120:         self.q_a_layernorm = MLARMSNorm(self.q_lora_rank)
   121:         self.q_b_proj = nn.Linear(
   122:             self.q_lora_rank, self.n_head * self.qk_head_dim, bias=config.bias
   123:         )
   124: 
   125:         self.kv_a_proj_with_mqa = nn.Linear(
   126:             config.n_embd, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.bias
   127:         )
   128:         self.kv_a_layernorm = MLARMSNorm(self.kv_lora_rank)
   129:         self.kv_b_proj = nn.Linear(
   130:             self.kv_lora_rank,
   131:             self.n_head * (self.qk_nope_head_dim + self.v_head_dim),
   132:             bias=False,
   133:         )
   134: 
   135:         self.o_proj = nn.Linear(self.n_head * self.v_head_dim, config.n_embd, bias=config.bias)
   136:         self.attn_dropout = nn.Dropout(config.dropout)
   137:         self.resid_dropout = nn.Dropout(config.dropout)
   138:         self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
   139:         if not self.flash:
   140:             self.register_buffer(
   141:                 "bias",
   142:                 torch.tril(torch.ones(config.block_size, config.block_size)).view(
   143:                     1, 1, config.block_size, config.block_size
   144:                 ),
   145:             )
   146:         self.use_pos_emb = False
   147:         self.head_sharing_ratio = float(self.n_head)
   148:         self.scaling = self.qk_head_dim ** -0.5
   149: 
   150:     def forward(self, x):
   151:         bsz, seq_len, _ = x.size()
   152: 
   153:         q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
   154:         q_states = q_states.view(bsz, seq_len, self.n_head, self.qk_head_dim).transpose(1, 2)
   155:         q_nope, q_rot = torch.split(
   156:             q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
   157:         )
   158: 
   159:         compressed_kv = self.kv_a_proj_with_mqa(x)
   160:         kv_latent, k_rot = torch.split(
   161:             compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
   162:         )
   163:         kv_states = self.kv_b_proj(self.kv_a_layernorm(kv_latent))
   164:         kv_states = kv_states.view(
   165:             bsz, seq_len, self.n_head, self.qk_nope_head_dim + self.v_head_dim
   166:         ).transpose(1, 2)
   167:         k_nope, value_states = torch.split(
   168:             kv_states, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
   169:         )
   170: 
   171:         k_rot = k_rot.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
   172:         cos, sin = build_rotary_cache(
   173:             seq_len, self.qk_rope_head_dim, x.device, q_rot.dtype
   174:         )
   175:         q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
   176: 
   177:         # DeepSeek-V2 official pattern: new_empty + slice-assign.
   178:         # Avoids k_rot.expand(-1, n_head, -1, -1) materialization (saves the
   179:         # expanded-contiguous intermediate) and the subsequent torch.cat's
   180:         # transient output buffer. slice __setitem__ is autograd-safe — the
   181:         # backward scatters gradients back into q_nope / q_rot / k_nope / k_rot
   182:         # (broadcast along head axis for k_rot).
   183:         query_states = q_states.new_empty(bsz, self.n_head, seq_len, self.qk_head_dim)
   184:         query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
   185:         query_states[:, :, :, self.qk_nope_head_dim :] = q_rot
   186: 
   187:         key_states = q_states.new_empty(bsz, self.n_head, seq_len, self.qk_head_dim)
   188:         key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
   189:         key_states[:, :, :, self.qk_nope_head_dim :] = k_rot  # broadcasts over n_head
   190: 
   191:         if self.flash:
   192:             y = torch.nn.functional.scaled_dot_product_attention(
   193:                 query_states,
   194:                 key_states,
   195:                 value_states,
   196:                 attn_mask=None,
   197:                 dropout_p=self.dropout if self.training else 0.0,
   198:                 is_causal=True,
   199:                 scale=self.scaling,
   200:             )
   201:         else:
   202:             att = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
   203:             att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
   204:             att = F.softmax(att, dim=-1)
   205:             att = self.attn_dropout(att)
   206:             y = torch.matmul(att, value_states)
   207: 
   208:         latent_ratio = self.kv_lora_rank / self.qk_head_dim
   209:         storage_ratio = (self.kv_lora_rank + self.qk_rope_head_dim) / (2 * self.head_dim)
   210:         self._last_latent_rank_ratio = float(latent_ratio)
   211:         self._last_kv_storage_ratio = float(storage_ratio)
   212:         self._uses_latent_compression = True
   213: 
   214:         y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_head * self.v_head_dim)
   215:         y = self.resid_dropout(self.o_proj(y))
   216:         return y
   217: # END KV EDITABLE REGION
   218: 
   219: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
