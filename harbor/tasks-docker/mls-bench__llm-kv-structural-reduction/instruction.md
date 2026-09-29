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

   Every tensor the attention keeps for past tokens (its KV cache) must be
   passed through the fixed helper `kv_cache(...)`, laid out as
   `(batch, seq_len, ...)` with each position's entry computed from that
   token's own input, and the attention must use the tensors it
   returns (the dense baselines call `k, v = kv_cache(k, v)`; MLA passes its
   compressed latent and rotary key). A layer that reuses an earlier layer's
   returned cache makes no call of its own.

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
- The evaluator measures the KV footprint from the tensors passed to
  `kv_cache(...)`, not from module attributes. It then re-runs every
  attention layer once per position with fresh unrelated inputs at all
  other positions, replaying the recorded cache before that position, and
  rejects the run (before training and again at the end) if the layer's
  output or its own cache entry at that position changes at all (the
  re-run must reproduce them bit for bit, so use deterministic kernels),
  i.e. if attention reads other tokens through anything other than its
  declared cache. A run in which no layer declares a cache is rejected too.
- KV budget: the submitted structure must realize at least a 4x KV
  reduction relative to the dense MHA control, i.e.
  `kv_bytes_per_token <= 1024` at 345M (dense MHA, which is what the
  unmodified template implements, measures 4096). A run above the budget
  has its score multiplied by `exp(-0.003 * (kv_bytes_per_token - 1024))`
  in every regime, so a dense-MHA submission scores near zero. Within the
  budget, further KV reduction is rewarded on a log scale (each halving
  counts the same) and traded off against model quality.

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
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

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
    45:     """Optionally reuse the previous layer's KV cache (default: no sharing)."""
    46: 
    47:     return False
    48: 
    49: 
    50: def latent_kv_project(k, v, config):
    51:     """Optional latent KV bottleneck.
    52: 
    53:     Default: identity projection.
    54:     """
    55: 
    56:     return k, v, 1.0
    57: 
    58: 
    59: def expand_kv_to_q_heads(tensor, target_heads):
    60:     """Expand KV heads to query heads while remaining safe for any head count."""
    61: 
    62:     current_heads = tensor.size(1)
    63:     if current_heads == target_heads:
    64:         return tensor
    65:     full_repeats = target_heads // current_heads
    66:     remainder = target_heads % current_heads
    67:     parts = []
    68:     if full_repeats > 0:
    69:         parts.append(tensor.repeat_interleave(full_repeats, dim=1))
    70:     if remainder > 0:
    71:         parts.append(tensor[:, :remainder, :, :])
    72:     return torch.cat(parts, dim=1)
    73: 
    74: 
    75: class CausalSelfAttention(nn.Module):
    76:     _shared_kv_cache = {}
    77: 
    78:     def __init__(self, config, layer_idx=0):
    79:         super().__init__()
    80:         assert config.n_embd % config.n_head == 0
    81:         self.n_head = config.n_head
    82:         self.n_embd = config.n_embd
    83:         self.dropout = config.dropout
    84:         self.layer_idx = layer_idx
    85:         self.n_kv_head, self.head_dim = build_kv_heads(config)
    86:         self.share_across_layers = cross_layer_share(layer_idx, config)
    87: 
    88:         q_dim = config.n_embd
    89:         kv_dim = 2 * self.n_kv_head * self.head_dim
    90:         self.c_attn = nn.Linear(config.n_embd, q_dim + kv_dim, bias=config.bias)
    91:         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
    92:         self.attn_dropout = nn.Dropout(config.dropout)
    93:         self.resid_dropout = nn.Dropout(config.dropout)
    94:         self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    95:         if not self.flash:
    96:             self.register_buffer(
    97:                 "bias",
    98:                 torch.tril(torch.ones(config.block_size, config.block_size)).view(
    99:                     1, 1, config.block_size, config.block_size
   100:                 ),
   101:             )
   102:         self.use_pos_emb = True
   103:         self.head_sharing_ratio = self.n_head / max(self.n_kv_head, 1)
   104: 
   105:     def forward(self, x):
   106:         bsz, seq_len, channels = x.size()
   107:         qkv = self.c_attn(x)
   108:         q, kv = qkv.split(
   109:             [self.n_embd, 2 * self.n_kv_head * self.head_dim],
   110:             dim=2,
   111:         )
   112:         k, v = kv.chunk(2, dim=2)
   113: 
   114:         q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
   115:         k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim)
   116:         v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim)
   117: 
   118:         reused_previous = False
   119:         if self.share_across_layers and (self.layer_idx - 1) in self._shared_kv_cache:
   120:             k, v = self._shared_kv_cache[self.layer_idx - 1]
   121:             reused_previous = True
   122:         else:
   123:             # Declare the per-token KV state (B, T, ...) that this layer caches.
   124:             k, v = kv_cache(k, v)
   125:             self._shared_kv_cache[self.layer_idx] = (k.detach(), v.detach())
   126:         k, v = k.transpose(1, 2), v.transpose(1, 2)
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
   209: # ── Fixed KV-cache accounting (not editable) ──────────────────────────────
   210: # kv_bytes_per_token is measured here from the tensors the attention layers
   211: # actually cache, never from attributes the editable code sets.  Every tensor
   212: # an attention layer keeps for past tokens must pass through kv_cache(...),
   213: # and the attention must use what kv_cache returns.  The probe below records
   214: # those tensors, then re-runs each layer once per position s with every
   215: # other position's input replaced by a fresh unrelated sequence; the cache
   216: # before s is replayed from the record, from s on the layer's own is used.
   217: # If the output at s or the layer's own cache entry at s changes, the layer
   218: # reads past tokens through something other than its declared cache (or
   219: # caches state that is not per-token) and the run is rejected.
   220: class _KVCacheProbe:
   221:     mode = None  # None (identity), "record" or "replay"
   222:     layer = None
   223:     shape = None
   224:     recorded = {}
   225:     replay = []
   226:     cursor = 0
   227:     split = 0
   228:     live = []
   229: 
   230: 
   231: _KV_PROBE = _KVCacheProbe()
   232: # Exact: an honest re-run reproduces every bit, and any slack (even relative
   233: # to the output's scale) leaves room to hide an undeclared read in it.
   234: _KV_PROBE_TOL = 0.0
   235: 
   236: 
   237: def kv_cache(*tensors):
   238:     """Declare the per-token KV state an attention layer caches.
   239: 
   240:     Pass every tensor the layer keeps for past tokens, laid out as
   241:     (batch, seq_len, ...) with each position's entry computed from that
   242:     token's own input, and use the returned tensors (same order; a single
   243:     tensor is returned unpacked).  Outside the evaluator's probe this is the
   244:     identity.  kv_bytes_per_token is the per-token size of these tensors,
   245:     counted at max(2, element_size) bytes per element, summed within a layer
   246:     and averaged over layers.  A layer that reuses an earlier layer's
   247:     returned cache without calling kv_cache adds no bytes.
   248:     """
   249:     if not tensors:
   250:         raise RuntimeError("kv_cache() needs at least one tensor")
   251:     for t in tensors:
   252:         if type(t) is not torch.Tensor or not t.is_floating_point() or t.dim() < 2 or t.numel() == 0:
   253:             raise RuntimeError(
   254:                 "kv_cache() takes non-empty plain floating-point tensors laid out as (batch, seq_len, ...)"
   255:             )
   256:     probe = _KV_PROBE
   257:     if probe.mode is None:
   258:         out = tensors
   259:     elif probe.layer is None:
   260:         raise RuntimeError("kv_cache() called outside an attention layer during the KV probe")
   261:     elif probe.mode == "record":
   262:         for t in tensors:
   263:             if tuple(t.shape[:2]) != probe.shape:
   264:                 raise RuntimeError(
   265:                     f"kv_cache() tensors must be laid out as (batch, seq_len, ...) = "
   266:                     f"{probe.shape}; got {tuple(t.shape)}"
   267:                 )
   268:             probe.recorded[probe.layer].append(t.detach().clone())
   269:         # Copies laid out like the replayed tensors below, so both passes compute alike.
   270:         out = [r.clone() for r in probe.recorded[probe.layer][-len(tensors):]]
   271:     elif probe.mode == "replay":
   272:         out = []
   273:         for t in tensors:
   274:             if probe.cursor >= len(probe.replay):
   275:                 raise RuntimeError("attention layer called kv_cache() more often than when recorded")
   276:             ref = probe.replay[probe.cursor]
   277:             probe.cursor += 1
   278:             if t.shape != ref.shape or t.dtype != ref.dtype:
   279:                 raise RuntimeError("attention layer changed its kv_cache() tensors between calls")
   280:             # The recorded cache before the probed position, this call's own from it on.
   281:             probe.live.append(t[:, probe.split].detach().clone())
   282:             r = t.detach().clone()
   283:             r[:, : probe.split] = ref[:, : probe.split]
   284:             out.append(r)
   285:     else:
   286:         raise RuntimeError(f"invalid KV probe mode {probe.mode!r}")
   287:     return out[0] if len(out) == 1 else tuple(out)
   288: 
   289: 
   290: @torch.no_grad()
   291: def measure_kv_cache(model, idx, ctx, _kv_cache_fn=kv_cache, _probe=_KV_PROBE):
   292:     """Measure kv_bytes_per_token from the realized cache and verify that the
   293:     cache is the only path by which attention reads past tokens.  Raises on
   294:     any violation."""
   295:     if globals().get("kv_cache") is not _kv_cache_fn or globals().get("_KV_PROBE") is not _probe:
   296:         raise RuntimeError("kv_cache / _KV_PROBE must not be rebound")
   297:     bsz, seq_len = idx.shape
   298:     # Unrelated tokens, secret and fresh on every call (the training RNG is untouched).
   299:     gen = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little") >> 1)
   300:     other = torch.randint(0, model.config.vocab_size, idx.shape, generator=gen).to(idx.device)
   301:     blocks = model.transformer.h
   302:     was_training = model.training
   303:     model.eval()
   304: 
   305:     def embed(tokens):
   306:         x = model.transformer.drop(model.transformer.wte(tokens))
   307:         if getattr(blocks[0].attn, "use_pos_emb", True):
   308:             pos = torch.arange(0, tokens.size(1), dtype=torch.long, device=tokens.device)
   309:             x = x + model.transformer.wpe(pos)
   310:         return x
   311: 
   312:     try:
   313:         with ctx:
   314:             # Pass 1: record every layer's declared cache (same math as Block.forward).
   315:             _probe.mode, _probe.shape, _probe.recorded = "record", (bsz, seq_len), {}
   316:             inputs, outputs = [], []
   317:             x = embed(idx)
   318:             for li, block in enumerate(blocks):
   319:                 h = block.ln_1(x)
   320:                 _probe.layer = li
   321:                 _probe.recorded[li] = []
   322:                 a = block.attn(h)
   323:                 _probe.layer = None
   324:                 inputs.append(h)
   325:                 outputs.append(a)
   326:                 x = x + a
   327:                 x = x + block.mlp(block.ln_2(x))
   328:             _probe.mode = None
   329:             # The layer inputs of an unrelated sequence (this pass also
   330:             # overwrites whatever state a layer kept from the recorded pass).
   331:             others = []
   332:             x = embed(other)
   333:             for block in blocks:
   334:                 h = block.ln_1(x)
   335:                 others.append(h)
   336:                 x = x + block.attn(h)
   337:                 x = x + block.mlp(block.ln_2(x))
   338:             # Pass 2: for every position s, feed every layer the unrelated
   339:             # sequence everywhere except s; kv_cache() replays the recorded
   340:             # cache before s and passes the layer's own cache from s on.  The
   341:             # output at s and the layer's own cache entry at s must not move.
   342:             for s in range(seq_len):
   343:                 errs, where = [], []
   344:                 for li, block in enumerate(blocks):
   345:                     h = others[li].clone()
   346:                     h[:, s] = inputs[li][:, s]
   347:                     _probe.mode, _probe.replay, _probe.cursor, _probe.layer = "replay", _probe.recorded[li], 0, li
   348:                     _probe.split, _probe.live = s, []
   349:                     a = block.attn(h)
   350:                     if _probe.cursor != len(_probe.replay):
   351:                         raise RuntimeError(
   352:                             f"layer {li}: kv_cache() received {_probe.cursor} tensors, "
   353:                             f"{len(_probe.replay)} when recorded"
   354:                         )
   355:                     _probe.mode, _probe.layer = None, None
   356:                     pairs = [(a[:, s], outputs[li][:, s])]
   357:                     pairs += [(live, rec[:, s]) for live, rec in zip(_probe.live, _probe.recorded[li])]
   358:                     for j, (got, ref) in enumerate(pairs):
   359:                         ref = ref.float()
   360:                         errs.append((got.float() - ref).abs().max() / (ref.abs().max() + 1e-6))
   361:                         where.append((li, "attention output" if j == 0 else f"kv_cache() tensor {j - 1}"))
   362:                 # One device sync per position: any change beyond the tolerance fails.
   363:                 for (li, what), rel in zip(where, torch.stack(errs).tolist()):
   364:                     if not rel <= _KV_PROBE_TOL:
   365:                         raise RuntimeError(
   366:                             f"layer {li}: {what} at position {s} depends on other positions' tokens through "
   367:                             f"state not declared via kv_cache() (max |diff| {rel:.3g} of its scale); "
   368:                             f"kv_bytes_per_token cannot be measured"
   369:                         )
   370:     finally:
   371:         _probe.mode, _probe.layer, _probe.shape = None, None, None
   372:         _probe.replay, _probe.cursor, _probe.split, _probe.live = [], 0, 0, []
   373:         model.train(was_training)
   374: 
   375:     per_layer = [
   376:         float(
   377:             sum(t.numel() // (bsz * seq_len) * max(2, t.element_size()) for t in _probe.recorded[li])
   378:         )
   379:         for li in range(len(blocks))
   380:     ]
   381:     _probe.recorded = {}
   382:     if not sum(per_layer) > 0:
   383:         raise RuntimeError("no attention layer declared a KV cache via kv_cache()")
   384:     return {
   385:         "kv_bytes_per_token": sum(per_layer) / len(per_layer),
   386:         "kv_bytes_per_layer": per_layer,
   387:     }
   388: 
   389: 
   390: class MLP(nn.Module):
   391:     def __init__(self, config):
   392:         super().__init__()
   393:         self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
   394:         self.gelu = nn.GELU()
   395:         self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
   396:         self.dropout = nn.Dropout(config.dropout)
   397: 
   398:     def forward(self, x):
   399:         x = self.c_fc(x)
   400:         x = self.gelu(x)
   401:         x = self.c_proj(x)
   402:         x = self.dropout(x)
   403:         return x
   404: 
   405: 
   406: class Block(nn.Module):
   407:     def __init__(self, config, layer_idx):
   408:         super().__init__()
   409:         self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
   410:         self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
   411:         self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
   412:         self.mlp = MLP(config)
   413: 
   414:     def forward(self, x):
   415:         x = x + self.attn(self.ln_1(x))
   416:         x = x + self.mlp(self.ln_2(x))
   417:         return x
   418: 
   419: 
   420: @dataclass
   421: class GPTConfig:
   422:     block_size: int = 1024
   423:     vocab_size: int = 50304
   424:     n_layer: int = 12
   425:     n_head: int = 12
   426:     n_embd: int = 768
   427:     dropout: float = 0.0
   428:     bias: bool = False
   429: 
   430: 
   431: class GPT(nn.Module):
   432:     def __init__(self, config):
   433:         super().__init__()
   434:         self.config = config
   435:         self.transformer = nn.ModuleDict(
   436:             dict(
   437:                 wte=nn.Embedding(config.vocab_size, config.n_embd),
   438:                 wpe=nn.Embedding(config.block_size, config.n_embd),
   439:                 drop=nn.Dropout(config.dropout),
   440:                 h=nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
   441:                 ln_f=LayerNorm(config.n_embd, bias=config.bias),
   442:             )
   443:         )
   444:         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
   445:         self.transformer.wte.weight = self.lm_head.weight
   446:         self.apply(self._init_weights)
   447:         for pn, p in self.named_parameters():
   448:             if pn.endswith("c_proj.weight"):
   449:                 torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
   450:         print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
   451: 
   452:     def get_num_params(self, non_embedding=True):
   453:         n_params = sum(p.numel() for p in self.parameters())
   454:         if non_embedding:
   455:             n_params -= self.transformer.wpe.weight.numel()
   456:         return n_params
   457: 
   458:     def _init_weights(self, module):
   459:         if isinstance(module, nn.Linear):
   460:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   461:             if module.bias is not None:
   462:                 torch.nn.init.zeros_(module.bias)
   463:         elif isinstance(module, nn.Embedding):
   464:             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
   465: 
   466:     def structural_metrics(self):
   467:         # Descriptive only.  kv_bytes_per_token is measured by measure_kv_cache().
   468:         head_sharing = []
   469:         latent_rank = []
   470:         for block in self.transformer.h:
   471:             attn = block.attn
   472:             n_head = int(getattr(attn, "n_head", self.config.n_head))
   473:             n_kv_head = int(getattr(attn, "n_kv_head", n_head))
   474:             head_dim = int(getattr(attn, "head_dim", self.config.n_embd // self.config.n_head))
   475:             head_sharing.append(n_head / max(n_kv_head, 1))
   476: 
   477:             if getattr(attn, "share_across_layers", False):
   478:                 latent_rank.append(0.0)
   479:             elif hasattr(attn, "kv_a_proj_with_mqa") and hasattr(attn, "kv_b_proj"):
   480:                 if hasattr(attn, "kv_a_layernorm"):
   481:                     kv_lora_rank = int(attn.kv_a_layernorm.weight.numel())
   482:                 else:
   483:                     kv_lora_rank = int(getattr(attn, "kv_lora_rank", head_dim))
   484:                 qk_rope_head_dim = int(attn.kv_a_proj_with_mqa.out_features - kv_lora_rank)
   485:                 qk_head_dim = int(getattr(attn, "qk_head_dim", head_dim + qk_rope_head_dim))
   486:                 latent_rank.append(kv_lora_rank / max(qk_head_dim, 1))
   487:             else:
   488:                 latent_rank.append(1.0)
   489:         return {
   490:             "head_sharing_ratio": sum(head_sharing) / len(head_sharing),
   491:             "latent_rank_ratio": sum(latent_rank) / len(latent_rank),
   492:         }
   493: 
   494:     def forward(self, idx, targets=None):
   495:         device = idx.device
   496:         _, t = idx.size()
   497:         assert t <= self.config.block_size
   498:         tok_emb = self.transformer.wte(idx)
   499:         x = self.transformer.drop(tok_emb)
   500:         use_pos = getattr(self.transformer.h[0].attn, "use_pos_emb", True)

[truncated: showing at most 500 lines / 60000 bytes from nanoGPT/custom_pretrain.py]
```

## Parameter Budget

Keep your model's total parameter count at or below the strongest reference baseline's. A check runs automatically — you don't need to invoke it — and a materially larger model makes the run invalid. The contribution must be algorithmic, not extra capacity.

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `mha` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 36–113:
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
   100:         k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim)
   101:         v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim)
   102:         k, v = kv_cache(k, v)  # the per-token KV state this layer caches
   103:         k, v = k.transpose(1, 2), v.transpose(1, 2)
   104:         k, v, latent_ratio = latent_kv_project(k, v, self)
   105:         self._last_latent_rank_ratio = float(latent_ratio)
   106:         self._last_kv_storage_ratio = 1.0
   107:         self._uses_latent_compression = False
   108:         y = torch.nn.functional.scaled_dot_product_attention(
   109:             q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
   110:         )
   111:         y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
   112:         y = self.resid_dropout(self.c_proj(y))
   113:         return y
   114: # END KV EDITABLE REGION
   115: 
   116: 
```

### `mqa` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 36–117:
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
   103:         k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim)
   104:         v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim)
   105:         k, v = kv_cache(k, v)  # the per-token KV state this layer caches
   106:         k, v = k.transpose(1, 2), v.transpose(1, 2)
   107:         k = expand_kv_to_q_heads(k, self.n_head)
   108:         v = expand_kv_to_q_heads(v, self.n_head)
   109:         self._last_latent_rank_ratio = 1.0
   110:         self._last_kv_storage_ratio = 1.0
   111:         self._uses_latent_compression = False
   112:         y = torch.nn.functional.scaled_dot_product_attention(
   113:             q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
   114:         )
   115:         y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
   116:         y = self.resid_dropout(self.c_proj(y))
   117:         return y
   118: # END KV EDITABLE REGION
   119: 
   120: 
```

### `gqa` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 36–106:
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
    91:         k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim)
    92:         v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim)
    93:         k, v = kv_cache(k, v)  # the per-token KV state this layer caches
    94:         k, v = k.transpose(1, 2), v.transpose(1, 2)
    95:         repeat_factor = self.n_head // self.n_kv_head
    96:         k = k.repeat_interleave(repeat_factor, dim=1)
    97:         v = v.repeat_interleave(repeat_factor, dim=1)
    98:         self._last_latent_rank_ratio = 1.0
    99:         self._last_kv_storage_ratio = 1.0
   100:         self._uses_latent_compression = False
   101:         y = torch.nn.functional.scaled_dot_product_attention(
   102:             q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
   103:         )
   104:         y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
   105:         y = self.resid_dropout(self.c_proj(y))
   106:         return y
   107: # END KV EDITABLE REGION
   108: 
   109: 
```

### `mla` baseline — editable region  [READ-ONLY — reference implementation]

In `nanoGPT/custom_pretrain.py`:

```python
Lines 36–218:
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
   163:         # MLA caches only the compressed latent and the shared rotary key.
   164:         kv_latent, k_rot = kv_cache(kv_latent, k_rot)
   165:         kv_states = self.kv_b_proj(self.kv_a_layernorm(kv_latent))
   166:         kv_states = kv_states.view(
   167:             bsz, seq_len, self.n_head, self.qk_nope_head_dim + self.v_head_dim
   168:         ).transpose(1, 2)
   169:         k_nope, value_states = torch.split(
   170:             kv_states, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
   171:         )
   172: 
   173:         k_rot = k_rot.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
   174:         cos, sin = build_rotary_cache(
   175:             seq_len, self.qk_rope_head_dim, x.device, q_rot.dtype
   176:         )
   177:         q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
   178: 
   179:         # DeepSeek-V2 official pattern: new_empty + slice-assign.
   180:         # Avoids k_rot.expand(-1, n_head, -1, -1) materialization (saves the
   181:         # expanded-contiguous intermediate) and the subsequent torch.cat's
   182:         # transient output buffer. slice __setitem__ is autograd-safe — the
   183:         # backward scatters gradients back into q_nope / q_rot / k_nope / k_rot
   184:         # (broadcast along head axis for k_rot).
   185:         query_states = q_states.new_empty(bsz, self.n_head, seq_len, self.qk_head_dim)
   186:         query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
   187:         query_states[:, :, :, self.qk_nope_head_dim :] = q_rot
   188: 
   189:         key_states = q_states.new_empty(bsz, self.n_head, seq_len, self.qk_head_dim)
   190:         key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
   191:         key_states[:, :, :, self.qk_nope_head_dim :] = k_rot  # broadcasts over n_head
   192: 
   193:         if self.flash:
   194:             y = torch.nn.functional.scaled_dot_product_attention(
   195:                 query_states,
   196:                 key_states,
   197:                 value_states,
   198:                 attn_mask=None,
   199:                 dropout_p=self.dropout if self.training else 0.0,
   200:                 is_causal=True,
   201:                 scale=self.scaling,
   202:             )
   203:         else:
   204:             att = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
   205:             att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
   206:             att = F.softmax(att, dim=-1)
   207:             att = self.attn_dropout(att)
   208:             y = torch.matmul(att, value_states)
   209: 
   210:         latent_ratio = self.kv_lora_rank / self.qk_head_dim
   211:         storage_ratio = (self.kv_lora_rank + self.qk_rope_head_dim) / (2 * self.head_dim)
   212:         self._last_latent_rank_ratio = float(latent_ratio)
   213:         self._last_kv_storage_ratio = float(storage_ratio)
   214:         self._uses_latent_compression = True
   215: 
   216:         y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_head * self.v_head_dim)
   217:         y = self.resid_dropout(self.o_proj(y))
   218:         return y
   219: # END KV EDITABLE REGION
   220: 
   221: 
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
