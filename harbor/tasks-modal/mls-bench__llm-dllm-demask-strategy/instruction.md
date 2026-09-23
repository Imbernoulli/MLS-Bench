# MLS-Bench: llm-dllm-demask-strategy

# Masked Diffusion LM: Demasking Strategy

## Research Question
Design a better demasking (decoding) strategy for masked diffusion language models. The strategy must generalize across **different decoding regimes**:

- **Block-based semi-autoregressive decoding** for downstream-task accuracy.
- **Fully-parallel decoding** for open-ended text generation.

## Background
Masked diffusion LMs generate by starting from a fully masked generation region and iteratively unmasking over `steps` denoising iterations. A demasking strategy decides at each step:

1. **Schedule**: how many tokens to unmask.
2. **Position selection**: which masked positions to unmask.
3. **Token assignment**: what token id to place.

Decoding can be **semi-autoregressive** (when `block_length < gen_length`, process one block at a time) or **fully parallel** (`block_length == gen_length`, all positions decoded together).

Reference papers:
- LLaDA (Nie et al., 2025; arXiv:2502.09992) — "Large Language Diffusion Models"; introduces LLaDA-8B-Base / LLaDA-8B-Instruct.
- Dream 7B (Ye, Xie, et al., 2025; arXiv:2508.15487) — "Dream 7B: Diffusion Large Language Models"; supports arbitrary-order generation and tunable quality–speed trade-offs.
- KLASS (Kim et al., NeurIPS 2025 Spotlight; arXiv:2511.05664) — "KLASS: KL-Guided Fast Inference in Masked Diffusion Models"; KL-adaptive stability sampling for unmasking multiple tokens per step.

## Fixed Pipeline
- The pretrained models, prompts, evaluation data, and task runners are fixed by the harness and not editable.
- Block scheduling constraint: `gen_length % block_length == 0`. When equal, decoding is fully parallel.
- Blocks are processed sequentially (no early-decoding into later blocks).
- The same `DemaskDecoder` must work in both semi-autoregressive and fully-parallel regimes.

## What you can modify
The `DemaskDecoder` class in `LLaDA/custom_demask_eval.py`.

### Interface
```python
class DemaskDecoder:
    def __init__(self, mask_id, temperature=0.0,
                 conf_threshold=0.9, kl_threshold=0.01, history_length=2):
        ...

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length, steps, block_length):
        # Returns (x_output [1, prompt_len + gen_length], used_steps)
```

`get_num_transfer_tokens(mask, steps)` is available outside the editable region — it returns the uniform schedule (`mask.sum() // steps` per step). Always return shape `[1, prompt_len + gen_length]`. `used_steps` counts model forward passes (lower = more efficient). The harness measures `avg_steps` itself: `model` is a handle that counts every forward call (each sequence in a batched call counts as one pass), and the counted passes, not the returned `used_steps`, are what gets scored.

## Reference baseline strategies
- `confidence_greedy` — LLaDA's `low_confidence` remasking: top-k by max prob.
- `topk_margin` — Dream's `topk_margin`: top-k by (top1 prob − top2 prob).
- `klass` — KLASS: KL-adaptive stability + confidence thresholds (KLASS paper, default `kl_threshold=0.01`, `conf_threshold=0.9`, `history_length=2`).

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/LLaDA/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits that change code outside these ranges — or creating new files, or
deleting whole files — will cause your submission to be invalid.

The line numbers mark an editable **region**, not a fixed line-count budget: you
may add or remove lines inside it. Only code outside the editable ranges must
stay unchanged.

- `LLaDA/custom_demask_eval.py`
- editable lines **59–151**




## Readable Context


### `LLaDA/custom_demask_eval.py`  [EDITABLE — lines 59–151 only]

```python
     1: """Downstream task evaluation (MATH, HumanEval) for masked diffusion LMs.
     2: 
     3: Following the KLASS evaluation protocol (Kim et al., NeurIPS 2025):
     4:   https://github.com/shkim0116/KLASS
     5: """
     6: 
     7: from __future__ import annotations
     8: 
     9: import argparse
    10: import gzip
    11: import json
    12: import os
    13: import re
    14: import sys
    15: import time
    16: from pathlib import Path
    17: 
    18: import numpy as np
    19: import torch
    20: import torch.nn.functional as F
    21: 
    22: MODEL_CONFIGS = {
    23:     "llada": {"path": os.environ.get("LLADA_INSTRUCT_PATH", "/data/llada-instruct"),
    24:               "mask_id": 126336},
    25:     "dream": {"path": "Dream-org/Dream-v0-Instruct-7B", "mask_id": None},
    26: }
    27: 
    28: 
    29: def load_instruct_model(name: str, device: str = "cuda"):
    30:     from transformers import AutoModel, AutoTokenizer
    31:     cfg = MODEL_CONFIGS[name]
    32:     tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    33:     mdl = AutoModel.from_pretrained(cfg["path"], trust_remote_code=True,
    34:                                     torch_dtype=torch.bfloat16).to(device).eval()
    35:     mid = cfg["mask_id"] or getattr(mdl.config, "mask_token_id", None) \
    36:                         or getattr(tok, "mask_token_id", None)
    37:     assert mid is not None
    38:     return mdl, tok, int(mid)
    39: 
    40: 
    41: def get_num_transfer_tokens(mask_index, steps):
    42:     """Uniform schedule: mask_num / steps tokens per step (+1 remainder)."""
    43:     mask_num = mask_index.sum(dim=1, keepdim=True)
    44:     base = mask_num // steps
    45:     remainder = mask_num % steps
    46:     out = torch.zeros(mask_num.size(0), steps,
    47:                       device=mask_index.device, dtype=torch.int64) + base
    48:     for i in range(mask_num.size(0)):
    49:         out[i, :remainder[i]] += 1
    50:     return out
    51: 
    52: 
    53: # ====================================================================
    54: # EDITABLE REGION — DemaskDecoder
    55: # ====================================================================
    56: 
    57: 
    58: 
    59: class DemaskDecoder:
    60:     """Masked-diffusion decoding strategy. Reference: KLASS (Kim et al.,
    61:     NeurIPS 2025).
    62: 
    63:     Performs semi-autoregressive decoding in blocks of length `block_length`.
    64:     Within each block, at each step it decides which masked positions to
    65:     unmask based on confidence and stability (KL divergence across steps).
    66: 
    67:     Params (set in __init__):
    68:       mask_id            : mask token id (LLaDA=126336, Dream=from tokenizer)
    69:       temperature        : Gumbel-max sampling temperature (0 = argmax)
    70:       conf_threshold     : min top-1 prob for a position to be "confident"
    71:       kl_threshold       : max KL over history for a position to be "stable"
    72:       history_length     : # recent steps to require stability over
    73: 
    74:     decode() returns (x_output [1, prompt_len+gen_len], used_steps).
    75:     """
    76: 
    77:     def __init__(self, mask_id: int, temperature: float = 0.0,
    78:                  conf_threshold: float = 0.9, kl_threshold: float = 0.01,
    79:                  history_length: int = 2):
    80:         self.mask_id = mask_id
    81:         self.temperature = temperature
    82:         self.conf_threshold = conf_threshold
    83:         self.kl_threshold = kl_threshold
    84:         self.history_length = history_length
    85: 
    86:     @torch.no_grad()
    87:     def decode(self, model, input_ids, gen_length: int, steps: int,
    88:                block_length: int):
    89:         mid = self.mask_id
    90:         x = torch.full((1, input_ids.shape[1] + gen_length), mid,
    91:                        dtype=torch.long, device=model.device)
    92:         x[:, :input_ids.shape[1]] = input_ids.clone()
    93:         assert gen_length % block_length == 0
    94:         num_blocks = gen_length // block_length
    95:         assert steps % num_blocks == 0
    96:         steps_per_block = steps // num_blocks
    97: 
    98:         V = model.lm_head.out_features if hasattr(model, "lm_head") \
    99:                                        else model.config.vocab_size
   100:         kl_hist = torch.zeros((1, x.shape[1], self.history_length),
   101:                               dtype=torch.float64, device=x.device)
   102:         p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64,
   103:                              device=x.device)
   104:         used = 0
   105: 
   106:         for b in range(num_blocks):
   107:             bs = input_ids.shape[1] + b * block_length
   108:             be = bs + block_length
   109:             num_xfer = get_num_transfer_tokens(
   110:                 (x[:, bs:be] == mid), steps_per_block)
   111: 
   112:             for step in range(steps_per_block):
   113:                 mask_idx = (x == mid)
   114:                 block_m = torch.zeros_like(mask_idx)
   115:                 block_m[:, bs:be] = True
   116:                 mask_idx = mask_idx & block_m
   117:                 if not mask_idx.any():
   118:                     break
   119: 
   120:                 logits = model(x).logits
   121:                 p_curr = F.softmax(logits.to(torch.float64), dim=-1)
   122:                 x0 = torch.argmax(p_curr, dim=-1)
   123:                 conf = torch.gather(p_curr, -1, x0.unsqueeze(-1)).squeeze(-1)
   124: 
   125:                 eps = 1e-12
   126:                 kl = (p_curr * (torch.log(p_curr + eps)
   127:                                 - torch.log(p_prev + eps))).sum(-1)
   128:                 kl_hist = torch.roll(kl_hist, -1, dims=-1)
   129:                 kl_hist[..., -1] = kl
   130:                 p_prev = p_curr.clone()
   131: 
   132:                 # KLASS: ready = stable ∩ confident ∩ still-masked
   133:                 if step >= self.history_length - 1:
   134:                     stable = torch.all(kl_hist < self.kl_threshold, dim=-1)
   135:                 else:
   136:                     stable = torch.zeros_like(conf, dtype=torch.bool)
   137:                 ready = stable & (conf > self.conf_threshold) & mask_idx
   138: 
   139:                 xfer = torch.zeros_like(x0, dtype=torch.bool)
   140:                 for j in range(ready.shape[0]):
   141:                     rdy = torch.where(ready[j])[0]
   142:                     if len(rdy) > 0:
   143:                         xfer[j, rdy] = True
   144:                     else:
   145:                         c = conf[j].clone()
   146:                         c[~mask_idx[j]] = -float("inf")
   147:                         _, topk = torch.topk(c, int(num_xfer[j, step].item()))
   148:                         xfer[j, topk] = True
   149:                 x = torch.where(xfer, x0, x)
   150:                 used += 1
   151:         return x, used
   152: 
   153: 
   154: # ====================================================================
   155: # END OF EDITABLE REGION
   156: # ====================================================================
   157: 
   158: 
   159: # ---------------------------------------------------------------------------
   160: # Forward-pass accounting (fixed)
   161: # ---------------------------------------------------------------------------
   162: 
   163: def _count_denoiser_forwards(raw_model):
   164:     """Wrap the denoiser so the harness, not the decoder, counts its forwards.
   165: 
   166:     Returns (handle, forward_count). The decoder only ever receives `handle`,
   167:     which exposes what a decoding strategy needs (calling it / `.forward`,
   168:     `.device`, `.dtype`, `.config`, and `.lm_head` when the model has one)
   169:     and keeps the raw model in a closure. Every call adds the number of
   170:     sequences in its batch to a counter that only `forward_count()` reads.
   171:     avg_steps is computed from this counter, not from the value the decoder
   172:     reports.
   173:     """
   174:     n = [0]
   175: 
   176:     def _rows(args, kwargs):
   177:         x = args[0] if args else kwargs.get("input_ids",
   178:                                             kwargs.get("inputs_embeds"))
   179:         if torch.is_tensor(x) and x.dim() >= 2:
   180:             return int(x.shape[0])
   181:         return 1
   182: 
   183:     def _forward(*args, **kwargs):
   184:         n[0] += _rows(args, kwargs)
   185:         return raw_model(*args, **kwargs)
   186: 
   187:     class DenoiserHandle:
   188:         __slots__ = ()
   189:         device = property(lambda self: raw_model.device)
   190:         dtype = property(lambda self: raw_model.dtype)
   191:         config = property(lambda self: raw_model.config)
   192: 
   193:         def __call__(self, *args, **kwargs):
   194:             return _forward(*args, **kwargs)
   195: 
   196:         def forward(self, *args, **kwargs):
   197:             return _forward(*args, **kwargs)
   198: 
   199:     if hasattr(raw_model, "lm_head"):
   200:         DenoiserHandle.lm_head = property(lambda self: raw_model.lm_head)
   201:     return DenoiserHandle(), (lambda: n[0])
   202: 
   203: 
   204: def _measured_steps(forward_count, before: int, reported, n_warned: list) -> int:
   205:     """Forward passes the decoder actually made in one decode() call."""
   206:     used = forward_count() - before
   207:     if reported != used and n_warned[0] < 3:
   208:         n_warned[0] += 1
   209:         print(f"[WARN] decode() reported used_steps={reported!r} but made "
   210:               f"{used} denoiser forward passes; avg_steps counts the "
   211:               f"measured passes.", flush=True)
   212:     return used
   213: 
   214: 
   215: # ---------------------------------------------------------------------------
   216: # Data loading
   217: # ---------------------------------------------------------------------------
   218: 
   219: def load_math(path: str) -> list[dict]:
   220:     with open(path) as f:
   221:         return [json.loads(line) for line in f if line.strip()]
   222: 
   223: 
   224: def load_humaneval(path: str) -> list[dict]:
   225:     opener = gzip.open if path.endswith(".gz") else open
   226:     with opener(path, "rt") as f:
   227:         return [json.loads(line) for line in f if line.strip()]
   228: 
   229: 
   230: # ---------------------------------------------------------------------------
   231: # MATH evaluation (uses klass_utils extract_math_answer + compare_answers)
   232: # ---------------------------------------------------------------------------
   233: 
   234: def _import_klass_utils():
   235:     """Import klass_utils from task data dir (mounted at /workspace/_task)."""
   236:     task_dir = os.environ.get("TASK_DIR", "/workspace/_task")
   237:     sys.path.insert(0, os.path.join(task_dir, "data"))
   238:     import klass_utils as ku
   239:     return ku
   240: 
   241: 
   242: def eval_math(model, tokenizer, decoder: DemaskDecoder, problems: list[dict],
   243:               gen_length: int, steps: int, block_length: int, forward_count):
   244:     ku = _import_klass_utils()
   245:     n_warned = [0]
   246:     sys_msg = ("Your task is to answer the question below. Give step by step "
   247:                "reasoning before you answer, and when you're ready to answer, "
   248:                "please use the format 'The final answer is'.")
   249:     correct = 0
   250:     total_steps = 0
   251:     for i, ex in enumerate(problems):
   252:         msgs = [{"role": "system", "content": sys_msg},
   253:                 {"role": "user", "content": ex["problem"]}]
   254:         prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
   255:                                                tokenize=False)
   256:         input_ids = torch.tensor(tokenizer(prompt)["input_ids"],
   257:                                  device=model.device).unsqueeze(0)
   258:         gt = ku.extract_math_answer(ex["problem"], ex["solution"])
   259:         before = forward_count()
   260:         x_out, used = decoder.decode(model, input_ids, gen_length, steps,
   261:                                      block_length)
   262:         used = _measured_steps(forward_count, before, used, n_warned)
   263:         gen_text = tokenizer.batch_decode(
   264:             x_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
   265:         pred = ku.extract_math_answer(ex["problem"], gen_text)
   266:         is_correct = ku.compare_answers(ex["problem"], gt, pred)
   267:         if i < 2:
   268:             print(f"[DEBUG] math example {i}:\n"
   269:                   f"  problem: {ex['problem'][:150]}\n"
   270:                   f"  gt={gt}\n"
   271:                   f"  gen (first 400 chars): {gen_text[:400]}\n"
   272:                   f"  pred={pred} correct={is_correct}", flush=True)
   273:         if is_correct:
   274:             correct += 1
   275:         total_steps += used
   276:         if (i + 1) % 10 == 0:
   277:             print(f"TRAIN_METRICS: math {i+1}/{len(problems)} "
   278:                   f"acc={correct/(i+1):.3f} "
   279:                   f"avg_steps={total_steps/(i+1):.1f}", flush=True)
   280:     return correct / max(len(problems), 1), total_steps / max(len(problems), 1)
   281: 
   282: 
   283: # ---------------------------------------------------------------------------
   284: # HumanEval evaluation (uses klass_utils evaluate_task)
   285: # ---------------------------------------------------------------------------
   286: 
   287: def _run_humaneval(code: str, test: str, entry_point: str) -> bool:
   288:     """Exec code + test + check(entry_point) in fresh namespace."""
   289:     try:
   290:         ns: dict = {}
   291:         exec(code + "\n" + test + f"\ncheck({entry_point})\n", ns)
   292:         return True
   293:     except Exception:
   294:         return False
   295: 
   296: 
   297: def check_humaneval_code(code: str, problem: dict, timeout: float = 3.0) -> bool:
   298:     import multiprocessing
   299:     entry = problem["entry_point"]
   300:     # If generated code lacks the function def, prepend problem prompt
   301:     # (which provides function signature + docstring).
   302:     if f"def {entry}" not in code:
   303:         code = problem["prompt"] + code
   304:     with multiprocessing.Pool(processes=1) as pool:
   305:         res = pool.apply_async(_run_humaneval, (code, problem["test"], entry))
   306:         try:
   307:             return bool(res.get(timeout=timeout))
   308:         except Exception:
   309:             return False
   310: 
   311: 
   312: def eval_humaneval(model, tokenizer, decoder: DemaskDecoder,
   313:                    problems: list[dict], gen_length: int, steps: int,
   314:                    block_length: int, forward_count):
   315:     n_warned = [0]
   316:     passed = 0
   317:     total_steps = 0
   318:     for i, p in enumerate(problems):
   319:         msgs = [{"role": "system", "content": "You complete only Python code."},
   320:                 {"role": "user", "content": p["prompt"]}]
   321:         prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
   322:                                                tokenize=False)
   323:         input_ids = torch.tensor(tokenizer(prompt)["input_ids"],
   324:                                  device=model.device).unsqueeze(0)
   325:         before = forward_count()
   326:         x_out, used = decoder.decode(model, input_ids, gen_length, steps,
   327:                                      block_length)
   328:         used = _measured_steps(forward_count, before, used, n_warned)
   329:         gen_text = tokenizer.batch_decode(
   330:             x_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
   331:         eos = tokenizer.eos_token or ""
   332:         if eos:
   333:             gen_text = gen_text.split(eos)[0]
   334:         m = re.search(r"```(?:python)?\n(.*?)(?:```|$)", gen_text, re.DOTALL)
   335:         code = m.group(1).strip() if m else gen_text.strip()
   336:         if i < 2:
   337:             print(f"[DEBUG] humaneval {p['entry_point']}:\n"
   338:                   f"gen (first 300 chars): {gen_text[:300]}\n"
   339:                   f"code (first 200 chars): {code[:200]}", flush=True)
   340:         ok = check_humaneval_code(code, p, timeout=3)
   341:         if ok:
   342:             passed += 1
   343:         total_steps += used
   344:         if (i + 1) % 10 == 0:
   345:             print(f"TRAIN_METRICS: humaneval {i+1}/{len(problems)} "
   346:                   f"pass@1={passed/(i+1):.3f} "
   347:                   f"avg_steps={total_steps/(i+1):.1f}", flush=True)
   348:     return passed / max(len(problems), 1), total_steps / max(len(problems), 1)
   349: 
   350: 
   351: # ---------------------------------------------------------------------------
   352: # Open-ended text generation evaluation (gen_ppl, MAUVE, entropy, rep2)
   353: # ---------------------------------------------------------------------------
   354: 
   355: def _truncate_at_eos(text: str, eos_tokens=("</s>", "<|endoftext|>", "<|im_end|>")):
   356:     for eos in eos_tokens:
   357:         idx = text.find(eos)
   358:         if idx >= 0:
   359:             text = text[:idx]
   360:     return text.strip()
   361: 
   362: 
   363: def compute_conditional_gen_ppl(prefix_texts, gen_texts, device):
   364:     from transformers import AutoModelForCausalLM, AutoTokenizer
   365:     import math as _m
   366:     tok = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
   367:     mdl = AutoModelForCausalLM.from_pretrained(
   368:         "openai-community/gpt2-large").to(device).eval()
   369:     total_loss, total_tokens = 0.0, 0
   370:     for prefix, gen in zip(prefix_texts, gen_texts):
   371:         if not gen.strip():
   372:             continue
   373:         p_ids = tok.encode(prefix, add_special_tokens=False)
   374:         g_ids = tok.encode(gen, add_special_tokens=False)
   375:         all_ids = (p_ids + g_ids)[:1024]
   376:         if len(p_ids) >= len(all_ids):
   377:             continue
   378:         ids = torch.tensor([all_ids], device=device)
   379:         with torch.no_grad():
   380:             logits = mdl(ids).logits[:, :-1, :]
   381:         labels = ids[:, 1:]
   382:         start = max(len(p_ids) - 1, 0)
   383:         loss = F.cross_entropy(
   384:             logits[:, start:, :].reshape(-1, logits.shape[-1]),
   385:             labels[:, start:].reshape(-1), reduction="sum")
   386:         total_loss += loss.item()
   387:         total_tokens += labels[:, start:].numel()
   388:     del mdl
   389:     torch.cuda.empty_cache()
   390:     return _m.exp(total_loss / total_tokens) if total_tokens else float("inf")
   391: 
   392: 
   393: def compute_mauve(gen_texts, ref_texts):
   394:     try:
   395:         import mauve
   396:         r = mauve.compute_mauve(p_text=ref_texts, q_text=gen_texts,
   397:                                 device_id=0 if torch.cuda.is_available() else -1,
   398:                                 max_text_length=512, verbose=False,
   399:                                 featurize_model_name="openai-community/gpt2-large")
   400:         return float(r.mauve)
   401:     except Exception as e:
   402:         print(f"[WARN] MAUVE failed: {e}", flush=True)
   403:         return 0.0
   404: 
   405: 
   406: def compute_entropy_rep2(texts):
   407:     from collections import Counter
   408:     import math as _m
   409:     all_bigrams = []
   410:     rep_ratios = []
   411:     for t in texts:
   412:         words = t.split()
   413:         bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
   414:         all_bigrams.extend(bigrams)
   415:         if bigrams:
   416:             rep_ratios.append(1.0 - len(set(bigrams)) / len(bigrams))
   417:         else:
   418:             rep_ratios.append(0.0)
   419:     ent = 0.0
   420:     if all_bigrams:
   421:         c = Counter(all_bigrams)
   422:         tot = sum(c.values())
   423:         for v in c.values():
   424:             p = v / tot
   425:             if p > 0:
   426:                 ent -= p * _m.log2(p)
   427:     rep2 = sum(rep_ratios) / max(len(rep_ratios), 1)
   428:     return ent, rep2
   429: 
   430: 
   431: def eval_text(model, tokenizer, decoder: DemaskDecoder, raw_texts: list[str],
   432:               prefix_len: int, gen_length: int, steps: int, block_length: int,
   433:               n_samples: int, seed: int, forward_count):
   434:     """Prefix-conditioned C4 continuation. Reports gen_ppl/MAUVE/entropy/rep2."""
   435:     import random as _r
   436:     rng = _r.Random(seed)
   437:     if len(raw_texts) > n_samples:
   438:         raw_texts = rng.sample(raw_texts, n_samples)
   439: 
   440:     # Build prefix prompts (raw, no chat template — Dream-Instruct as a base LM)
   441:     prefix_ids_list, prefix_texts, valid_refs = [], [], []
   442:     for txt in raw_texts:
   443:         ids = tokenizer.encode(txt, add_special_tokens=False)
   444:         if len(ids) >= prefix_len + gen_length:
   445:             pids = ids[:prefix_len]
   446:             prefix_ids_list.append(pids)
   447:             prefix_texts.append(tokenizer.decode(pids, skip_special_tokens=True))
   448:             ref_ids = ids[prefix_len:prefix_len + gen_length]
   449:             valid_refs.append(tokenizer.decode(ref_ids, skip_special_tokens=True))
   450:     print(f"[INFO] kept {len(prefix_ids_list)}/{len(raw_texts)} texts "
   451:           f"long enough for prefix={prefix_len}+gen={gen_length}", flush=True)
   452: 
   453:     gen_texts, total_used = [], 0
   454:     n_warned = [0]
   455:     for i, pids in enumerate(prefix_ids_list):
   456:         ids = torch.tensor([pids], dtype=torch.long, device=model.device)
   457:         before = forward_count()
   458:         x_out, used = decoder.decode(model, ids, gen_length, steps, block_length)
   459:         used = _measured_steps(forward_count, before, used, n_warned)
   460:         gen = tokenizer.decode(x_out[0, ids.shape[1]:].tolist(),
   461:                                skip_special_tokens=True)
   462:         gen = _truncate_at_eos(gen)
   463:         gen_texts.append(gen)
   464:         total_used += used
   465:         if (i + 1) % 10 == 0:
   466:             print(f"TRAIN_METRICS: text {i+1}/{len(prefix_ids_list)} "
   467:                   f"avg_steps={total_used/(i+1):.1f}", flush=True)
   468: 
   469:     avg_steps = total_used / max(len(prefix_ids_list), 1)
   470:     print("[INFO] unloading gen model, computing GPT-2 ppl...", flush=True)
   471:     del model
   472:     torch.cuda.empty_cache()
   473: 
   474:     ppl = compute_conditional_gen_ppl(prefix_texts, gen_texts, "cuda")
   475:     mauve = compute_mauve(gen_texts, valid_refs)
   476:     entropy, rep2 = compute_entropy_rep2(gen_texts)
   477:     return ppl, mauve, entropy, rep2, avg_steps
   478: 
   479: 
   480: # ---------------------------------------------------------------------------
   481: # Main
   482: # ---------------------------------------------------------------------------
   483: 
   484: def main():
   485:     parser = argparse.ArgumentParser()
   486:     parser.add_argument("--task", choices=["math", "humaneval", "text"],
   487:                         required=True)
   488:     parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
   489:     parser.add_argument("--steps", type=int, default=256)
   490:     parser.add_argument("--gen-length", type=int, default=256)
   491:     parser.add_argument("--block-length", type=int, default=64)
   492:     parser.add_argument("--conf-threshold", type=float, default=0.9)
   493:     parser.add_argument("--kl-threshold", type=float, default=0.01)
   494:     parser.add_argument("--history-length", type=int, default=2)
   495:     parser.add_argument("--temperature", type=float, default=0.0)
   496:     parser.add_argument("--seed", type=int, default=42)
   497:     parser.add_argument("--data-path", required=True)
   498:     parser.add_argument("--n-samples", type=int, default=0,
   499:                         help="0 = use all problems")
   500:     parser.add_argument("--prefix-len", type=int, default=32,
   501:                         help="Prefix length (text task)")
   502:     parser.add_argument("--output-dir", default=".")
   503:     args = parser.parse_args()
   504: 
   505:     torch.manual_seed(args.seed)
   506:     np.random.seed(args.seed)
   507:     device = "cuda" if torch.cuda.is_available() else "cpu"
   508: 
   509:     print(f"[INFO] Loading {args.model}...", flush=True)
   510:     raw_model, tokenizer, mask_id = load_instruct_model(args.model, device)
   511:     # The decoder and eval loops only see the counting handle.
   512:     model, forward_count = _count_denoiser_forwards(raw_model)
   513:     del raw_model
   514: 
   515:     decoder = DemaskDecoder(
   516:         mask_id=mask_id,
   517:         temperature=args.temperature,
   518:         conf_threshold=args.conf_threshold,
   519:         kl_threshold=args.kl_threshold,
   520:         history_length=args.history_length,
   521:     )
   522: 
   523:     print(f"[INFO] task={args.task} steps={args.steps} "
   524:           f"gen_length={args.gen_length} block_length={args.block_length}",
   525:           flush=True)
   526: 
   527:     if args.task == "math":
   528:         problems = load_math(args.data_path)
   529:         if args.n_samples > 0:
   530:             problems = problems[:args.n_samples]
   531:         acc, avg_steps = eval_math(
   532:             model, tokenizer, decoder, problems,
   533:             args.gen_length, args.steps, args.block_length, forward_count)
   534:         print(f"TEST_METRICS: accuracy={acc:.4f} avg_steps={avg_steps:.2f} "
   535:               f"n_samples={len(problems)}", flush=True)
   536:     elif args.task == "humaneval":
   537:         problems = load_humaneval(args.data_path)
   538:         if args.n_samples > 0:
   539:             problems = problems[:args.n_samples]
   540:         acc, avg_steps = eval_humaneval(
   541:             model, tokenizer, decoder, problems,
   542:             args.gen_length, args.steps, args.block_length, forward_count)
   543:         print(f"TEST_METRICS: accuracy={acc:.4f} avg_steps={avg_steps:.2f} "
   544:               f"n_samples={len(problems)}", flush=True)
   545:     else:  # text
   546:         with open(args.data_path) as f:
   547:             texts = json.load(f)
   548:         n = args.n_samples if args.n_samples > 0 else 256
   549:         ppl, mauve, ent, rep2, avg_steps = eval_text(
   550:             model, tokenizer, decoder, texts,
   551:             args.prefix_len, args.gen_length, args.steps, args.block_length,
   552:             n_samples=n, seed=args.seed, forward_count=forward_count)
   553:         print(f"TEST_METRICS: gen_ppl={ppl:.4f} mauve={mauve:.4f} "
   554:               f"entropy={ent:.4f} rep2={rep2:.4f} avg_steps={avg_steps:.2f} "
   555:               f"n_samples={n}", flush=True)
   556: 
   557: 
   558: if __name__ == "__main__":
   559:     main()
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `topk_margin` baseline — editable region  [READ-ONLY — reference implementation]

In `LLaDA/custom_demask_eval.py`:

```python
Lines 59–105:
    56: 
    57: 
    58: 
    59: class DemaskDecoder:
    60:     """topk_margin: unmask top-k positions by (top1_prob - top2_prob)."""
    61: 
    62:     def __init__(self, mask_id: int, temperature: float = 0.0,
    63:                  conf_threshold: float = 0.9, kl_threshold: float = 0.01,
    64:                  history_length: int = 2):
    65:         self.mask_id = mask_id
    66:         self.temperature = temperature
    67: 
    68:     @torch.no_grad()
    69:     def decode(self, model, input_ids, gen_length: int, steps: int,
    70:                block_length: int):
    71:         mid = self.mask_id
    72:         x = torch.full((1, input_ids.shape[1] + gen_length), mid,
    73:                        dtype=torch.long, device=model.device)
    74:         x[:, :input_ids.shape[1]] = input_ids.clone()
    75:         assert gen_length % block_length == 0
    76:         num_blocks = gen_length // block_length
    77:         assert steps % num_blocks == 0
    78:         steps_per_block = steps // num_blocks
    79:         used = 0
    80:         for b in range(num_blocks):
    81:             bs = input_ids.shape[1] + b * block_length
    82:             be = bs + block_length
    83:             num_xfer = get_num_transfer_tokens(
    84:                 (x[:, bs:be] == mid), steps_per_block)
    85:             for step in range(steps_per_block):
    86:                 mask_idx = (x == mid)
    87:                 block_m = torch.zeros_like(mask_idx)
    88:                 block_m[:, bs:be] = True
    89:                 mask_idx = mask_idx & block_m
    90:                 if not mask_idx.any():
    91:                     break
    92:                 logits = model(x).logits
    93:                 p_curr = F.softmax(logits.to(torch.float64), dim=-1)
    94:                 x0 = torch.argmax(p_curr, dim=-1)
    95:                 sorted_probs, _ = torch.sort(p_curr, dim=-1, descending=True)
    96:                 margin = sorted_probs[..., 0] - sorted_probs[..., 1]
    97:                 xfer = torch.zeros_like(x0, dtype=torch.bool)
    98:                 for j in range(margin.shape[0]):
    99:                     m = margin[j].clone()
   100:                     m[~mask_idx[j]] = -float("inf")
   101:                     _, topk = torch.topk(m, int(num_xfer[j, step].item()))
   102:                     xfer[j, topk] = True
   103:                 x = torch.where(xfer, x0, x)
   104:                 used += 1
   105:         return x, used
   106: 
   107: 
   108: # ====================================================================
```

### `confidence_greedy` baseline — editable region  [READ-ONLY — reference implementation]

In `LLaDA/custom_demask_eval.py`:

```python
Lines 59–104:
    56: 
    57: 
    58: 
    59: class DemaskDecoder:
    60:     """low_confidence remasking: unmask top-k positions by confidence."""
    61: 
    62:     def __init__(self, mask_id: int, temperature: float = 0.0,
    63:                  conf_threshold: float = 0.9, kl_threshold: float = 0.01,
    64:                  history_length: int = 2):
    65:         self.mask_id = mask_id
    66:         self.temperature = temperature
    67: 
    68:     @torch.no_grad()
    69:     def decode(self, model, input_ids, gen_length: int, steps: int,
    70:                block_length: int):
    71:         mid = self.mask_id
    72:         x = torch.full((1, input_ids.shape[1] + gen_length), mid,
    73:                        dtype=torch.long, device=model.device)
    74:         x[:, :input_ids.shape[1]] = input_ids.clone()
    75:         assert gen_length % block_length == 0
    76:         num_blocks = gen_length // block_length
    77:         assert steps % num_blocks == 0
    78:         steps_per_block = steps // num_blocks
    79:         used = 0
    80:         for b in range(num_blocks):
    81:             bs = input_ids.shape[1] + b * block_length
    82:             be = bs + block_length
    83:             num_xfer = get_num_transfer_tokens(
    84:                 (x[:, bs:be] == mid), steps_per_block)
    85:             for step in range(steps_per_block):
    86:                 mask_idx = (x == mid)
    87:                 block_m = torch.zeros_like(mask_idx)
    88:                 block_m[:, bs:be] = True
    89:                 mask_idx = mask_idx & block_m
    90:                 if not mask_idx.any():
    91:                     break
    92:                 logits = model(x).logits
    93:                 p_curr = F.softmax(logits.to(torch.float64), dim=-1)
    94:                 x0 = torch.argmax(p_curr, dim=-1)
    95:                 conf = torch.gather(p_curr, -1, x0.unsqueeze(-1)).squeeze(-1)
    96:                 xfer = torch.zeros_like(x0, dtype=torch.bool)
    97:                 for j in range(conf.shape[0]):
    98:                     c = conf[j].clone()
    99:                     c[~mask_idx[j]] = -float("inf")
   100:                     _, topk = torch.topk(c, int(num_xfer[j, step].item()))
   101:                     xfer[j, topk] = True
   102:                 x = torch.where(xfer, x0, x)
   103:                 used += 1
   104:         return x, used
   105: 
   106: 
   107: # ====================================================================
```

### `klass` baseline — editable region  [READ-ONLY — reference implementation]

In `LLaDA/custom_demask_eval.py`:

```python
Lines 59–128:
    56: 
    57: 
    58: 
    59: class DemaskDecoder:
    60:     """KLASS: stability + confidence, KL-adaptive (Kim et al., NeurIPS 2025)."""
    61: 
    62:     def __init__(self, mask_id: int, temperature: float = 0.0,
    63:                  conf_threshold: float = 0.9, kl_threshold: float = 0.01,
    64:                  history_length: int = 2):
    65:         self.mask_id = mask_id
    66:         self.temperature = temperature
    67:         self.conf_threshold = conf_threshold
    68:         self.kl_threshold = kl_threshold
    69:         self.history_length = history_length
    70: 
    71:     @torch.no_grad()
    72:     def decode(self, model, input_ids, gen_length: int, steps: int,
    73:                block_length: int):
    74:         mid = self.mask_id
    75:         x = torch.full((1, input_ids.shape[1] + gen_length), mid,
    76:                        dtype=torch.long, device=model.device)
    77:         x[:, :input_ids.shape[1]] = input_ids.clone()
    78:         assert gen_length % block_length == 0
    79:         num_blocks = gen_length // block_length
    80:         assert steps % num_blocks == 0
    81:         steps_per_block = steps // num_blocks
    82:         V = model.lm_head.out_features if hasattr(model, "lm_head") \
    83:                                        else model.config.vocab_size
    84:         kl_hist = torch.zeros((1, x.shape[1], self.history_length),
    85:                               dtype=torch.float64, device=x.device)
    86:         p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64,
    87:                              device=x.device)
    88:         used = 0
    89:         for b in range(num_blocks):
    90:             bs = input_ids.shape[1] + b * block_length
    91:             be = bs + block_length
    92:             num_xfer = get_num_transfer_tokens(
    93:                 (x[:, bs:be] == mid), steps_per_block)
    94:             for step in range(steps_per_block):
    95:                 mask_idx = (x == mid)
    96:                 block_m = torch.zeros_like(mask_idx)
    97:                 block_m[:, bs:be] = True
    98:                 mask_idx = mask_idx & block_m
    99:                 if not mask_idx.any():
   100:                     break
   101:                 logits = model(x).logits
   102:                 p_curr = F.softmax(logits.to(torch.float64), dim=-1)
   103:                 x0 = torch.argmax(p_curr, dim=-1)
   104:                 conf = torch.gather(p_curr, -1, x0.unsqueeze(-1)).squeeze(-1)
   105:                 eps = 1e-12
   106:                 kl = (p_curr * (torch.log(p_curr + eps)
   107:                                 - torch.log(p_prev + eps))).sum(-1)
   108:                 kl_hist = torch.roll(kl_hist, -1, dims=-1)
   109:                 kl_hist[..., -1] = kl
   110:                 p_prev = p_curr.clone()
   111:                 if step >= self.history_length - 1:
   112:                     stable = torch.all(kl_hist < self.kl_threshold, dim=-1)
   113:                 else:
   114:                     stable = torch.zeros_like(conf, dtype=torch.bool)
   115:                 ready = stable & (conf > self.conf_threshold) & mask_idx
   116:                 xfer = torch.zeros_like(x0, dtype=torch.bool)
   117:                 for j in range(ready.shape[0]):
   118:                     rdy = torch.where(ready[j])[0]
   119:                     if len(rdy) > 0:
   120:                         xfer[j, rdy] = True
   121:                     else:
   122:                         c = conf[j].clone()
   123:                         c[~mask_idx[j]] = -float("inf")
   124:                         _, topk = torch.topk(c, int(num_xfer[j, step].item()))
   125:                         xfer[j, topk] = True
   126:                 x = torch.where(xfer, x0, x)
   127:                 used += 1
   128:         return x, used
   129: 
   130: 
   131: # ====================================================================
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
