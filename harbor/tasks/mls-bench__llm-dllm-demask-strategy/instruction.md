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
- Pretrained models (LLaDA-8B-Instruct, Dream-v0-Instruct-7B), prompts, evaluation data, and task runners are fixed.
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

`get_num_transfer_tokens(mask, steps)` is available outside the editable region — it returns the uniform schedule (`mask.sum() // steps` per step). Always return shape `[1, prompt_len + gen_length]`. `used_steps` counts model forward passes (lower = more efficient).

## Reference baseline strategies
- `confidence_greedy` — LLaDA's `low_confidence` remasking: top-k by max prob.
- `topk_margin` — Dream's `topk_margin`: top-k by (top1 prob − top2 prob).
- `klass` — KLASS: KL-adaptive stability + confidence thresholds (KLASS paper, default `kl_threshold=0.01`, `conf_threshold=0.9`, `history_length=2`).

## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/LLaDA/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

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
   160: # Data loading
   161: # ---------------------------------------------------------------------------
   162: 
   163: def load_math(path: str) -> list[dict]:
   164:     with open(path) as f:
   165:         return [json.loads(line) for line in f if line.strip()]
   166: 
   167: 
   168: def load_humaneval(path: str) -> list[dict]:
   169:     opener = gzip.open if path.endswith(".gz") else open
   170:     with opener(path, "rt") as f:
   171:         return [json.loads(line) for line in f if line.strip()]
   172: 
   173: 
   174: # ---------------------------------------------------------------------------
   175: # MATH evaluation (uses klass_utils extract_math_answer + compare_answers)
   176: # ---------------------------------------------------------------------------
   177: 
   178: def _import_klass_utils():
   179:     """Import klass_utils from task data dir (mounted at /workspace/_task)."""
   180:     task_dir = os.environ.get("TASK_DIR", "/workspace/_task")
   181:     sys.path.insert(0, os.path.join(task_dir, "data"))
   182:     import klass_utils as ku
   183:     return ku
   184: 
   185: 
   186: def eval_math(model, tokenizer, decoder: DemaskDecoder, problems: list[dict],
   187:               gen_length: int, steps: int, block_length: int):
   188:     ku = _import_klass_utils()
   189:     sys_msg = ("Your task is to answer the question below. Give step by step "
   190:                "reasoning before you answer, and when you're ready to answer, "
   191:                "please use the format 'The final answer is'.")
   192:     correct = 0
   193:     total_steps = 0
   194:     for i, ex in enumerate(problems):
   195:         msgs = [{"role": "system", "content": sys_msg},
   196:                 {"role": "user", "content": ex["problem"]}]
   197:         prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
   198:                                                tokenize=False)
   199:         input_ids = torch.tensor(tokenizer(prompt)["input_ids"],
   200:                                  device=model.device).unsqueeze(0)
   201:         gt = ku.extract_math_answer(ex["problem"], ex["solution"])
   202:         x_out, used = decoder.decode(model, input_ids, gen_length, steps,
   203:                                      block_length)
   204:         gen_text = tokenizer.batch_decode(
   205:             x_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
   206:         pred = ku.extract_math_answer(ex["problem"], gen_text)
   207:         is_correct = ku.compare_answers(ex["problem"], gt, pred)
   208:         if i < 2:
   209:             print(f"[DEBUG] math example {i}:\n"
   210:                   f"  problem: {ex['problem'][:150]}\n"
   211:                   f"  gt={gt}\n"
   212:                   f"  gen (first 400 chars): {gen_text[:400]}\n"
   213:                   f"  pred={pred} correct={is_correct}", flush=True)
   214:         if is_correct:
   215:             correct += 1
   216:         total_steps += used
   217:         if (i + 1) % 10 == 0:
   218:             print(f"TRAIN_METRICS: math {i+1}/{len(problems)} "
   219:                   f"acc={correct/(i+1):.3f} "
   220:                   f"avg_steps={total_steps/(i+1):.1f}", flush=True)
   221:     return correct / max(len(problems), 1), total_steps / max(len(problems), 1)
   222: 
   223: 
   224: # ---------------------------------------------------------------------------
   225: # HumanEval evaluation (uses klass_utils evaluate_task)
   226: # ---------------------------------------------------------------------------
   227: 
   228: def _run_humaneval(code: str, test: str, entry_point: str) -> bool:
   229:     """Exec code + test + check(entry_point) in fresh namespace."""
   230:     try:
   231:         ns: dict = {}
   232:         exec(code + "\n" + test + f"\ncheck({entry_point})\n", ns)
   233:         return True
   234:     except Exception:
   235:         return False
   236: 
   237: 
   238: def check_humaneval_code(code: str, problem: dict, timeout: float = 3.0) -> bool:
   239:     import multiprocessing
   240:     entry = problem["entry_point"]
   241:     # If generated code lacks the function def, prepend problem prompt
   242:     # (which provides function signature + docstring).
   243:     if f"def {entry}" not in code:
   244:         code = problem["prompt"] + code
   245:     with multiprocessing.Pool(processes=1) as pool:
   246:         res = pool.apply_async(_run_humaneval, (code, problem["test"], entry))
   247:         try:
   248:             return bool(res.get(timeout=timeout))
   249:         except Exception:
   250:             return False
   251: 
   252: 
   253: def eval_humaneval(model, tokenizer, decoder: DemaskDecoder,
   254:                    problems: list[dict], gen_length: int, steps: int,
   255:                    block_length: int):
   256:     passed = 0
   257:     total_steps = 0
   258:     for i, p in enumerate(problems):
   259:         msgs = [{"role": "system", "content": "You complete only Python code."},
   260:                 {"role": "user", "content": p["prompt"]}]
   261:         prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
   262:                                                tokenize=False)
   263:         input_ids = torch.tensor(tokenizer(prompt)["input_ids"],
   264:                                  device=model.device).unsqueeze(0)
   265:         x_out, used = decoder.decode(model, input_ids, gen_length, steps,
   266:                                      block_length)
   267:         gen_text = tokenizer.batch_decode(
   268:             x_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
   269:         eos = tokenizer.eos_token or ""
   270:         if eos:
   271:             gen_text = gen_text.split(eos)[0]
   272:         m = re.search(r"```(?:python)?\n(.*?)(?:```|$)", gen_text, re.DOTALL)
   273:         code = m.group(1).strip() if m else gen_text.strip()
   274:         if i < 2:
   275:             print(f"[DEBUG] humaneval {p['entry_point']}:\n"
   276:                   f"gen (first 300 chars): {gen_text[:300]}\n"
   277:                   f"code (first 200 chars): {code[:200]}", flush=True)
   278:         ok = check_humaneval_code(code, p, timeout=3)
   279:         if ok:
   280:             passed += 1
   281:         total_steps += used
   282:         if (i + 1) % 10 == 0:
   283:             print(f"TRAIN_METRICS: humaneval {i+1}/{len(problems)} "
   284:                   f"pass@1={passed/(i+1):.3f} "
   285:                   f"avg_steps={total_steps/(i+1):.1f}", flush=True)
   286:     return passed / max(len(problems), 1), total_steps / max(len(problems), 1)
   287: 
   288: 
   289: # ---------------------------------------------------------------------------
   290: # Open-ended text generation evaluation (gen_ppl, MAUVE, entropy, rep2)
   291: # ---------------------------------------------------------------------------
   292: 
   293: def _truncate_at_eos(text: str, eos_tokens=("</s>", "<|endoftext|>", "<|im_end|>")):
   294:     for eos in eos_tokens:
   295:         idx = text.find(eos)
   296:         if idx >= 0:
   297:             text = text[:idx]
   298:     return text.strip()
   299: 
   300: 
   301: def compute_conditional_gen_ppl(prefix_texts, gen_texts, device):
   302:     from transformers import AutoModelForCausalLM, AutoTokenizer
   303:     import math as _m
   304:     tok = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
   305:     mdl = AutoModelForCausalLM.from_pretrained(
   306:         "openai-community/gpt2-large").to(device).eval()
   307:     total_loss, total_tokens = 0.0, 0
   308:     for prefix, gen in zip(prefix_texts, gen_texts):
   309:         if not gen.strip():
   310:             continue
   311:         p_ids = tok.encode(prefix, add_special_tokens=False)
   312:         g_ids = tok.encode(gen, add_special_tokens=False)
   313:         all_ids = (p_ids + g_ids)[:1024]
   314:         if len(p_ids) >= len(all_ids):
   315:             continue
   316:         ids = torch.tensor([all_ids], device=device)
   317:         with torch.no_grad():
   318:             logits = mdl(ids).logits[:, :-1, :]
   319:         labels = ids[:, 1:]
   320:         start = max(len(p_ids) - 1, 0)
   321:         loss = F.cross_entropy(
   322:             logits[:, start:, :].reshape(-1, logits.shape[-1]),
   323:             labels[:, start:].reshape(-1), reduction="sum")
   324:         total_loss += loss.item()
   325:         total_tokens += labels[:, start:].numel()
   326:     del mdl
   327:     torch.cuda.empty_cache()
   328:     return _m.exp(total_loss / total_tokens) if total_tokens else float("inf")
   329: 
   330: 
   331: def compute_mauve(gen_texts, ref_texts):
   332:     try:
   333:         import mauve
   334:         r = mauve.compute_mauve(p_text=ref_texts, q_text=gen_texts,
   335:                                 device_id=0 if torch.cuda.is_available() else -1,
   336:                                 max_text_length=512, verbose=False,
   337:                                 featurize_model_name="openai-community/gpt2-large")
   338:         return float(r.mauve)
   339:     except Exception as e:
   340:         print(f"[WARN] MAUVE failed: {e}", flush=True)
   341:         return 0.0
   342: 
   343: 
   344: def compute_entropy_rep2(texts):
   345:     from collections import Counter
   346:     import math as _m
   347:     all_bigrams = []
   348:     rep_ratios = []
   349:     for t in texts:
   350:         words = t.split()
   351:         bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
   352:         all_bigrams.extend(bigrams)
   353:         if bigrams:
   354:             rep_ratios.append(1.0 - len(set(bigrams)) / len(bigrams))
   355:         else:
   356:             rep_ratios.append(0.0)
   357:     ent = 0.0
   358:     if all_bigrams:
   359:         c = Counter(all_bigrams)
   360:         tot = sum(c.values())
   361:         for v in c.values():
   362:             p = v / tot
   363:             if p > 0:
   364:                 ent -= p * _m.log2(p)
   365:     rep2 = sum(rep_ratios) / max(len(rep_ratios), 1)
   366:     return ent, rep2
   367: 
   368: 
   369: def eval_text(model, tokenizer, decoder: DemaskDecoder, raw_texts: list[str],
   370:               prefix_len: int, gen_length: int, steps: int, block_length: int,
   371:               n_samples: int, seed: int):
   372:     """Prefix-conditioned C4 continuation. Reports gen_ppl/MAUVE/entropy/rep2."""
   373:     import random as _r
   374:     rng = _r.Random(seed)
   375:     if len(raw_texts) > n_samples:
   376:         raw_texts = rng.sample(raw_texts, n_samples)
   377: 
   378:     # Build prefix prompts (raw, no chat template — Dream-Instruct as a base LM)
   379:     prefix_ids_list, prefix_texts, valid_refs = [], [], []
   380:     for txt in raw_texts:
   381:         ids = tokenizer.encode(txt, add_special_tokens=False)
   382:         if len(ids) >= prefix_len + gen_length:
   383:             pids = ids[:prefix_len]
   384:             prefix_ids_list.append(pids)
   385:             prefix_texts.append(tokenizer.decode(pids, skip_special_tokens=True))
   386:             ref_ids = ids[prefix_len:prefix_len + gen_length]
   387:             valid_refs.append(tokenizer.decode(ref_ids, skip_special_tokens=True))
   388:     print(f"[INFO] kept {len(prefix_ids_list)}/{len(raw_texts)} texts "
   389:           f"long enough for prefix={prefix_len}+gen={gen_length}", flush=True)
   390: 
   391:     gen_texts, total_used = [], 0
   392:     for i, pids in enumerate(prefix_ids_list):
   393:         ids = torch.tensor([pids], dtype=torch.long, device=model.device)
   394:         x_out, used = decoder.decode(model, ids, gen_length, steps, block_length)
   395:         gen = tokenizer.decode(x_out[0, ids.shape[1]:].tolist(),
   396:                                skip_special_tokens=True)
   397:         gen = _truncate_at_eos(gen)
   398:         gen_texts.append(gen)
   399:         total_used += used
   400:         if (i + 1) % 10 == 0:
   401:             print(f"TRAIN_METRICS: text {i+1}/{len(prefix_ids_list)} "
   402:                   f"avg_steps={total_used/(i+1):.1f}", flush=True)
   403: 
   404:     avg_steps = total_used / max(len(prefix_ids_list), 1)
   405:     print("[INFO] unloading gen model, computing GPT-2 ppl...", flush=True)
   406:     del model
   407:     torch.cuda.empty_cache()
   408: 
   409:     ppl = compute_conditional_gen_ppl(prefix_texts, gen_texts, "cuda")
   410:     mauve = compute_mauve(gen_texts, valid_refs)
   411:     entropy, rep2 = compute_entropy_rep2(gen_texts)
   412:     return ppl, mauve, entropy, rep2, avg_steps
   413: 
   414: 
   415: # ---------------------------------------------------------------------------
   416: # Main
   417: # ---------------------------------------------------------------------------
   418: 
   419: def main():
   420:     parser = argparse.ArgumentParser()
   421:     parser.add_argument("--task", choices=["math", "humaneval", "text"],
   422:                         required=True)
   423:     parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
   424:     parser.add_argument("--steps", type=int, default=256)
   425:     parser.add_argument("--gen-length", type=int, default=256)
   426:     parser.add_argument("--block-length", type=int, default=64)
   427:     parser.add_argument("--conf-threshold", type=float, default=0.9)
   428:     parser.add_argument("--kl-threshold", type=float, default=0.01)
   429:     parser.add_argument("--history-length", type=int, default=2)
   430:     parser.add_argument("--temperature", type=float, default=0.0)
   431:     parser.add_argument("--seed", type=int, default=42)
   432:     parser.add_argument("--data-path", required=True)
   433:     parser.add_argument("--n-samples", type=int, default=0,
   434:                         help="0 = use all problems")
   435:     parser.add_argument("--prefix-len", type=int, default=32,
   436:                         help="Prefix length (text task)")
   437:     parser.add_argument("--output-dir", default=".")
   438:     args = parser.parse_args()
   439: 
   440:     torch.manual_seed(args.seed)
   441:     np.random.seed(args.seed)
   442:     device = "cuda" if torch.cuda.is_available() else "cpu"
   443: 
   444:     print(f"[INFO] Loading {args.model}...", flush=True)
   445:     model, tokenizer, mask_id = load_instruct_model(args.model, device)
   446: 
   447:     decoder = DemaskDecoder(
   448:         mask_id=mask_id,
   449:         temperature=args.temperature,
   450:         conf_threshold=args.conf_threshold,
   451:         kl_threshold=args.kl_threshold,
   452:         history_length=args.history_length,
   453:     )
   454: 
   455:     print(f"[INFO] task={args.task} steps={args.steps} "
   456:           f"gen_length={args.gen_length} block_length={args.block_length}",
   457:           flush=True)
   458: 
   459:     if args.task == "math":
   460:         problems = load_math(args.data_path)
   461:         if args.n_samples > 0:
   462:             problems = problems[:args.n_samples]
   463:         acc, avg_steps = eval_math(
   464:             model, tokenizer, decoder, problems,
   465:             args.gen_length, args.steps, args.block_length)
   466:         print(f"TEST_METRICS: accuracy={acc:.4f} avg_steps={avg_steps:.2f} "
   467:               f"n_samples={len(problems)}", flush=True)
   468:     elif args.task == "humaneval":
   469:         problems = load_humaneval(args.data_path)
   470:         if args.n_samples > 0:
   471:             problems = problems[:args.n_samples]
   472:         acc, avg_steps = eval_humaneval(
   473:             model, tokenizer, decoder, problems,
   474:             args.gen_length, args.steps, args.block_length)
   475:         print(f"TEST_METRICS: accuracy={acc:.4f} avg_steps={avg_steps:.2f} "
   476:               f"n_samples={len(problems)}", flush=True)
   477:     else:  # text
   478:         with open(args.data_path) as f:
   479:             texts = json.load(f)
   480:         n = args.n_samples if args.n_samples > 0 else 256
   481:         ppl, mauve, ent, rep2, avg_steps = eval_text(
   482:             model, tokenizer, decoder, texts,
   483:             args.prefix_len, args.gen_length, args.steps, args.block_length,
   484:             n_samples=n, seed=args.seed)
   485:         print(f"TEST_METRICS: gen_ppl={ppl:.4f} mauve={mauve:.4f} "
   486:               f"entropy={ent:.4f} rep2={rep2:.4f} avg_steps={avg_steps:.2f} "
   487:               f"n_samples={n}", flush=True)
   488: 
   489: 
   490: if __name__ == "__main__":
   491:     main()
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

Good luck.
