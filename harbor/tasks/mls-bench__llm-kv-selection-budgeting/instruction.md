# MLS-Bench: llm-kv-selection-budgeting

# LLM KV Cache: Selection Budgeting

## Research Question

Design a KV token-retention controller inside a shared full-attention
Hugging Face decoding harness. The task asks how a token-scoring rule
should rank prefill KV entries so that the model can decode from a small
fixed-budget subset of those entries without losing long-context quality.

## Background

Long-context LLM inference stores key-value (KV) tensors for every
attention layer. KV selection methods reduce this cache by retaining a
subset of historical tokens while preserving enough context for generation
quality. The shared hook in this task exposes the union of behaviors
needed by three reference families:

- StreamingLLM (Xiao et al., ICLR 2024; arXiv:2309.17453) preserves a
  small set of attention sinks plus a recent window, with optional
  RoPE-aware key rerotation so the rolling window remains positionally
  consistent.
- Expected Attention (Devoto, Jeblick, Jegou, 2025; arXiv:2510.00636)
  estimates how future queries will attend to each prefill KV pair, in
  closed form from observed distributional statistics, and prunes
  low-expected-attention pairs.
- LagKV (Liang et al., 2025; arXiv:2504.04704) is an attention-free
  scoring rule that uses subsequent (lag) tokens to normalize and score
  earlier KV entries via a hardware-friendly token/channel statistic.

Reference defaults follow each paper's experimental section and are
mirrored in the NVIDIA `kvpress` library.

## Task

Modify only the `SelectionPolicy` class in
`transformers-kv-lab/custom_selection_eval.py`. Implement:

- `retention_plan(layer_id, request_meta, cache_meta)`
- `score_tokens(module, hidden_states, keys, values, kwargs, plan)`
- `select_cache(module, keys, values, scores, n_kept)`

The harness owns the model, datasets, prompt templates, cache budget,
decode loop, and scoring. The editable policy owns the retention metadata
and the per-token scoring rule used to rank prefill KV entries. The shared
hook exposes the union needed by the included source-backed baselines:

- no compression
- attention-sink/recent-window retention
- expected future-attention scoring
- LagKV lag-relative key/value scoring
- optional RoPE-aware key rerotation after pruning

The canonical compression setting is `SELECTION_KV_COMPRESSION_RATIO=0.8`,
meaning methods retain roughly 20% of prefill KV tokens unless the
`full_attention` anchor explicitly disables compression. Final scoring
applies a soft over-budget penalty once `mean_retained_fraction > 0.25`.

## Evaluation

The canonical model is aligned with `llm-kv-adaptive-quantization`:
`Qwen/Qwen2.5-3B-Instruct`. Workloads share the same public
text-benchmark protocol used by `llm-kv-adaptive-quantization`. Dataset
sources, split names, prompt templates, generation limits, and scoring
semantics stay aligned across those tasks whenever the same workload label
is used.

| Label | Source | Final score |
|---|---|---|
| `longbench_hotpotqa` | LongBench-E `hotpotqa_e` | LongBench QA F1 (0-100) |
| `longbench_passage_retrieval` | LongBench-E `passage_retrieval_en_e` | LongBench retrieval (0-100) |
| `longbench_repobench` | LongBench-E `repobench-p_e` | LongBench code similarity (0-100) |
| `longbench_v2` | LongBench v2 `train` split | multiple-choice exact accuracy (0-100), official head-tail truncation when over context |
| `gsm8k` | `openai/gsm8k` main test split | exact final-answer accuracy after numeric normalization (0-100) |

Canonical data sources are the public upstream datasets listed above:
`THUDM/LongBench`, `THUDM/LongBench-v2` / `zai-org/LongBench-v2`, and
`openai/gsm8k`. The runtime resolves model and datasets from the local
build-time or mounted Hugging Face cache only; missing assets are hard
failures rather than implicit network downloads.

`SELECTION_KV_MAX_EXAMPLES=0` means full available workload. Setting it to
a positive integer is allowed for local smoke validation but should not be
used for final leaderboard evidence.

## Baselines

Each baseline cites its paper and identifies the canonical reference
implementation it mirrors on the shared hook surface.

- `full_attention`: uncompressed full-cache reference anchor (HuggingFace
  `DynamicCache`).
- `streamingllm`: Xiao et al. (ICLR 2024; arXiv:2309.17453), "Efficient
  Streaming Language Models with Attention Sinks". Section 3.2 + Table 2
  defaults: `sink_tokens = 4`, recent-window inferred from the budget.
  RoPE-aware key rerotation following the paper's "Rolling KV Cache with
  Attention Sinks" formulation; reference re-rotation routine mirrors
  NVIDIA/kvpress `KeyRerotationPress` (audit commit 0.3.0). Original
  code: github.com/mit-han-lab/streaming-llm.
- `expected_attention`: Devoto, Jeblick, Jegou (NVIDIA), 2025
  (arXiv:2510.00636), "Expected Attention: KV Cache Compression by
  Estimating Attention from Future Queries Distribution". Equations 4-7
  for the score; defaults from the paper's experimental section
  (matched in NVIDIA/kvpress `ExpectedAttentionPress`, audit 0.3.0):
  `n_future_positions=512`, `n_sink=4`, covariance enabled, value-norm
  rescaling enabled.
- `lagkv`: Liang et al., 2025 (arXiv:2504.04704), "LagKV: Lag-Relative
  Information of the KV Cache Tells Which Tokens Are Important".
  Algorithm 1 (Section 3.2) defines the lag-relative score; defaults from
  paper Section 4.1 / Table 1 (mirrored in NVIDIA/kvpress `LagKVPress`,
  audit 0.3.0): `n_sink=4`, `lag_size=128`, `cross_scoring=False`.

### Harness enforcement vs. advisory metadata

The `retention_plan(...)` dict serves as the policy's internal
communication channel between `retention_plan` and `score_tokens`. The
harness enforces only what is observable from the post-`select_cache`
cache state:

| Field | Status | Notes |
|---|---|---|
| `compression_ratio` | enforced | Harness force-overrides to its own value at the call site (`PrefillSelectionCompressor.forward_hook`). Policies cannot lie about the budget. |
| `mean_retained_fraction` | measured, enforced | Computed from `select_cache`'s actual output `n_kept / keys.shape[2]` per layer, then averaged. Drives the soft budget penalty in `score_spec.py`. |
| `disable_compression` | enforced | If `True`, harness skips `score_tokens`/`select_cache` entirely and reports `retained = 1.0`. Used by the `full_attention` anchor. |
| `method` | logged only | Recorded for provenance; not used in scoring. |
| `sink_tokens`, `lag_size`, `n_future_positions`, `subspace_dim`, etc. | advisory | Used internally by the policy's own `score_tokens`. The harness does not verify that declared "sinks" are actually preserved by `select_cache`'s top-K output. Honesty here only matters for provenance and ablation reproducibility, not for scoring. |

Final scoring depends only on the end-to-end measured signals
(`final_score`, `mean_retained_fraction`, `runtime_seconds`).

## Metrics

The parser expects one `TEST_METRICS:` line per workload with:

- `final_score`: benchmark-native final task score on a 0-100 scale
- `mean_retained_fraction`: average retained prefill KV fraction after the
  policy runs
- `runtime_seconds`: workload wall-clock runtime in seconds

## Canonical Ranking

The leaderboard uses a single scalar computed from accuracy, runtime, and
cache reduction under the fixed retained-fraction constraint. Each workload
combines three normalized terms with weights `accuracy:time:reduction = 6:2:2`:

- `accuracy_score`: bounded 0-100 quality normalization calibrated against
  the visible baseline envelope
- `time_score`: soft lower-is-better sigmoid normalization of
  `runtime_seconds`, calibrated from the visible baseline runtime envelope
- `reduction_score`: bounded lower-is-better normalization of
  `mean_retained_fraction`

The per-workload score is the weighted mean of those three terms, and the
task score is the geometric mean across workloads. Rows whose
`mean_retained_fraction_*` exceeds the fixed budget tolerance receive a
soft upper-bound penalty, so the `full_attention` row remains a visible
reference anchor rather than a valid compressed-cache submission.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/transformers-kv-lab/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to score zero.

- `transformers-kv-lab/custom_selection_eval.py`
- editable lines **40–101**




## Readable Context


### `transformers-kv-lab/custom_selection_eval.py`  [EDITABLE — lines 40–101 only]

```python
     1: """Full-attention KV selection replay harness.
     2: 
     3: The scaffold owns the model, data, decode loop, and fixed cache budget. Policies
     4: only describe how to score already-computed KV tokens after a standard
     5: full-attention prefill. The selected tokens are then used for greedy decoding on
     6: the same public text workloads as llm-kv-adaptive-quantization.
     7: """
     8: 
     9: from __future__ import annotations
    10: 
    11: import argparse
    12: import contextlib
    13: import difflib
    14: import json
    15: import math
    16: import os
    17: import re
    18: import string
    19: import time
    20: import zipfile
    21: from collections import Counter
    22: from dataclasses import dataclass
    23: from pathlib import Path
    24: from statistics import mean
    25: 
    26: import torch
    27: from torch import nn
    28: from torch.nn import functional as F
    29: from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, set_seed
    30: 
    31: 
    32: @dataclass
    33: class WorkloadSpec:
    34:     name: str
    35:     source_family: str
    36:     examples: list[dict]
    37:     max_new_tokens: int
    38: 
    39: 
    40: class SelectionPolicy:
    41:     """Editable semantic hook for KV token retention after full-attention prefill."""
    42: 
    43:     method_name = "streamingllm"
    44:     rerotate_selected_keys = True
    45: 
    46:     def retention_plan(self, layer_id, request_meta, cache_meta):
    47:         return {
    48:             "method": self.method_name,
    49:             "sink_tokens": 4,
    50:             "compression_ratio": cache_meta["compression_ratio"],
    51:         }
    52: 
    53:     def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
    54:         k_len = int(keys.shape[2])
    55:         n_sink = int(plan.get("sink_tokens", 4))
    56:         ratio = float(plan["compression_ratio"])
    57:         assert k_len > n_sink, f"Input should contain more tokens than sink_tokens={n_sink}"
    58:         n_pruned = k_len - int(k_len * (1.0 - ratio))
    59:         scores = torch.ones_like(keys[..., 0])
    60:         scores[:, :, n_sink : n_sink + n_pruned] = 0
    61:         return scores
    62: 
    63:     def rotate_half(self, x):
    64:         x1 = x[..., : x.shape[-1] // 2]
    65:         x2 = x[..., x.shape[-1] // 2 :]
    66:         return torch.cat((-x2, x1), dim=-1)
    67: 
    68:     def rerotate_cache_keys(self, module, indices, keys):
    69:         bsz, num_key_value_heads, n_kept = indices.shape
    70:         device = indices.device
    71:         device_type = keys.device.type
    72:         dtype = keys.dtype
    73:         inv_freq = module.rotary_emb.inv_freq[None, None, :, None].float().expand(
    74:             bsz, num_key_value_heads, -1, 1
    75:         )
    76:         new_positions = torch.arange(0, n_kept, device=device).unsqueeze(0)[:, None, :].float()
    77:         new_positions = new_positions.expand(bsz, num_key_value_heads, n_kept)
    78:         delta_pos = (new_positions - indices.float()).unsqueeze(2)
    79:         device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    80:         with torch.autocast(device_type=device_type, enabled=False):
    81:             freqs = (delta_pos.float() * inv_freq.float()).transpose(2, 3)
    82:             emb = torch.cat((freqs, freqs), dim=-1)
    83:             cos = emb.cos().contiguous()
    84:             sin = emb.sin().contiguous()
    85:         cos = cos.to(dtype=dtype)
    86:         sin = sin.to(dtype=dtype)
    87:         gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
    88:         gathered = keys.gather(2, gather_idx).contiguous()
    89:         return (gathered * cos) + (self.rotate_half(gathered) * sin)
    90: 
    91:     def select_cache(self, module, keys, values, scores, n_kept):
    92:         indices = scores.topk(n_kept, dim=-1).indices
    93:         if self.rerotate_selected_keys:
    94:             indices = torch.sort(indices, dim=2).values
    95:             selected_keys = self.rerotate_cache_keys(module, indices, keys)
    96:         else:
    97:             gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
    98:             selected_keys = keys.gather(2, gather_idx).contiguous()
    99:         gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
   100:         selected_values = values.gather(2, gather_idx).contiguous()
   101:         return selected_keys, selected_values
   102: 
   103: 
   104: def resolve_task_dir() -> Path:
   105:     here = Path(__file__).resolve().parent
   106:     candidates = [
   107:         here,
   108:         here.parent,
   109:         Path(os.environ.get("MLSBENCH_TASK_DIR", "")),
   110:         Path(os.environ.get("TASK_DIR", "")),
   111:         Path.cwd() / "_task",
   112:         Path.cwd().parent / "_task",
   113:     ]
   114:     seen = set()
   115:     for candidate in candidates:
   116:         if not candidate or str(candidate) == ".":
   117:             continue
   118:         resolved = candidate.resolve()
   119:         if resolved in seen:
   120:             continue
   121:         seen.add(resolved)
   122:         if (resolved / "task_description.md").exists():
   123:             return resolved
   124:     raise FileNotFoundError(f"Unable to locate task directory from {here}")
   125: 
   126: 
   127: TASK_DIR = resolve_task_dir()
   128: DEFAULT_MAX_EXAMPLES = int(os.environ.get("SELECTION_KV_MAX_EXAMPLES", "0"))
   129: DEFAULT_MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
   130: DEFAULT_COMPRESSION_RATIO = float(os.environ.get("SELECTION_KV_COMPRESSION_RATIO", "0.8"))
   131: DEFAULT_MAX_PROMPT_TOKENS = int(os.environ.get("SELECTION_KV_MAX_PROMPT_TOKENS", "0"))
   132: 
   133: LONG_BENCH_TEMPLATES = {
   134:     "hotpotqa_e": (
   135:         "Answer the question based on the given passages. "
   136:         "Only give me the answer and do not output any other words.\n\n"
   137:         "The following are given passages.\n{context}\n\n"
   138:         "Question: {input}\nAnswer:"
   139:     ),
   140:     "passage_retrieval_en_e": (
   141:         "Here are 30 paragraphs from Wikipedia, along with an abstract. "
   142:         "Please determine which paragraph the abstract is from.\n\n"
   143:         "{context}\n\n"
   144:         "The following is an abstract.\n\n{input}\n\n"
   145:         "Please enter the number of the paragraph that the abstract is from. "
   146:         "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
   147:         "The answer is: "
   148:     ),
   149:     "repobench-p_e": "Please complete the code given below.\n{context}{input}Next line of code:\n",
   150: }
   151: 
   152: LONG_BENCH_V2_TEMPLATE = (
   153:     "Please read the following text and answer the question below.\n\n"
   154:     "{context}\n\n"
   155:     "What is the correct answer to this question: {question}\n"
   156:     "Choices:\n"
   157:     "(A) {choice_A}\n"
   158:     "(B) {choice_B}\n"
   159:     "(C) {choice_C}\n"
   160:     "(D) {choice_D}\n"
   161:     "Format your response as follows: \"The correct answer is (insert answer here)\"."
   162: )
   163: 
   164: WORKLOAD_CONFIGS = {
   165:     "longbench_hotpotqa": {
   166:         "source_family": "THUDM/LongBench hotpotqa_e",
   167:         "dataset_name": "hotpotqa_e",
   168:         "max_new_tokens": 32,
   169:     },
   170:     "longbench_passage_retrieval": {
   171:         "source_family": "THUDM/LongBench passage_retrieval_en_e",
   172:         "dataset_name": "passage_retrieval_en_e",
   173:         "max_new_tokens": 32,
   174:     },
   175:     "longbench_repobench": {
   176:         "source_family": "THUDM/LongBench repobench-p_e",
   177:         "dataset_name": "repobench-p_e",
   178:         "max_new_tokens": 64,
   179:     },
   180:     "longbench_v2": {
   181:         "source_family": "THUDM/LongBench-v2 train split",
   182:         "max_new_tokens": 128,
   183:     },
   184:     "gsm8k": {
   185:         "source_family": "openai/gsm8k main test split",
   186:         "max_new_tokens": 256,
   187:     },
   188: }
   189: WORKLOADS = WORKLOAD_CONFIGS
   190: 
   191: 
   192: def load_hf_dataset(repo: str, config: str | None = None, split: str = "test"):
   193:     from datasets import DownloadConfig, load_dataset
   194: 
   195:     download_config = DownloadConfig(local_files_only=True)
   196:     if config is None:
   197:         dataset = load_dataset(repo, split=split, download_config=download_config)
   198:     else:
   199:         dataset = load_dataset(repo, config, split=split, download_config=download_config)
   200:     if hasattr(dataset, "keys"):
   201:         if split not in dataset:
   202:             raise RuntimeError(f"Cached dataset {repo}/{config or ''} does not contain required split {split!r}")
   203:         return dataset[split]
   204:     return dataset
   205: 
   206: 
   207: def load_cached_gsm8k_test():
   208:     from datasets import Dataset
   209:     from datasets import config as datasets_config
   210: 
   211:     cache_roots = []
   212:     if os.environ.get("HF_DATASETS_CACHE"):
   213:         cache_roots.append(Path(os.environ["HF_DATASETS_CACHE"]))
   214:     if os.environ.get("HF_HOME"):
   215:         cache_roots.append(Path(os.environ["HF_HOME"]) / "datasets")
   216:         cache_roots.append(Path(os.environ["HF_HOME"]))
   217:     if os.environ.get("HF_HUB_CACHE"):
   218:         cache_roots.append(Path(os.environ["HF_HUB_CACHE"]).parent / "datasets")
   219:     if os.environ.get("HUGGINGFACE_HUB_CACHE"):
   220:         cache_roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]).parent / "datasets")
   221:     if os.environ.get("MODEL_ID"):
   222:         model_path = Path(os.environ["MODEL_ID"]).expanduser()
   223:         if model_path.exists():
   224:             for parent in model_path.resolve().parents:
   225:                 cache_roots.append(parent / "datasets")
   226:                 if parent.name == "hub":
   227:                     cache_roots.append(parent.parent / "datasets")
   228:     if getattr(datasets_config, "HF_DATASETS_CACHE", None):
   229:         cache_roots.append(Path(datasets_config.HF_DATASETS_CACHE))
   230:     cache_roots.append(Path.home() / ".cache" / "huggingface" / "datasets")
   231: 
   232:     seen = set()
   233:     for root in cache_roots:
   234:         root = root.expanduser()
   235:         if root in seen or not root.exists():
   236:             continue
   237:         seen.add(root)
   238:         for pattern in (
   239:             "openai___gsm8k/main/**/gsm8k-test.arrow",
   240:             "**/openai___gsm8k/main/**/gsm8k-test.arrow",
   241:             "**/gsm8k-test.arrow",
   242:         ):
   243:             matches = sorted(root.glob(pattern))
   244:             if matches:
   245:                 return Dataset.from_file(str(matches[0]))
   246:     raise FileNotFoundError("Unable to locate cached openai/gsm8k main test split")
   247: 
   248: 
   249: def hf_dataset_file(repo: str, filename: str) -> Path:
   250:     from huggingface_hub import hf_hub_download
   251: 
   252:     return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_files_only=True))
   253: 
   254: 
   255: def limit_reached(examples: list[dict], max_examples: int) -> bool:
   256:     return max_examples > 0 and len(examples) >= max_examples
   257: 
   258: 
   259: def read_jsonl_lines(lines, max_examples: int = 0) -> list[dict]:
   260:     rows = []
   261:     for line in lines:
   262:         if isinstance(line, bytes):
   263:             line = line.decode("utf-8")
   264:         line = line.strip()
   265:         if not line:
   266:             continue
   267:         rows.append(json.loads(line))
   268:         if limit_reached(rows, max_examples):
   269:             break
   270:     return rows
   271: 
   272: 
   273: def load_longbench_rows(dataset_name: str, max_examples: int) -> list[dict]:
   274:     archive_path = hf_dataset_file("THUDM/LongBench", "data.zip")
   275:     with zipfile.ZipFile(archive_path) as archive:
   276:         candidates = [
   277:             name for name in archive.namelist()
   278:             if name.endswith(f"{dataset_name}.jsonl") or name.endswith(f"{dataset_name}.json")
   279:         ]
   280:         if not candidates:
   281:             raise FileNotFoundError(f"Unable to find {dataset_name} inside THUDM/LongBench data.zip")
   282:         with archive.open(candidates[0]) as fh:
   283:             return read_jsonl_lines(fh, max_examples)
   284: 
   285: 
   286: def build_longbench_examples(workload_name: str, max_examples: int) -> list[dict]:
   287:     dataset_name = WORKLOAD_CONFIGS[workload_name]["dataset_name"]
   288:     template = LONG_BENCH_TEMPLATES[dataset_name]
   289:     examples = []
   290:     for raw in load_longbench_rows(dataset_name, max_examples):
   291:         answers = raw.get("answers") or raw.get("outputs") or raw.get("answer") or []
   292:         if isinstance(answers, str):
   293:             answers = [answers]
   294:         examples.append(
   295:             {
   296:                 "example_id": raw.get("_id", f"{dataset_name}-{len(examples)}"),
   297:                 "dataset": dataset_name,
   298:                 "prompt": template.format(context=raw["context"], input=raw["input"]),
   299:                 "answers": [str(answer) for answer in answers],
   300:             }
   301:         )
   302:         if limit_reached(examples, max_examples):
   303:             return examples
   304:     return examples
   305: 
   306: 
   307: def load_longbench_v2_rows(max_examples: int) -> list[dict]:
   308:     errors = []
   309:     for repo in ("THUDM/LongBench-v2", "zai-org/LongBench-v2"):
   310:         try:
   311:             dataset = load_hf_dataset(repo, split="train")
   312:         except Exception as exc:
   313:             errors.append(exc)
   314:             continue
   315:         rows = []
   316:         for raw in dataset:
   317:             rows.append(dict(raw))
   318:             if limit_reached(rows, max_examples):
   319:                 break
   320:         return rows
   321:     raise RuntimeError("Unable to load LongBench v2 from Hugging Face.") from errors[-1]
   322: 
   323: 
   324: def build_longbench_v2_examples(max_examples: int) -> list[dict]:
   325:     examples = []
   326:     for raw in load_longbench_v2_rows(max_examples):
   327:         examples.append(
   328:             {
   329:                 "example_id": raw.get("_id", f"longbench-v2-{len(examples)}"),
   330:                 "dataset": "longbench_v2",
   331:                 "difficulty": raw.get("difficulty", ""),
   332:                 "length": raw.get("length", ""),
   333:                 "domain": raw.get("domain", ""),
   334:                 "prompt": LONG_BENCH_V2_TEMPLATE.format(
   335:                     context=str(raw["context"]).strip(),
   336:                     question=str(raw["question"]).strip(),
   337:                     choice_A=str(raw["choice_A"]).strip(),
   338:                     choice_B=str(raw["choice_B"]).strip(),
   339:                     choice_C=str(raw["choice_C"]).strip(),
   340:                     choice_D=str(raw["choice_D"]).strip(),
   341:                 ),
   342:                 "answers": [str(raw["answer"]).strip().upper()],
   343:             }
   344:         )
   345:         if limit_reached(examples, max_examples):
   346:             return examples
   347:     return examples
   348: 
   349: 
   350: def extract_boxed_answer(text: str) -> str:
   351:     marker = "\\boxed{"
   352:     start = text.rfind(marker)
   353:     if start == -1:
   354:         return ""
   355:     i = start + len(marker)
   356:     depth = 1
   357:     chars = []
   358:     while i < len(text):
   359:         ch = text[i]
   360:         if ch == "{":
   361:             depth += 1
   362:         elif ch == "}":
   363:             depth -= 1
   364:             if depth == 0:
   365:                 return "".join(chars).strip()
   366:         chars.append(ch)
   367:         i += 1
   368:     return ""
   369: 
   370: 
   371: def normalize_math_answer(text: str) -> str:
   372:     answer = extract_boxed_answer(text) or text
   373:     answer = answer.strip().replace("$", "")
   374:     answer = answer.replace("\\left", "").replace("\\right", "")
   375:     answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
   376:     answer = re.sub(r"\s+", "", answer)
   377:     answer = answer.rstrip(".;,").replace(",", "")
   378:     return answer.lower()
   379: 
   380: 
   381: def extract_math_answer(text: str) -> str:
   382:     boxed = extract_boxed_answer(text)
   383:     if boxed:
   384:         return boxed
   385:     numeric_matches = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text.replace(",", ""))
   386:     if numeric_matches:
   387:         return numeric_matches[-1]
   388:     return text.strip()
   389: 
   390: 
   391: def build_gsm8k_examples(max_examples: int) -> list[dict]:
   392:     try:
   393:         dataset = load_hf_dataset("openai/gsm8k", "main", split="test")
   394:     except Exception:
   395:         dataset = load_cached_gsm8k_test()
   396:     examples = []
   397:     for raw in dataset:
   398:         answer = extract_math_answer(raw.get("answer", ""))
   399:         prompt = (
   400:             "Solve the following grade-school math word problem carefully. "
   401:             "Show your reasoning and end with the final answer wrapped in \\\\boxed{}.\n\n"
   402:             f"Problem: {raw['question']}\n\nSolution:"
   403:         )
   404:         examples.append(
   405:             {
   406:                 "example_id": f"gsm8k-{len(examples)}",
   407:                 "dataset": "gsm8k",
   408:                 "prompt": prompt,
   409:                 "answers": [answer],
   410:             }
   411:         )
   412:         if limit_reached(examples, max_examples):
   413:             return examples
   414:     return examples
   415: 
   416: 
   417: def load_workload(name: str, max_examples: int = DEFAULT_MAX_EXAMPLES) -> WorkloadSpec:
   418:     if name.startswith("longbench_"):
   419:         if name == "longbench_v2":
   420:             examples = build_longbench_v2_examples(max_examples)
   421:         else:
   422:             examples = build_longbench_examples(name, max_examples)
   423:     elif name == "gsm8k":
   424:         examples = build_gsm8k_examples(max_examples)
   425:     else:
   426:         raise ValueError(f"Unsupported workload: {name}")
   427:     return WorkloadSpec(
   428:         name=name,
   429:         source_family=WORKLOAD_CONFIGS[name]["source_family"],
   430:         examples=examples,
   431:         max_new_tokens=WORKLOAD_CONFIGS[name]["max_new_tokens"],
   432:     )
   433: 
   434: 
   435: def normalize_text(text: str) -> str:
   436:     def remove_articles(value: str) -> str:
   437:         return re.sub(r"\b(a|an|the)\b", " ", value)
   438: 
   439:     value = text.lower()
   440:     value = "".join(ch for ch in value if ch not in set(string.punctuation))
   441:     value = remove_articles(value)
   442:     return " ".join(value.split())
   443: 
   444: 
   445: def token_f1(prediction: str, ground_truth: str) -> float:
   446:     pred_tokens = normalize_text(prediction).split()
   447:     gold_tokens = normalize_text(ground_truth).split()
   448:     if not pred_tokens or not gold_tokens:
   449:         return 0.0
   450:     common = Counter(pred_tokens) & Counter(gold_tokens)
   451:     num_same = sum(common.values())
   452:     if num_same == 0:
   453:         return 0.0
   454:     precision = num_same / len(pred_tokens)
   455:     recall = num_same / len(gold_tokens)
   456:     return 2 * precision * recall / (precision + recall)
   457: 
   458: 
   459: def retrieval_score(prediction: str, ground_truth: str) -> float:
   460:     match = re.search(r"Paragraph (\d+)", ground_truth)
   461:     if not match:
   462:         return 0.0
   463:     gold_id = match.group(1)
   464:     numbers = re.findall(r"\d+", prediction)
   465:     if not numbers:
   466:         return 0.0
   467:     return sum(1.0 for number in numbers if number == gold_id) / len(numbers)
   468: 
   469: 
   470: def code_similarity_score(prediction: str, ground_truth: str) -> float:
   471:     candidate = ""
   472:     for line in prediction.lstrip("\n").splitlines():
   473:         if "`" not in line and "#" not in line and "//" not in line:
   474:             candidate = line
   475:             break
   476:     return int(round(100 * difflib.SequenceMatcher(None, candidate, ground_truth).ratio())) / 100.0
   477: 
   478: 
   479: def extract_choice_answer(text: str) -> str:
   480:     response = text.replace("*", "")
   481:     patterns = (
   482:         r"The correct answer is \(([A-D])\)",
   483:         r"The correct answer is ([A-D])",
   484:         r"answer is \(([A-D])\)",
   485:         r"answer is ([A-D])",
   486:         r"\(([A-D])\)",
   487:         r"\b([A-D])\b",
   488:     )
   489:     for pattern in patterns:
   490:         match = re.search(pattern, response, re.IGNORECASE)
   491:         if match:
   492:             return match.group(1).upper()
   493:     return ""
   494: 
   495: 
   496: def score_prediction(workload_name: str, example: dict, prediction: str) -> float:
   497:     answers = [str(answer) for answer in example.get("answers", []) if str(answer)]
   498:     if not prediction or not answers:
   499:         return 0.0
   500:     dataset = example.get("dataset", "")

[truncated: showing at most 500 lines / 60000 bytes from transformers-kv-lab/custom_selection_eval.py]
```




## How You Will Be Evaluated

After you finish, evaluation runs a fixed set of scripts and aggregates the
metrics they emit. These scripts are **not** in your workspace — you cannot
read or modify them. The labels below indicate what each evaluation tests:

- **longbench-hotpotqa** — wall-clock budget `02:00:00`, compute share `1.0`
- **longbench-passage-retrieval** — wall-clock budget `02:00:00`, compute share `1.0`
- **longbench-repobench** — wall-clock budget `03:00:00`, compute share `1.0`
- **longbench-v2** — wall-clock budget `06:00:00`, compute share `1.0`
- **gsm8k** — wall-clock budget `04:00:00`, compute share `1.0`


Scoring uses the same `combined_score` aggregation as the MLS-Bench
leaderboard. Multiple seeds are averaged.



## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `full_attention` baseline — editable region  [READ-ONLY — reference implementation]

In `transformers-kv-lab/custom_selection_eval.py`:

```python
Lines 40–56:
    37:     max_new_tokens: int
    38: 
    39: 
    40: class SelectionPolicy:
    41:     """Naive full-attention anchor: keep every prefill KV token."""
    42: 
    43:     method_name = "full_attention"
    44:     rerotate_selected_keys = False
    45: 
    46:     def retention_plan(self, layer_id, request_meta, cache_meta):
    47:         return {
    48:             "method": self.method_name,
    49:             "disable_compression": True,
    50:         }
    51: 
    52:     def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
    53:         return None
    54: 
    55:     def select_cache(self, module, keys, values, scores, n_kept):
    56:         return keys, values
    57: 
    58: 
    59: def resolve_task_dir() -> Path:
```

### `streamingllm` baseline — editable region  [READ-ONLY — reference implementation]

In `transformers-kv-lab/custom_selection_eval.py`:

```python
Lines 40–97:
    37:     max_new_tokens: int
    38: 
    39: 
    40: class SelectionPolicy:
    41:     """StreamingLLM: keep attention sinks and the most recent tokens."""
    42: 
    43:     method_name = "streamingllm"
    44:     rerotate_selected_keys = True
    45: 
    46:     def retention_plan(self, layer_id, request_meta, cache_meta):
    47:         return {
    48:             "method": self.method_name,
    49:             "sink_tokens": 4,
    50:             "compression_ratio": cache_meta["compression_ratio"],
    51:         }
    52: 
    53:     def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
    54:         k_len = int(keys.shape[2])
    55:         n_sink = int(plan.get("sink_tokens", 4))
    56:         ratio = float(plan["compression_ratio"])
    57:         assert k_len > n_sink, f"Input should contain more tokens than sink_tokens={n_sink}"
    58:         n_pruned = k_len - int(k_len * (1.0 - ratio))
    59:         scores = torch.ones_like(keys[..., 0])
    60:         scores[:, :, n_sink : n_sink + n_pruned] = 0
    61:         return scores
    62: 
    63:     def rotate_half(self, x):
    64:         x1 = x[..., : x.shape[-1] // 2]
    65:         x2 = x[..., x.shape[-1] // 2 :]
    66:         return torch.cat((-x2, x1), dim=-1)
    67: 
    68:     def rerotate_cache_keys(self, module, indices, keys):
    69:         bsz, num_key_value_heads, n_kept = indices.shape
    70:         device = indices.device
    71:         device_type = keys.device.type
    72:         dtype = keys.dtype
    73:         inv_freq = module.rotary_emb.inv_freq[None, None, :, None].float().expand(
    74:             bsz, num_key_value_heads, -1, 1
    75:         )
    76:         new_positions = torch.arange(0, n_kept, device=device).unsqueeze(0)[:, None, :].float()
    77:         new_positions = new_positions.expand(bsz, num_key_value_heads, n_kept)
    78:         delta_pos = (new_positions - indices.float()).unsqueeze(2)
    79:         device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    80:         with torch.autocast(device_type=device_type, enabled=False):
    81:             freqs = (delta_pos.float() * inv_freq.float()).transpose(2, 3)
    82:             emb = torch.cat((freqs, freqs), dim=-1)
    83:             cos = emb.cos().contiguous()
    84:             sin = emb.sin().contiguous()
    85:         cos = cos.to(dtype=dtype)
    86:         sin = sin.to(dtype=dtype)
    87:         gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
    88:         gathered = keys.gather(2, gather_idx).contiguous()
    89:         return (gathered * cos) + (self.rotate_half(gathered) * sin)
    90: 
    91:     def select_cache(self, module, keys, values, scores, n_kept):
    92:         indices = scores.topk(n_kept, dim=-1).indices
    93:         indices = torch.sort(indices, dim=2).values
    94:         selected_keys = self.rerotate_cache_keys(module, indices, keys)
    95:         gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
    96:         selected_values = values.gather(2, gather_idx).contiguous()
    97:         return selected_keys, selected_values
    98: 
    99: 
   100: def resolve_task_dir() -> Path:
```

### `expected_attention` baseline — editable region  [READ-ONLY — reference implementation]

In `transformers-kv-lab/custom_selection_eval.py`:

```python
Lines 40–134:
    37:     max_new_tokens: int
    38: 
    39: 
    40: class SelectionPolicy:
    41:     """Expected Attention: estimate future-query attention before pruning."""
    42: 
    43:     method_name = "expected_attention"
    44:     rerotate_selected_keys = False
    45: 
    46:     def repeat_kv(self, hidden_states, n_rep):
    47:         if n_rep == 1:
    48:             return hidden_states
    49:         bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
    50:         hidden_states = hidden_states[:, :, None, :, :].expand(
    51:             bsz, num_key_value_heads, n_rep, slen, head_dim
    52:         )
    53:         return hidden_states.reshape(bsz, num_key_value_heads * n_rep, slen, head_dim)
    54: 
    55:     def get_prerope_query_states(self, module, hidden_states):
    56:         bsz, q_len, _ = hidden_states.shape
    57:         num_heads = int(module.config.num_attention_heads)
    58:         head_dim = int(module.head_dim)
    59:         if hasattr(module, "q_proj"):
    60:             query_states = module.q_proj(hidden_states)
    61:         elif hasattr(module, "qkv_proj"):
    62:             qkv = module.qkv_proj(hidden_states)
    63:             query_states = qkv[..., : num_heads * head_dim]
    64:         else:
    65:             raise NotImplementedError(f"Query projection not implemented for {module.__class__}.")
    66:         query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    67:         if hasattr(module, "q_norm"):
    68:             query_states = module.q_norm(query_states)
    69:         return query_states
    70: 
    71:     def avg_rope(self, module, mu, cov, q_len, n_future_positions):
    72:         position_ids = torch.arange(q_len, q_len + n_future_positions, device=mu.device).unsqueeze(0)
    73:         head_dim = int(module.head_dim)
    74:         cos, sin = module.rotary_emb(mu, position_ids)
    75:         cos, sin = cos[0], sin[0]
    76:         identity = torch.eye(head_dim, device=cos.device, dtype=cos.dtype)
    77:         perm = torch.zeros((head_dim, head_dim), device=cos.device, dtype=cos.dtype)
    78:         half = head_dim // 2
    79:         perm[half:, :half] = torch.eye(half, device=cos.device, dtype=cos.dtype)
    80:         perm[:half, half:] = -torch.eye(half, device=cos.device, dtype=cos.dtype)
    81:         rotation = (cos.unsqueeze(1) * identity + sin.unsqueeze(1) * perm).mean(dim=0).to(mu.device)
    82:         mu = torch.matmul(mu, rotation.T)
    83:         if cov is not None:
    84:             cov = torch.matmul(rotation, torch.matmul(cov, rotation.T))
    85:         return mu, cov
    86: 
    87:     def retention_plan(self, layer_id, request_meta, cache_meta):
    88:         return {
    89:             "method": self.method_name,
    90:             "sink_tokens": 4,
    91:             "n_future_positions": 512,
    92:             "use_covariance": True,
    93:             "use_value_norm": True,
    94:             "epsilon": 0.0,
    95:             "compression_ratio": cache_meta["compression_ratio"],
    96:         }
    97: 
    98:     def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
    99:         n_sink = int(plan.get("sink_tokens", 4))
   100:         n_future = int(plan.get("n_future_positions", 512))
   101:         use_covariance = bool(plan.get("use_covariance", True))
   102:         use_vnorm = bool(plan.get("use_value_norm", True))
   103:         epsilon = float(plan.get("epsilon", 0.0))
   104:         assert keys.size(2) > n_sink, f"Input should contain more tokens than sink_tokens={n_sink}"
   105:         keys_body = keys[:, :, n_sink:]
   106:         values_body = values[:, :, n_sink:]
   107:         h = hidden_states[:, n_sink:]
   108:         query_states = self.get_prerope_query_states(module, h)
   109:         mean_query = query_states.mean(dim=2, keepdim=True)
   110:         cov_query = None
   111:         if use_covariance:
   112:             centered_states = query_states - mean_query
   113:             cov_query = torch.einsum("bnsi,bnsj->bnij", centered_states, centered_states) / max(h.shape[1], 1)
   114:         mean_query = mean_query.squeeze(2)
   115:         mean_query, cov_query = self.avg_rope(module, mean_query, cov_query, hidden_states.shape[1], n_future)
   116:         bsz, num_key_value_heads, q_len, dim = keys_body.shape
   117:         num_key_value_groups = int(module.config.num_attention_heads) // num_key_value_heads
   118:         repeated_keys = self.repeat_kv(keys_body, num_key_value_groups).transpose(2, 3)
   119:         scores = torch.matmul(mean_query.unsqueeze(2), repeated_keys).squeeze(2) / math.sqrt(dim)
   120:         if use_covariance:
   121:             scores += torch.einsum("bhin,bhij,bhjn->bhn", repeated_keys, cov_query, repeated_keys) / dim / 2
   122:         scores = F.softmax(scores, dim=-1)
   123:         scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len).mean(dim=2)
   124:         if use_vnorm:
   125:             scores = (scores + epsilon) * values_body.norm(dim=-1)
   126:         return F.pad(scores, (n_sink, 0), value=scores.max().item())
   127: 
   128:     def select_cache(self, module, keys, values, scores, n_kept):
   129:         indices = scores.topk(n_kept, dim=-1).indices
   130:         gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
   131:         selected_keys = keys.gather(2, gather_idx).contiguous()
   132:         gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
   133:         selected_values = values.gather(2, gather_idx).contiguous()
   134:         return selected_keys, selected_values
   135: 
   136: 
   137: def resolve_task_dir() -> Path:
```

### `lagkv` baseline — editable region  [READ-ONLY — reference implementation]

In `transformers-kv-lab/custom_selection_eval.py`:

```python
Lines 40–92:
    37:     max_new_tokens: int
    38: 
    39: 
    40: class SelectionPolicy:
    41:     """LagKV: score tokens by lag-relative key/value variation."""
    42: 
    43:     method_name = "lagkv"
    44:     rerotate_selected_keys = False
    45: 
    46:     def retention_plan(self, layer_id, request_meta, cache_meta):
    47:         return {
    48:             "method": self.method_name,
    49:             "sink_tokens": 4,
    50:             "lag_size": 128,
    51:             "cross_scoring": False,
    52:             "compression_ratio": cache_meta["compression_ratio"],
    53:         }
    54: 
    55:     def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
    56:         bsz, num_key_value_heads, q_len, dim = keys.shape
    57:         n_sink = int(plan.get("sink_tokens", 4))
    58:         lag_size = int(plan.get("lag_size", 128))
    59:         if q_len < n_sink + 2 * lag_size:
    60:             scores = torch.ones((bsz, num_key_value_heads, q_len), dtype=keys.dtype, device=keys.device)
    61:             if q_len > n_sink:
    62:                 scores[:, :, n_sink:] = (
    63:                     torch.arange(q_len - n_sink, device=keys.device) / (q_len - n_sink)
    64:                 ).to(keys.dtype)
    65:             return scores
    66:         end_idx = n_sink + ((q_len - n_sink) // lag_size) * lag_size
    67:         tail_len = lag_size + q_len - end_idx
    68: 
    69:         def state_score(target):
    70:             ref = target[:, :, 1:, :, :]
    71:             value = target[:, :, :-1, :, :]
    72:             min_ref = ref.min(dim=-2).values.unsqueeze(-2).expand_as(value)
    73:             max_ref = ref.max(dim=-2).values.unsqueeze(-2).expand_as(value)
    74:             return ((value - min_ref) / (max_ref - min_ref)).std(dim=-1).softmax(dim=-1)
    75: 
    76:         key_score = state_score(keys[:, :, n_sink:end_idx].view(bsz, num_key_value_heads, -1, lag_size, dim))
    77:         value_score = state_score(values[:, :, n_sink:end_idx].view(bsz, num_key_value_heads, -1, lag_size, dim))
    78:         scores = (key_score + value_score) / 2
    79:         if not bool(plan.get("cross_scoring", False)):
    80:             scores = scores.argsort(dim=-1).argsort(dim=-1) / lag_size
    81:             scores = scores.to(keys.dtype)
    82:         sink_scores = torch.ones((bsz, num_key_value_heads, n_sink), dtype=scores.dtype, device=scores.device)
    83:         tail_scores = torch.ones((bsz, num_key_value_heads, tail_len), dtype=scores.dtype, device=scores.device)
    84:         return torch.cat((sink_scores, scores.reshape(bsz, num_key_value_heads, -1), tail_scores), dim=-1)
    85: 
    86:     def select_cache(self, module, keys, values, scores, n_kept):
    87:         indices = scores.topk(n_kept, dim=-1).indices
    88:         gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
    89:         selected_keys = keys.gather(2, gather_idx).contiguous()
    90:         gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
    91:         selected_values = values.gather(2, gather_idx).contiguous()
    92:         return selected_keys, selected_values
    93: 
    94: 
    95: def resolve_task_dir() -> Path:
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
