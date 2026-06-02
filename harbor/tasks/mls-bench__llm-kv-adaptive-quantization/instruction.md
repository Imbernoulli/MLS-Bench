# MLS-Bench: llm-kv-adaptive-quantization

# LLM KV Cache: Adaptive Quantization Policy

## Research Question

Design an adaptive low-bit KV-cache quantizer for decoder-only LLM
inference on top of a tensor-level Hugging Face `Transformers` replay
harness. The task asks whether an algorithm can preserve benchmark output
quality while reducing the effective KV footprint through bit allocation,
axis selection, residual windows, and optional prefill-time observation.

## Background

KV-cache quantization compresses the running key/value tensors used during
decode rather than the model weights. Three reference families inform the
design space exposed by this task:

- KIVI (Liu et al., ICML 2024; arXiv:2402.02750) is a tuning-free 2-bit KV
  quantizer that quantizes the key cache per-channel and the value cache
  per-token, with a residual tail of recent FP entries to preserve the
  freshly generated rows.
- KVTuner (Li et al., 2025; arXiv:2502.04420) treats KV quantization as a
  layer-wise mixed-precision search, picking different bit-widths per layer
  based on sensitivity. Its Qwen2.5-3B configurations are used directly as
  baseline presets for this task.
- SQuat (Zhang et al., 2025; arXiv:2503.24358) builds a query subspace via
  SVD during prefill and enforces that the (de)quantized key error is
  orthogonal to that subspace, reducing attention-output drift from key
  quantization.

The harness exposes the union of behaviors needed to implement these
methods on a common tensor-level interface, so the contribution is a
quantization policy rather than a backend choice.

## What You Can Modify

The editable region is the `AdaptiveKVQuantizer` class in
`custom_quant_eval.py`. The fixed harness supplies real KV-cache tensors and
calls the editable class:

- `reset_request(request_meta, budget_state)` before each example
- `needs_prefill_qkv_observer() -> bool`
- `query_observation_position() -> str`
- `observe_prefill_qkv(layer_id, query_states, key_states, value_states, attention_meta)`
- `quantize_key(layer_id, key_states, cache_meta) -> tensor | (tensor, avg_bits)`
- `quantize_value(layer_id, value_states, cache_meta) -> tensor | (tensor, avg_bits)`
- `estimate_bits(layer_id, kv_kind, seq_len, head_dim, cache_meta) -> float`

`key_states` and `value_states` have shape
`[batch, heads, seq_len, head_dim]`. The class implements the actual tensor
algorithm: grouping, asymmetric ranges, zero-points, per-layer bit presets,
residual retention, query-subspace transforms, and memory accounting all
belong inside this class. The task does not expose a fixed algorithm enum
or a backend selector.

## What You Cannot Modify

- The model family and deterministic decode replay loop
- The benchmark workload definitions, prompts, generation limits
- The parser or final-score definitions
- The underlying `Transformers` model implementation

## Evaluation

The visible model is `Qwen/Qwen2.5-3B-Instruct`. Workloads share the same
public text-benchmark protocol with `llm-kv-selection-budgeting`; dataset
sources, split names, prompt templates, generation limits, and scoring
semantics stay aligned across those tasks whenever a workload label is
shared.

| Label | Source | Final score |
|---|---|---|
| `longbench_hotpotqa` | LongBench-E `hotpotqa_e` | LongBench QA F1 (0-100) |
| `longbench_passage_retrieval` | LongBench-E `passage_retrieval_en_e` | LongBench retrieval score (0-100) |
| `longbench_repobench` | LongBench-E `repobench-p_e` | LongBench code similarity (0-100) |
| `needlebench_niah` | RULER/NeedleBench-style needle in public essay text | exact phrase retrieval accuracy (0-100) |
| `gsm8k` | `openai/gsm8k` main test split | exact final-answer accuracy after numeric normalization (0-100) |

For NIAH the canonical needle is:
`The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.`

The parser expects one `TEST_METRICS:` line per workload with:

- `final_score`: benchmark-native quality on a 0-100 scale
- `effective_kv_bits`: quantizer-level effective KV bits per cached element
- `kv_compression_ratio`: `16 / effective_kv_bits`, using FP16 KV as the
  reference footprint
- `runtime_seconds`: task-level wall-clock runtime for the workload command

`effective_kv_bits` is computed from the submitted quantizer at a 4096-token
reference KV span so the efficiency term is hardware-independent.

## Baselines

Baselines are paper-linked configurations implemented on the same tensor
interface; only the algorithm is the contribution, not a paper repository
backend.

- `kivi_overlap_4bit`: KIVI-style K4/V4 (arXiv:2402.02750) with key
  per-channel, value per-token, group size `32`, key `block_modulo`
  residual blocks, value tail residual length `128`.
- `kvtuner4_pertoken_qwen25_3b`: KVTuner (arXiv:2502.04420)
  `Qwen2.5-3B-Instruct_pertoken_KVTuner4_0.yaml` preset, signed-asymmetric
  vanilla cache formula, axis `0/0`, `q_group_size=-1`, `residual_length=0`.
- `kvtuner4_kivi_qwen25_3b`: KVTuner
  `Qwen2.5-3B-Instruct_kivi_KVTuner4_0.yaml` preset, signed-asymmetric
  formula, axis `1/0`, `block_modulo` residual length `32`,
  `q_group_size=32`.
- `squat_subspace_4bit`: SQuat (arXiv:2503.24358) LongBench-style 4-bit
  configuration: query SVD during prefill, future-dimension key correction,
  subspace dimension `60`, `squat_lambda=0.001`, `quant_group_size=64`,
  `shared_svd=True`, K/V group size `32`, residual block length `32`.

## Canonical Ranking

The leaderboard uses the standard mature MLS-Bench text-task pattern:

- each workload emits a benchmark-native `final_score_*` quality column
- each workload also emits a hardware-independent `kv_compression_ratio_*`
  efficiency column
- quality uses the repository standard bounded-power normalization with the
  worst current baseline as the floor, `100` as the bound, and the best
  current baseline as the reference point
- efficiency uses bounded-power normalization on `kv_compression_ratio` with
  the worst current baseline as the floor, `4x` compression as the
  reference, and `8x` compression as the bound
- `runtime_seconds_*` is an emitted diagnostic column but does not enter
  the final score (this tensor-replay harness is not a runtime-native
  packed-cache speed benchmark)
- each workload score is a weighted mean with quality weight `6` and KV
  efficiency weight `4`
- the task score is the geometric mean across the LongBench-E workloads,
  NIAH, and GSM8K

## Notes

- The harness runs deterministic greedy generation over `Transformers`
  decode steps and scores generated answers.
- At each decode step after prefill, it snapshots real KV tensors,
  quantizes them with the current quantizer, restores the quantized cache,
  and advances generation.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/transformers-kv-lab/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `transformers-kv-lab/custom_quant_eval.py`
- editable lines **41–172**


Other files you may **read** for context (do not modify):
- `transformers-kv-lab/src/transformers/cache_utils.py`


## Readable Context


### `transformers-kv-lab/custom_quant_eval.py`  [EDITABLE — lines 41–172 only]

```python
     1: """Tensor-level KV-cache quantization replay harness.
     2: 
     3: This scaffold replays deterministic decode steps on top of Hugging Face
     4: Transformers. Instead of collapsing the policy into one global
     5: QuantizedCacheConfig, it snapshots real KV tensors, quantizes them with
     6: source-backed overlap rules, and replays the next decode step with the
     7: quantized cache.
     8: """
     9: 
    10: from __future__ import annotations
    11: 
    12: import argparse
    13: import difflib
    14: import json
    15: import math
    16: import os
    17: import re
    18: import string
    19: import zipfile
    20: from collections import Counter
    21: from dataclasses import dataclass
    22: from pathlib import Path
    23: from statistics import mean
    24: 
    25: import torch
    26: from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, set_seed
    27: 
    28: 
    29: @dataclass
    30: class WorkloadSpec:
    31:     name: str
    32:     source_family: str
    33:     examples: list[dict]
    34:     max_new_tokens: int
    35: 
    36:     @property
    37:     def prompts(self) -> list[str]:
    38:         return [example["prompt"] for example in self.examples]
    39: 
    40: 
    41: class AdaptiveKVQuantizer:
    42:     """Editable KV-cache quantizer.
    43: 
    44:     The fixed harness supplies real key/value tensors from a Hugging Face
    45:     DynamicCache and calls this class for the actual algorithm. Participants
    46:     may rewrite the quantization math, residual policy, optional prefill
    47:     observation, and memory accounting here without changing the benchmark
    48:     datasets, model, or decode loop.
    49:     """
    50: 
    51:     def __init__(self):
    52:         self.bits = 4
    53:         self.key_group_size = 32
    54:         self.value_group_size = 32
    55:         self.key_residual_length = 128
    56:         self.value_residual_length = 128
    57: 
    58:     def reset_request(self, request_meta: dict, budget_state: dict):
    59:         self.bits = min(4, int(budget_state.get("budget_bits", 4)))
    60:         workload = str(request_meta.get("workload", ""))
    61:         residual = 128 if workload.startswith("longbench_") else 32
    62:         self.key_residual_length = residual
    63:         self.value_residual_length = residual
    64: 
    65:     def needs_prefill_qkv_observer(self) -> bool:
    66:         return False
    67: 
    68:     def observe_prefill_qkv(
    69:         self,
    70:         layer_id: int,
    71:         query_states: torch.Tensor | None,
    72:         key_states: torch.Tensor | None,
    73:         value_states: torch.Tensor | None,
    74:         attention_meta: dict,
    75:     ) -> None:
    76:         return None
    77: 
    78:     def query_observation_position(self) -> str:
    79:         return "post_rope"
    80: 
    81:     def _residual_keep_length(self, seq_len: int, residual_length: int, residual_policy: str = "tail") -> int:
    82:         residual_length = max(0, min(seq_len, int(residual_length)))
    83:         if residual_length == 0 or residual_policy in {"none", ""}:
    84:             return 0
    85:         if residual_policy == "block_modulo":
    86:             return seq_len % residual_length
    87:         if residual_policy == "tail":
    88:             return residual_length
    89:         raise ValueError(f"Unsupported residual_policy={residual_policy}")
    90: 
    91:     def _minmax_quantize_last_dim(self, data: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    92:         if data.numel() == 0 or bits >= FP_BITS - 0.5:
    93:             return data
    94:         max_int = max(1, int(2**int(bits)) - 1)
    95:         trailing = data.shape[-1]
    96:         group_size = trailing if int(group_size) <= 0 else int(group_size)
    97:         padded = math.ceil(trailing / group_size) * group_size
    98:         work = data
    99:         if padded != trailing:
   100:             work = torch.nn.functional.pad(work, (0, padded - trailing))
   101:         grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
   102:         gmin = grouped.amin(dim=-1, keepdim=True)
   103:         gmax = grouped.amax(dim=-1, keepdim=True)
   104:         scale = (gmax - gmin).clamp(min=1e-5) / max_int
   105:         quant = torch.round((grouped - gmin) / scale).clamp(0, max_int)
   106:         dequant = quant.mul(scale).add(gmin)
   107:         return dequant.reshape(*work.shape[:-1], padded)[..., :trailing]
   108: 
   109:     def _quantize_grouped_minmax(
   110:         self,
   111:         layer_tensor: torch.Tensor,
   112:         *,
   113:         axis: str,
   114:         bits: int,
   115:         group_size: int,
   116:         residual_length: int,
   117:         residual_policy: str = "tail",
   118:     ) -> tuple[torch.Tensor, float]:
   119:         work = layer_tensor.float().clone()
   120:         batch, heads, seq_len, head_dim = work.shape
   121:         residual = self._residual_keep_length(seq_len, residual_length, residual_policy)
   122:         quant_end = seq_len - residual
   123:         if quant_end <= 0 or bits >= FP_BITS - 0.5:
   124:             return work.to(layer_tensor.dtype), FP_BITS
   125: 
   126:         quant_slice = work[:, :, :quant_end, :]
   127:         if axis == "channel":
   128:             quant_len = quant_slice.shape[-2]
   129:             group_size = quant_len if int(group_size) <= 0 else int(group_size)
   130:             usable = quant_len - (quant_len % group_size)
   131:             main = quant_slice[:, :, :usable, :]
   132:             tail = quant_slice[:, :, usable:, :]
   133:             if usable > 0:
   134:                 main = main.transpose(2, 3).reshape(batch, heads, head_dim, usable // group_size, group_size)
   135:                 main = self._minmax_quantize_last_dim(main, bits, group_size)
   136:                 work[:, :, :usable, :] = main.reshape(batch, heads, head_dim, usable).transpose(2, 3)
   137:             if tail.numel() > 0:
   138:                 work[:, :, usable:quant_end, :] = tail
   139:             fp_tokens = residual + (quant_len - usable)
   140:             avg_bits = (usable * bits + fp_tokens * FP_BITS) / max(seq_len, 1)
   141:         else:
   142:             flat = quant_slice.transpose(1, 2).reshape(batch, quant_slice.shape[-2], heads * head_dim)
   143:             flat = self._minmax_quantize_last_dim(flat, bits, group_size)
   144:             work[:, :, :quant_end, :] = flat.reshape(batch, quant_slice.shape[-2], heads, head_dim).transpose(1, 2)
   145:             avg_bits = (quant_end * bits + residual * FP_BITS) / max(seq_len, 1)
   146:         return work.to(layer_tensor.dtype), float(avg_bits)
   147: 
   148:     def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
   149:         return self._quantize_grouped_minmax(
   150:             key_states,
   151:             axis="channel",
   152:             bits=self.bits,
   153:             group_size=self.key_group_size,
   154:             residual_length=self.key_residual_length,
   155:             residual_policy="tail",
   156:         )
   157: 
   158:     def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
   159:         return self._quantize_grouped_minmax(
   160:             value_states,
   161:             axis="token",
   162:             bits=self.bits,
   163:             group_size=self.value_group_size,
   164:             residual_length=self.value_residual_length,
   165:             residual_policy="tail",
   166:         )
   167: 
   168:     def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
   169:         residual = self.key_residual_length if kv_kind == "key" else self.value_residual_length
   170:         residual = self._residual_keep_length(seq_len, residual, "tail")
   171:         quant_tokens = max(0, seq_len - residual)
   172:         return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
   173: 
   174: 
   175: def resolve_task_dir() -> Path:
   176:     here = Path(__file__).resolve().parent
   177:     candidates = [
   178:         here,
   179:         here.parent,
   180:         Path(os.environ.get("MLSBENCH_TASK_DIR", "")),
   181:         Path(os.environ.get("TASK_DIR", "")),
   182:         Path.cwd() / "_task",
   183:         Path.cwd().parent / "_task",
   184:     ]
   185:     seen = set()
   186:     for candidate in candidates:
   187:         if not candidate or str(candidate) == ".":
   188:             continue
   189:         resolved = candidate.resolve()
   190:         if resolved in seen:
   191:             continue
   192:         seen.add(resolved)
   193:         if (resolved / "task_description.md").exists():
   194:             return resolved
   195:     raise FileNotFoundError(f"Unable to locate task directory from {here}")
   196: 
   197: 
   198: TASK_DIR = resolve_task_dir()
   199: FP_BITS = float(torch.finfo(torch.float16).bits)
   200: DEFAULT_MAX_EXAMPLES = int(os.environ.get("ADAPTIVE_KV_MAX_EXAMPLES", "0"))
   201: 
   202: LONG_BENCH_TEMPLATES = {
   203:     "hotpotqa_e": (
   204:         "Answer the question based on the given passages. "
   205:         "Only give me the answer and do not output any other words.\n\n"
   206:         "The following are given passages.\n{context}\n\n"
   207:         "Question: {input}\nAnswer:"
   208:     ),
   209:     "passage_retrieval_en_e": (
   210:         "Here are 30 paragraphs from Wikipedia, along with an abstract. "
   211:         "Please determine which paragraph the abstract is from.\n\n"
   212:         "{context}\n\n"
   213:         "The following is an abstract.\n\n{input}\n\n"
   214:         "Please enter the number of the paragraph that the abstract is from. "
   215:         "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
   216:         "The answer is: "
   217:     ),
   218:     "repobench-p_e": "Please complete the code given below.\n{context}{input}Next line of code:\n",
   219: }
   220: 
   221: LONG_BENCH_DATASETS = ("hotpotqa_e", "passage_retrieval_en_e", "repobench-p_e")
   222: NEEDLE_SENTENCE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
   223: NEEDLE_QUESTION = "The best thing to do in San Francisco is: "
   224: NEEDLE_DEPTHS = tuple(
   225:     float(x)
   226:     for x in os.environ.get("ADAPTIVE_KV_NEEDLE_DEPTHS", "0.10,0.50,0.90").split(",")
   227:     if x.strip()
   228: )
   229: 
   230: WORKLOAD_CONFIGS = {
   231:     "longbench_hotpotqa": {
   232:         "source_family": "THUDM/LongBench hotpotqa_e",
   233:         "dataset_name": "hotpotqa_e",
   234:         "max_new_tokens": 32,
   235:     },
   236:     "longbench_passage_retrieval": {
   237:         "source_family": "THUDM/LongBench passage_retrieval_en_e",
   238:         "dataset_name": "passage_retrieval_en_e",
   239:         "max_new_tokens": 32,
   240:     },
   241:     "longbench_repobench": {
   242:         "source_family": "THUDM/LongBench repobench-p_e",
   243:         "dataset_name": "repobench-p_e",
   244:         "max_new_tokens": 64,
   245:     },
   246:     "needlebench_niah": {
   247:         "source_family": "opencompass/NeedleBench English haystacks with the shared RULER/NeedleBench-style NIAH probe",
   248:         "max_new_tokens": 32,
   249:     },
   250:     "gsm8k": {
   251:         "source_family": "openai/gsm8k main test split",
   252:         "max_new_tokens": 256,
   253:     },
   254: }
   255: 
   256: 
   257: def load_hf_dataset(repo: str, config: str | None = None, split: str = "test"):
   258:     from datasets import load_dataset
   259: 
   260:     errors = []
   261:     for candidate_split in (split, "train", None):
   262:         try:
   263:             if config is None:
   264:                 dataset = load_dataset(repo, split=candidate_split) if candidate_split else load_dataset(repo)
   265:             else:
   266:                 dataset = (
   267:                     load_dataset(repo, config, split=candidate_split)
   268:                     if candidate_split
   269:                     else load_dataset(repo, config)
   270:                 )
   271:         except Exception as exc:
   272:             errors.append(exc)
   273:             continue
   274:         if hasattr(dataset, "keys"):
   275:             for key in (split, "test", "validation", "train"):
   276:                 if key in dataset:
   277:                     return dataset[key]
   278:             first_key = next(iter(dataset.keys()))
   279:             return dataset[first_key]
   280:         return dataset
   281:     raise RuntimeError(f"Unable to load Hugging Face dataset {repo}/{config or ''}") from errors[-1]
   282: 
   283: 
   284: def load_cached_gsm8k_test():
   285:     from datasets import Dataset
   286:     from datasets import config as datasets_config
   287: 
   288:     cache_roots = []
   289:     if os.environ.get("HF_DATASETS_CACHE"):
   290:         cache_roots.append(Path(os.environ["HF_DATASETS_CACHE"]))
   291:     if os.environ.get("HF_HOME"):
   292:         cache_roots.append(Path(os.environ["HF_HOME"]) / "datasets")
   293:         cache_roots.append(Path(os.environ["HF_HOME"]))
   294:     if os.environ.get("HF_HUB_CACHE"):
   295:         cache_roots.append(Path(os.environ["HF_HUB_CACHE"]).parent / "datasets")
   296:     if os.environ.get("HUGGINGFACE_HUB_CACHE"):
   297:         cache_roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]).parent / "datasets")
   298:     if os.environ.get("MODEL_ID"):
   299:         model_path = Path(os.environ["MODEL_ID"]).expanduser()
   300:         if model_path.exists():
   301:             for parent in model_path.resolve().parents:
   302:                 cache_roots.append(parent / "datasets")
   303:                 if parent.name == "hub":
   304:                     cache_roots.append(parent.parent / "datasets")
   305:     if getattr(datasets_config, "HF_DATASETS_CACHE", None):
   306:         cache_roots.append(Path(datasets_config.HF_DATASETS_CACHE))
   307:     cache_roots.append(Path.home() / ".cache" / "huggingface" / "datasets")
   308: 
   309:     seen = set()
   310:     for root in cache_roots:
   311:         if root in seen or not root.exists():
   312:             continue
   313:         seen.add(root)
   314:         patterns = (
   315:             "openai___gsm8k/main/**/gsm8k-test.arrow",
   316:             "**/openai___gsm8k/main/**/gsm8k-test.arrow",
   317:             "**/gsm8k-test.arrow",
   318:         )
   319:         for pattern in patterns:
   320:             matches = sorted(root.glob(pattern))
   321:             if matches:
   322:                 return Dataset.from_file(str(matches[0]))
   323:     raise FileNotFoundError("Unable to locate cached openai/gsm8k main test split")
   324: 
   325: 
   326: def hf_dataset_file(repo: str, filename: str) -> Path:
   327:     from huggingface_hub import hf_hub_download
   328: 
   329:     return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset"))
   330: 
   331: 
   332: def read_jsonl_lines(lines, max_examples: int = 0) -> list[dict]:
   333:     rows = []
   334:     for line in lines:
   335:         if isinstance(line, bytes):
   336:             line = line.decode("utf-8")
   337:         line = line.strip()
   338:         if not line:
   339:             continue
   340:         rows.append(json.loads(line))
   341:         if limit_reached(rows, max_examples):
   342:             break
   343:     return rows
   344: 
   345: 
   346: def load_longbench_rows(dataset_name: str, max_examples: int) -> list[dict]:
   347:     archive_path = hf_dataset_file("THUDM/LongBench", "data.zip")
   348:     with zipfile.ZipFile(archive_path) as archive:
   349:         candidates = [
   350:             name for name in archive.namelist()
   351:             if name.endswith(f"{dataset_name}.jsonl") or name.endswith(f"{dataset_name}.json")
   352:         ]
   353:         if not candidates:
   354:             raise FileNotFoundError(f"Unable to find {dataset_name} inside THUDM/LongBench data.zip")
   355:         with archive.open(candidates[0]) as fh:
   356:             return read_jsonl_lines(fh, max_examples)
   357: 
   358: 
   359: def load_needle_haystack_rows(max_examples: int = 0) -> list[dict]:
   360:     path = hf_dataset_file("opencompass/NeedleBench", "PaulGrahamEssays.jsonl")
   361:     with path.open() as fh:
   362:         return read_jsonl_lines(fh, max_examples)
   363: 
   364: 
   365: def first_string_field(row: dict, preferred: tuple[str, ...]) -> str:
   366:     for key in preferred:
   367:         value = row.get(key)
   368:         if isinstance(value, str) and value.strip():
   369:             return value
   370:     for value in row.values():
   371:         if isinstance(value, str) and value.strip():
   372:             return value
   373:     return ""
   374: 
   375: 
   376: def limit_reached(examples: list[dict], max_examples: int) -> bool:
   377:     return max_examples > 0 and len(examples) >= max_examples
   378: 
   379: 
   380: def build_longbench_examples(workload_name: str, max_examples: int) -> list[dict]:
   381:     dataset_name = WORKLOAD_CONFIGS[workload_name]["dataset_name"]
   382:     template = LONG_BENCH_TEMPLATES[dataset_name]
   383:     examples = []
   384:     for raw in load_longbench_rows(dataset_name, max_examples):
   385:         answers = raw.get("answers") or raw.get("outputs") or raw.get("answer") or []
   386:         if isinstance(answers, str):
   387:             answers = [answers]
   388:         examples.append(
   389:             {
   390:                 "example_id": raw.get("_id", f"{dataset_name}-{len(examples)}"),
   391:                 "dataset": dataset_name,
   392:                 "prompt": template.format(context=raw["context"], input=raw["input"]),
   393:                 "answers": [str(answer) for answer in answers],
   394:             }
   395:         )
   396:         if limit_reached(examples, max_examples):
   397:             return examples
   398:     return examples
   399: 
   400: 
   401: def build_needlebench_examples(max_examples: int) -> list[dict]:
   402:     haystacks = load_needle_haystack_rows()
   403:     haystack_texts = [
   404:         first_string_field(row, ("text", "content", "English", "english"))
   405:         for row in haystacks
   406:     ]
   407:     haystack_texts = [text for text in haystack_texts if text]
   408:     if not haystack_texts:
   409:         raise RuntimeError("Unable to build NIAH examples from opencompass/NeedleBench haystacks.")
   410: 
   411:     examples = []
   412:     for essay_idx, haystack in enumerate(haystack_texts):
   413:         for depth_idx, depth in enumerate(NEEDLE_DEPTHS):
   414:             split = int(len(haystack) * min(max(depth, 0.0), 1.0))
   415:             context = haystack[:split] + "\n" + NEEDLE_SENTENCE + "\n" + haystack[split:]
   416:             prompt = (
   417:                 "A single relevant sentence is hidden in the following long document. "
   418:                 "Read the document carefully and answer the retrieval question with the exact phrase.\n\n"
   419:                 f"{context}\n\nQuestion: {NEEDLE_QUESTION}\nAnswer:"
   420:             )
   421:             examples.append(
   422:                 {
   423:                     "example_id": f"niah-{essay_idx}-{depth:.2f}",
   424:                     "dataset": "needlebench_niah",
   425:                     "prompt": prompt,
   426:                     "answers": [NEEDLE_SENTENCE],
   427:                 }
   428:             )
   429:             if limit_reached(examples, max_examples):
   430:                 return examples
   431:     return examples
   432: 
   433: 
   434: def build_gsm8k_examples(max_examples: int) -> list[dict]:
   435:     try:
   436:         dataset = load_hf_dataset("openai/gsm8k", "main")
   437:     except Exception:
   438:         dataset = load_cached_gsm8k_test()
   439:     examples = []
   440:     for raw in dataset:
   441:         answer = extract_math_answer(raw.get("answer", ""))
   442:         prompt = (
   443:             "Solve the following grade-school math word problem carefully. "
   444:             "Show your reasoning and end with the final answer wrapped in \\\\boxed{}.\n\n"
   445:             f"Problem: {raw['question']}\n\nSolution:"
   446:         )
   447:         examples.append(
   448:             {
   449:                 "example_id": f"gsm8k-{len(examples)}",
   450:                 "dataset": "gsm8k",
   451:                 "prompt": prompt,
   452:                 "answers": [answer],
   453:             }
   454:         )
   455:         if limit_reached(examples, max_examples):
   456:             return examples
   457:     return examples
   458: 
   459: 
   460: def load_workload(name: str, max_examples: int = DEFAULT_MAX_EXAMPLES) -> WorkloadSpec:
   461:     if name.startswith("longbench_"):
   462:         examples = build_longbench_examples(name, max_examples)
   463:     elif name == "needlebench_niah":
   464:         examples = build_needlebench_examples(max_examples)
   465:     elif name == "gsm8k":
   466:         examples = build_gsm8k_examples(max_examples)
   467:     else:
   468:         raise ValueError(f"Unsupported workload: {name}")
   469:     return WorkloadSpec(
   470:         name=name,
   471:         source_family=WORKLOAD_CONFIGS[name]["source_family"],
   472:         examples=examples,
   473:         max_new_tokens=WORKLOAD_CONFIGS[name]["max_new_tokens"],
   474:     )
   475: 
   476: 
   477: WORKLOADS = WORKLOAD_CONFIGS
   478: 
   479: 
   480: def maybe_write_output_artifacts(workload: str, trace: dict, metrics: dict) -> None:
   481:     output_dir = os.environ.get("OUTPUT_DIR")
   482:     if not output_dir:
   483:         return
   484:     out_path = Path(output_dir)
   485:     out_path.mkdir(parents=True, exist_ok=True)
   486:     stem = workload.replace("-", "_")
   487:     (out_path / f"{stem}_trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
   488:     (out_path / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
   489: 
   490: 
   491: def snapshot_cache(cache: DynamicCache) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
   492:     snapshot = []
   493:     for layer_idx, layer_entry in enumerate(cache):
   494:         if len(layer_entry) == 2:
   495:             keys, values = layer_entry
   496:             sliding = None
   497:         elif len(layer_entry) == 3:
   498:             keys, values, sliding = layer_entry
   499:         else:
   500:             raise RuntimeError(

[truncated: showing at most 500 lines / 60000 bytes from transformers-kv-lab/custom_quant_eval.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `kivi_overlap_4bit` baseline — editable region  [READ-ONLY — reference implementation]

In `transformers-kv-lab/custom_quant_eval.py`:

```python
Lines 41–119:
    38:         return [example["prompt"] for example in self.examples]
    39: 
    40: 
    41: class AdaptiveKVQuantizer:
    42:     """KIVI K4/V4 axes with streaming residual behavior."""
    43: 
    44:     def __init__(self):
    45:         self.bits = 4
    46:         self.group_size = 32
    47:         self.key_residual_length = 128
    48:         self.value_residual_length = 128
    49: 
    50:     def reset_request(self, request_meta: dict, budget_state: dict):
    51:         self.bits = 4
    52: 
    53:     def needs_prefill_qkv_observer(self) -> bool:
    54:         return False
    55: 
    56:     def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
    57:         return None
    58: 
    59:     def query_observation_position(self) -> str:
    60:         return "post_rope"
    61: 
    62:     def _residual_keep_length(self, seq_len: int, residual_length: int, residual_policy: str) -> int:
    63:         residual_length = max(0, min(seq_len, int(residual_length)))
    64:         if residual_policy == "block_modulo":
    65:             return seq_len % residual_length if residual_length else 0
    66:         if residual_policy == "tail":
    67:             return residual_length
    68:         return 0
    69: 
    70:     def _minmax_last_dim(self, data: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    71:         if data.numel() == 0 or bits >= FP_BITS - 0.5:
    72:             return data
    73:         max_int = max(1, int(2**bits) - 1)
    74:         trailing = data.shape[-1]
    75:         group_size = trailing if int(group_size) <= 0 else int(group_size)
    76:         padded = math.ceil(trailing / group_size) * group_size
    77:         work = torch.nn.functional.pad(data, (0, padded - trailing)) if padded != trailing else data
    78:         grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
    79:         gmin = grouped.amin(dim=-1, keepdim=True)
    80:         gmax = grouped.amax(dim=-1, keepdim=True)
    81:         scale = (gmax - gmin).clamp(min=1e-5) / max_int
    82:         q = torch.round((grouped - gmin) / scale).clamp(0, max_int)
    83:         return q.mul(scale).add(gmin).reshape(*work.shape[:-1], padded)[..., :trailing]
    84: 
    85:     def _quantize(self, tensor: torch.Tensor, axis: str, residual_policy: str) -> tuple[torch.Tensor, float]:
    86:         work = tensor.float().clone()
    87:         batch, heads, seq_len, head_dim = work.shape
    88:         residual = self._residual_keep_length(seq_len, self.key_residual_length, residual_policy)
    89:         quant_end = seq_len - residual
    90:         if quant_end <= 0:
    91:             return work.to(tensor.dtype), FP_BITS
    92:         quant_slice = work[:, :, :quant_end, :]
    93:         if axis == "channel":
    94:             usable = quant_slice.shape[-2] - (quant_slice.shape[-2] % self.group_size)
    95:             if usable > 0:
    96:                 main = quant_slice[:, :, :usable, :].transpose(2, 3)
    97:                 main = main.reshape(batch, heads, head_dim, usable // self.group_size, self.group_size)
    98:                 main = self._minmax_last_dim(main, self.bits, self.group_size)
    99:                 work[:, :, :usable, :] = main.reshape(batch, heads, head_dim, usable).transpose(2, 3)
   100:             fp_tokens = residual + (quant_slice.shape[-2] - usable)
   101:             avg_bits = (usable * self.bits + fp_tokens * FP_BITS) / max(seq_len, 1)
   102:         else:
   103:             flat = quant_slice.transpose(1, 2).reshape(batch, quant_slice.shape[-2], heads * head_dim)
   104:             flat = self._minmax_last_dim(flat, self.bits, self.group_size)
   105:             work[:, :, :quant_end, :] = flat.reshape(batch, quant_slice.shape[-2], heads, head_dim).transpose(1, 2)
   106:             avg_bits = (quant_end * self.bits + residual * FP_BITS) / max(seq_len, 1)
   107:         return work.to(tensor.dtype), float(avg_bits)
   108: 
   109:     def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
   110:         return self._quantize(key_states, "channel", "block_modulo")
   111: 
   112:     def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
   113:         return self._quantize(value_states, "token", "tail")
   114: 
   115:     def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
   116:         policy = "block_modulo" if kv_kind == "key" else "tail"
   117:         residual = self._residual_keep_length(seq_len, self.key_residual_length, policy)
   118:         quant_tokens = max(0, seq_len - residual)
   119:         return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
   120: 
   121: 
   122: def resolve_task_dir() -> Path:
```

### `kvtuner4_pertoken_qwen25_3b` baseline — editable region  [READ-ONLY — reference implementation]

In `transformers-kv-lab/custom_quant_eval.py`:

```python
Lines 41–107:
    38:         return [example["prompt"] for example in self.examples]
    39: 
    40: 
    41: class AdaptiveKVQuantizer:
    42:     """KVTuner FlexibleVanillaQuantizedCache with official per-token preset."""
    43: 
    44:     _PRESET = {
    45:         0: {"key": 8, "value": 4}, 1: {"key": 4, "value": 2}, 2: {"key": 4, "value": 2},
    46:         3: {"key": 4, "value": 2}, 4: {"key": 4, "value": 2}, 5: {"key": 4, "value": 2},
    47:         6: {"key": 4, "value": 2}, 7: {"key": 4, "value": 2}, 8: {"key": 4, "value": 2},
    48:         9: {"key": 4, "value": 2}, 10: {"key": 4, "value": 4}, 11: {"key": 4, "value": 2},
    49:         12: {"key": 4, "value": 2}, 13: {"key": 4, "value": 2}, 14: {"key": 4, "value": 2},
    50:         15: {"key": 4, "value": 2}, 16: {"key": 4, "value": 2}, 17: {"key": 4, "value": 2},
    51:         18: {"key": 8, "value": 8}, 19: {"key": 4, "value": 4}, 20: {"key": 4, "value": 2},
    52:         21: {"key": 4, "value": 2}, 22: {"key": 4, "value": 2}, 23: {"key": 4, "value": 2},
    53:         24: {"key": 4, "value": 4}, 25: {"key": 4, "value": 2}, 26: {"key": 4, "value": 4},
    54:         27: {"key": 8, "value": 8}, 28: {"key": 4, "value": 2}, 29: {"key": 8, "value": 8},
    55:         30: {"key": 4, "value": 2}, 31: {"key": 4, "value": 2}, 32: {"key": 4, "value": 2},
    56:         33: {"key": 4, "value": 4}, 34: {"key": 4, "value": 2}, 35: {"key": 4, "value": 2},
    57:     }
    58: 
    59:     def reset_request(self, request_meta: dict, budget_state: dict):
    60:         return None
    61: 
    62:     def needs_prefill_qkv_observer(self) -> bool:
    63:         return False
    64: 
    65:     def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
    66:         return None
    67: 
    68:     def query_observation_position(self) -> str:
    69:         return "post_rope"
    70: 
    71:     def _signed_asymmetric(self, tensor: torch.Tensor, bits: int, axis: int, group_size: int, residual_length: int) -> tuple[torch.Tensor, float]:
    72:         work = tensor.float().clone()
    73:         _, _, seq_len, _ = work.shape
    74:         residual = max(0, min(seq_len, int(residual_length)))
    75:         quant_end = seq_len - residual
    76:         if quant_end <= 0 or bits >= FP_BITS - 0.5:
    77:             return work.to(tensor.dtype), FP_BITS
    78:         quant_slice = work[:, :, :quant_end, :]
    79:         shaped = quant_slice.transpose(-2, -1).contiguous() if axis == 1 else quant_slice
    80:         group_size = shaped.shape[-1] if int(group_size) == -1 else int(group_size)
    81:         original_shape = shaped.shape
    82:         trailing = shaped.shape[-1]
    83:         padded = math.ceil(trailing / group_size) * group_size
    84:         shaped = torch.nn.functional.pad(shaped, (0, padded - trailing)) if padded != trailing else shaped
    85:         rows = shaped.reshape(-1, group_size)
    86:         q_max, q_min = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    87:         max_vals = rows.max(dim=1).values
    88:         min_vals = rows.min(dim=1).values
    89:         scale = (max_vals - min_vals).clamp(min=1e-5) / (q_max - q_min)
    90:         zeros = (min_vals / scale).round() - q_min
    91:         quant = torch.round(rows / scale.unsqueeze(1) - zeros.unsqueeze(1)).clamp(q_min, q_max)
    92:         dequant = (quant + zeros.unsqueeze(1)) * scale.unsqueeze(1)
    93:         dequant = dequant.reshape(*original_shape[:-1], padded)[..., :trailing]
    94:         if axis == 1:
    95:             dequant = dequant.transpose(-2, -1).contiguous()
    96:         work[:, :, :quant_end, :] = dequant
    97:         avg_bits = (quant_end * bits + residual * FP_BITS) / max(seq_len, 1)
    98:         return work.to(tensor.dtype), float(avg_bits)
    99: 
   100:     def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
   101:         return self._signed_asymmetric(key_states, self._PRESET[layer_id]["key"], axis=0, group_size=-1, residual_length=0)
   102: 
   103:     def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
   104:         return self._signed_asymmetric(value_states, self._PRESET[layer_id]["value"], axis=0, group_size=-1, residual_length=0)
   105: 
   106:     def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
   107:         return float(self._PRESET[layer_id][kv_kind])
   108: 
   109: 
   110: def resolve_task_dir() -> Path:
```

### `kvtuner4_kivi_qwen25_3b` baseline — editable region  [READ-ONLY — reference implementation]

In `transformers-kv-lab/custom_quant_eval.py`:

```python
Lines 41–114:
    38:         return [example["prompt"] for example in self.examples]
    39: 
    40: 
    41: class AdaptiveKVQuantizer:
    42:     """KVTuner FlexibleVanillaQuantizedCache with official KIVI-style preset."""
    43: 
    44:     _PRESET = {
    45:         0: {"key": 4, "value": 8}, 1: {"key": 4, "value": 8}, 2: {"key": 2, "value": 4},
    46:         3: {"key": 4, "value": 2}, 4: {"key": 2, "value": 4}, 5: {"key": 4, "value": 2},
    47:         6: {"key": 4, "value": 2}, 7: {"key": 4, "value": 2}, 8: {"key": 4, "value": 2},
    48:         9: {"key": 4, "value": 2}, 10: {"key": 4, "value": 2}, 11: {"key": 4, "value": 2},
    49:         12: {"key": 2, "value": 2}, 13: {"key": 4, "value": 2}, 14: {"key": 4, "value": 2},
    50:         15: {"key": 4, "value": 2}, 16: {"key": 4, "value": 2}, 17: {"key": 4, "value": 2},
    51:         18: {"key": 4, "value": 2}, 19: {"key": 4, "value": 2}, 20: {"key": 4, "value": 2},
    52:         21: {"key": 4, "value": 2}, 22: {"key": 4, "value": 2}, 23: {"key": 4, "value": 2},
    53:         24: {"key": 4, "value": 2}, 25: {"key": 4, "value": 2}, 26: {"key": 4, "value": 2},
    54:         27: {"key": 4, "value": 2}, 28: {"key": 2, "value": 2}, 29: {"key": 4, "value": 2},
    55:         30: {"key": 4, "value": 2}, 31: {"key": 4, "value": 2}, 32: {"key": 4, "value": 2},
    56:         33: {"key": 4, "value": 2}, 34: {"key": 4, "value": 4}, 35: {"key": 4, "value": 4},
    57:     }
    58: 
    59:     def reset_request(self, request_meta: dict, budget_state: dict):
    60:         return None
    61: 
    62:     def needs_prefill_qkv_observer(self) -> bool:
    63:         return False
    64: 
    65:     def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
    66:         return None
    67: 
    68:     def query_observation_position(self) -> str:
    69:         return "post_rope"
    70: 
    71:     def _residual_keep_length(self, seq_len: int, residual_length: int) -> int:
    72:         residual_length = max(0, min(seq_len, int(residual_length)))
    73:         return seq_len % residual_length if residual_length else 0
    74: 
    75:     def _signed_asymmetric(self, tensor: torch.Tensor, bits: int, axis: int, group_size: int, residual_length: int) -> tuple[torch.Tensor, float]:
    76:         work = tensor.float().clone()
    77:         _, _, seq_len, _ = work.shape
    78:         residual = self._residual_keep_length(seq_len, residual_length)
    79:         quant_end = seq_len - residual
    80:         if quant_end <= 0 or bits >= FP_BITS - 0.5:
    81:             return work.to(tensor.dtype), FP_BITS
    82:         quant_slice = work[:, :, :quant_end, :]
    83:         shaped = quant_slice.transpose(-2, -1).contiguous() if axis == 1 else quant_slice
    84:         group_size = shaped.shape[-1] if int(group_size) == -1 else int(group_size)
    85:         original_shape = shaped.shape
    86:         trailing = shaped.shape[-1]
    87:         padded = math.ceil(trailing / group_size) * group_size
    88:         shaped = torch.nn.functional.pad(shaped, (0, padded - trailing)) if padded != trailing else shaped
    89:         rows = shaped.reshape(-1, group_size)
    90:         q_max, q_min = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    91:         max_vals = rows.max(dim=1).values
    92:         min_vals = rows.min(dim=1).values
    93:         scale = (max_vals - min_vals).clamp(min=1e-5) / (q_max - q_min)
    94:         zeros = (min_vals / scale).round() - q_min
    95:         quant = torch.round(rows / scale.unsqueeze(1) - zeros.unsqueeze(1)).clamp(q_min, q_max)
    96:         dequant = (quant + zeros.unsqueeze(1)) * scale.unsqueeze(1)
    97:         dequant = dequant.reshape(*original_shape[:-1], padded)[..., :trailing]
    98:         if axis == 1:
    99:             dequant = dequant.transpose(-2, -1).contiguous()
   100:         work[:, :, :quant_end, :] = dequant
   101:         avg_bits = (quant_end * bits + residual * FP_BITS) / max(seq_len, 1)
   102:         return work.to(tensor.dtype), float(avg_bits)
   103: 
   104:     def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
   105:         return self._signed_asymmetric(key_states, self._PRESET[layer_id]["key"], axis=1, group_size=32, residual_length=32)
   106: 
   107:     def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
   108:         return self._signed_asymmetric(value_states, self._PRESET[layer_id]["value"], axis=0, group_size=32, residual_length=32)
   109: 
   110:     def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
   111:         residual = self._residual_keep_length(seq_len, 32)
   112:         quant_tokens = max(0, seq_len - residual)
   113:         bits = self._PRESET[layer_id][kv_kind]
   114:         return float((quant_tokens * bits + residual * FP_BITS) / max(seq_len, 1))
   115: 
   116: 
   117: def resolve_task_dir() -> Path:
```

### `squat_subspace_4bit` baseline — editable region  [READ-ONLY — reference implementation]

In `transformers-kv-lab/custom_quant_eval.py`:

```python
Lines 41–170:
    38:         return [example["prompt"] for example in self.examples]
    39: 
    40: 
    41: class AdaptiveKVQuantizer:
    42:     """SQuat-inspired subspace-orthogonal K/V 4-bit quantization."""
    43: 
    44:     def __init__(self):
    45:         self.bits = 4
    46:         self.group_size = 32
    47:         self.residual_length = 32
    48:         self.subspace_dim = 60
    49:         self.squat_lambda = 0.001
    50:         self.quant_group_size = 64
    51:         self.shared_svd = True
    52:         self.query_subspaces = {}
    53: 
    54:     def reset_request(self, request_meta: dict, budget_state: dict):
    55:         self.query_subspaces = {}
    56: 
    57:     def needs_prefill_qkv_observer(self) -> bool:
    58:         return True
    59: 
    60:     def query_observation_position(self) -> str:
    61:         return "post_rope"
    62: 
    63:     def observe_prefill_qkv(self, layer_id, query_states, key_states, value_states, attention_meta):
    64:         if query_states is None:
    65:             return None
    66:         batch, query_heads, _, head_dim = query_states.shape
    67:         kv_heads = int(attention_meta.get("kv_heads", query_heads))
    68:         if query_heads % kv_heads != 0:
    69:             kv_heads = query_heads
    70:         matrix = query_states.reshape(batch, kv_heads, -1, head_dim).float()
    71:         rank = min(int(self.subspace_dim), matrix.shape[-2], matrix.shape[-1])
    72:         if rank <= 0:
    73:             return None
    74:         _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    75:         scaled_vh = torch.diag_embed(singular_values[:, :, :rank]).matmul(vh[:, :, :rank, :])
    76:         self.query_subspaces[layer_id] = (scaled_vh[0:1] if self.shared_svd else scaled_vh).detach()
    77:         return None
    78: 
    79:     def _residual_keep_length(self, seq_len: int) -> int:
    80:         residual_length = max(0, min(seq_len, int(self.residual_length)))
    81:         return seq_len % residual_length if residual_length else 0
    82: 
    83:     def _minmax_last_dim(self, data: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    84:         if data.numel() == 0 or bits >= FP_BITS - 0.5:
    85:             return data
    86:         max_int = max(1, 2**int(bits) - 1)
    87:         trailing = data.shape[-1]
    88:         group_size = trailing if int(group_size) <= 0 else int(group_size)
    89:         padded = math.ceil(trailing / group_size) * group_size
    90:         work = torch.nn.functional.pad(data, (0, padded - trailing)) if padded != trailing else data
    91:         grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
    92:         gmin = grouped.amin(dim=-1, keepdim=True)
    93:         gmax = grouped.amax(dim=-1, keepdim=True)
    94:         scale = (gmax - gmin).clamp(min=1e-5) / max_int
    95:         q = torch.round((grouped - gmin) / scale).clamp(0, max_int)
    96:         return q.mul(scale).add(gmin).reshape(*work.shape[:-1], padded)[..., :trailing]
    97: 
    98:     def _generate_At_inv(self, query_subspace: torch.Tensor, tol: float = 1e-7):
    99:         batch, heads, _, head_dim = query_subspace.shape
   100:         q_group = head_dim if int(self.quant_group_size) <= 0 else int(self.quant_group_size)
   101:         groups = math.ceil(head_dim / q_group)
   102:         matrices = [None] * groups
   103:         eye = torch.eye(head_dim, device=query_subspace.device, dtype=torch.float32)
   104:         A_t = eye.expand(batch, heads, head_dim, head_dim) + float(self.squat_lambda) * query_subspace.float().transpose(
   105:             -1, -2
   106:         ).matmul(query_subspace.float())
   107:         matrices[groups - 1] = A_t
   108:         for group_idx in range(groups - 1, 0, -1):
   109:             current_dim = group_idx * q_group
   110:             width = min(q_group, A_t.shape[-1] - current_dim)
   111:             M_t1 = A_t[:, :, :current_dim, :current_dim]
   112:             N_t1 = A_t[:, :, current_dim : current_dim + width, :current_dim]
   113:             O_t1 = A_t[:, :, current_dim : current_dim + width, current_dim : current_dim + width]
   114:             local_eye = torch.eye(width, device=query_subspace.device, dtype=torch.float32)
   115:             O_inv = torch.inverse(O_t1 + tol * local_eye.expand(batch, heads, width, width))
   116:             A_t = M_t1 - N_t1.transpose(-1, -2).matmul(O_inv.matmul(N_t1))
   117:             matrices[group_idx - 1] = A_t[:, :, :, -q_group:]
   118:         return matrices
   119: 
   120:     def _squat_quantize_keys(self, key_states: torch.Tensor, query_subspace: torch.Tensor) -> torch.Tensor:
   121:         batch, heads, _, head_dim = key_states.shape
   122:         query_subspace = query_subspace.to(device=key_states.device)
   123:         if query_subspace.shape[0] == 1 and batch > 1:
   124:             query_subspace = query_subspace.expand(batch, -1, -1, -1)
   125:         if query_subspace.shape[1] != heads or query_subspace.shape[-1] != head_dim:
   126:             raise ValueError("SQuat query subspace shape does not match the key tensor")
   127:         matrices = self._generate_At_inv(query_subspace)
   128:         P_inv = torch.inverse(matrices[-1])
   129:         work = key_states.float().clone()
   130:         q_group = head_dim if int(self.quant_group_size) <= 0 else int(self.quant_group_size)
   131:         groups = math.ceil(head_dim / q_group)
   132:         for group_idx in range(groups):
   133:             start = group_idx * q_group
   134:             end = min(head_dim, start + q_group)
   135:             chunk = work[:, :, :, start:end]
   136:             dequant = self._minmax_last_dim(chunk.transpose(2, 3).contiguous(), self.group_size, self.bits).transpose(2, 3)
   137:             if group_idx < groups - 1:
   138:                 d_vec = (dequant - chunk).float()
   139:                 next_start = end
   140:                 H_t = matrices[group_idx]
   141:                 B_t = P_inv[:, :, next_start:, :next_start]
   142:                 update = d_vec.matmul(H_t.transpose(-2, -1)).matmul(B_t.transpose(-2, -1))
   143:                 work[:, :, :, next_start:] = work[:, :, :, next_start:] + update
   144:             work[:, :, :, start:end] = dequant
   145:         return work
   146: 
   147:     def _quantize_with_residual(self, tensor: torch.Tensor, quant_fn) -> tuple[torch.Tensor, float]:
   148:         work = tensor.float().clone()
   149:         _, _, seq_len, _ = work.shape
   150:         residual = self._residual_keep_length(seq_len)
   151:         quant_end = seq_len - residual
   152:         if quant_end <= 0:
   153:             return work.to(tensor.dtype), FP_BITS
   154:         work[:, :, :quant_end, :] = quant_fn(work[:, :, :quant_end, :])
   155:         avg_bits = (quant_end * self.bits + residual * FP_BITS) / max(seq_len, 1)
   156:         return work.to(tensor.dtype), float(avg_bits)
   157: 
   158:     def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
   159:         query_subspace = self.query_subspaces.get(layer_id)
   160:         if query_subspace is None:
   161:             raise RuntimeError("SQuat key quantization requires the prefill query observer")
   162:         return self._quantize_with_residual(key_states, lambda data: self._squat_quantize_keys(data, query_subspace))
   163: 
   164:     def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
   165:         return self._quantize_with_residual(value_states, lambda data: self._minmax_last_dim(data, self.group_size, self.bits))
   166: 
   167:     def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
   168:         residual = self._residual_keep_length(seq_len)
   169:         quant_tokens = max(0, seq_len - residual)
   170:         return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))
   171: 
   172: 
   173: def resolve_task_dir() -> Path:
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
