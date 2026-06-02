# MLS-Bench: dlm-dkv-policy

# Diffusion LM KV Cache Policy

## Research Question

Design a cache policy for diffusion language-model inference. Given a fixed
LLaDA-8B-Instruct host model and public final-task benchmarks, can a method
preserve benchmark accuracy while reusing KV state during the bidirectional
denoising rollout?

## Background

LLaDA (Large Language Diffusion Models, Nie et al., 2025; arXiv:2502.09992)
predicts masked tokens with a Transformer that attends bidirectionally over
the entire sequence at every denoising step. Unlike autoregressive models,
the standard prefix-only KV cache is not directly reusable, because keys and
values for previously committed tokens can keep changing as more tokens are
unmasked. A growing line of work studies how to nonetheless reuse cached
features: dLLM-Cache (arXiv:2506.06295) refreshes prompt features on a long
interval and recomputes only low-similarity generated rows; d2Cache
(arXiv:2509.23094) uses a two-stage selection that combines an active query
mask with an attention rollout / certainty-density top-up to decide which
tokens to re-encode each step; and Elastic-Cache (Nguyen et al., ICLR 2026;
arXiv:2510.14973) uses an attention-aware drift test on the most-attended
token to decide when to refresh, plus a depth-aware schedule that recomputes
only deeper layers from a chosen layer onward.

The task isolates this design space onto a shared cache-control surface so a
policy can be evaluated end-to-end on real LLaDA generation rather than on a
proxy token-trajectory metric.

## Evaluation Setup

The harness runs real `LLaDA-8B-Instruct` inference end to end. For each
workload it:

1. Loads the public benchmark dataset.
2. Runs one fixed denoising rollout with a shared cache-plan interface.
3. Generates deterministic outputs using the submitted cache policy.
4. Scores the generated outputs with benchmark-native final-task metrics.
5. Emits the benchmark-native final score.

The task is not a backend-selection problem: paper baselines are implemented
through the same cache-control surface rather than called as black-box
generation backends. Some cache mechanisms require additional LLaDA forward
arguments such as active query rows or tracked-token positions; the harness
may load task-local compatibility model classes to expose those forward
hooks, but the outer rollout remains policy-driven and does not call paper
repository generation functions.

## Editable Surface

You may edit only the policy class in `dLLM-cache/custom_dlm_eval.py`. The
compatibility class name is `DLMRefreshPolicy`, but semantically it is a DLM
cache-plan policy.

The required hook families are:

| Method | Purpose |
|---|---|
| `block_schedule(request_meta)` | Controls generation length, block length, steps per block, and whether a block starts with a full warm forward. |
| `query_plan(step_meta, mask_state, cache_state)` | Selects token positions to forward or recompute: full sequence, current block, active query rows, tracked tokens, or a masked query window. |
| `cache_refresh_plan(layer_meta, step_meta, token_stats, cache_state)` | Decides per-layer recompute/reuse, prompt-vs-generation refresh, selected row refresh, KV overwrite, and layer reset. |
| `attention_probe_plan(layer_meta, step_meta)` | Requests attention weights or attention-similarity probes and supplies parameters such as rollout fraction, `current_k`, `gamma`, and `track_num`. |
| `token_transfer_plan(logits, mask_state, step_meta)` | Chooses which masked tokens are committed back to the global denoising state. |
| `after_step(step_meta, logits, attention_stats, transfer_state, cache_state)` | Updates state such as active query masks, attention rollout, tracked tokens, density scores, and layer reset boundaries. |

The full hook contract and baseline mapping are recorded in
`CACHE_HOOK_CONTRACT.md`.

## Fixed Components

Participants may not modify:

- the model weights or tokenizer
- benchmark loaders and scorers
- task scripts, parser, score spec, or leaderboard schema
- source-reference snapshots under `third_party/official_dlm_cache_baselines`
- any harness code outside the editable policy region

Each baseline uses one predeclared cache policy across all workloads to
avoid rewarding per-benchmark hyperparameter search.

## Workloads

| Label | Workload | Public source | Final metric |
|---|---|---|---|
| `math` | MATH-500 test split | exact final-answer accuracy |
| `humaneval` | OpenAI HumanEval | pass@1 execution accuracy |
| `lm-eval` | ARC-Challenge test split | exact answer-letter accuracy |

All examples in the selected public splits are evaluated by default.

## Metrics

Each script prints one `TEST_METRICS:` line. The parser records the benchmark
score and runtime diagnostics:

| Metric | Direction | Meaning |
|---|---|---|
| `final_score` | higher | benchmark-native final task score on a 0-100 scale |
| `reuse_ratio` | higher | diagnostic fraction of generated-token cache work reused by the hook plan |
| `refresh_ratio` | lower | diagnostic `1 - reuse_ratio` |
| `tokens_per_s` | higher | diagnostic decode throughput on the current hardware |
| `peak_memory_mb` | lower | diagnostic peak GPU memory allocated during the example loop |
| `n_examples` | fixed | number of examples evaluated |
| `elapsed` | lower | diagnostic wall-clock time recorded by the harness for the script |

`final_score` is the canonical quality metric. `reuse_ratio` and
`tokens_per_s` enter the scalar ranking because the task is a cache-policy
benchmark: methods should preserve final-task quality while reducing
redundant denoising work and improving decode throughput.

## Canonical Ranking

The score in `score_spec.py` follows the MLS-Bench efficiency-task pattern:

- each workload applies `final_score_*` as a near-lossless soft quality gate
- once the quality gate is satisfied, small benchmark-native score
  differences are not rewarded further
- each workload ranks cache reuse and decode throughput as efficiency terms
- throughput is normalized against the visible baseline envelope rather than
  a hard hardware-specific pass/fail range
- the task score is the geometric mean across the three workloads

## Baselines

| Baseline | Source |
|---|---|
| `vanilla_uncached` | no-cache LLaDA control: full denoising forward every step |
| `dllm_cache` | dLLM-Cache (arXiv:2506.06295), `maomaocun/dLLM-cache`: prompt/generation feature refresh and low-similarity generated-row update |
| `d2cache` | d2Cache (arXiv:2509.23094), `Kamichanw/d2Cache`: active query mask, eager attention rollout, and certainty-density top-up |
| `elastic_cache` | Elastic-Cache (arXiv:2510.14973), `VILA-Lab/Elastic-Cache`: tracked-token query window and attention-similarity layer reset |

Task-local source snapshots and commit hashes are documented in
`third_party/official_dlm_cache_baselines/NOTICE.md`.

Small compatibility shims may be used for model loading and source-oracle
checks when a hook requires extra forward arguments. They are capability
adapters, not participant-facing backend choices; canonical task behavior
must still be explained in terms of the shared hook contract.


## Your Workspace

You are working inside `/workspace`. The package source tree
`/workspace/dLLM-cache/` is the research scaffold for this task.

## Files You May Edit

You may **only** modify these files, and **only within the listed line ranges
(inclusive, 1-indexed)**. Edits outside these ranges — or creating new files,
or deleting existing ones — will cause your submission to be invalid.

- `dLLM-cache/custom_dlm_eval.py`
- editable lines **53–113**




## Readable Context


### `dLLM-cache/custom_dlm_eval.py`  [EDITABLE — lines 53–113 only]

```python
     1: """Real LLaDA rollout harness for dlm-dkv-policy."""
     2: 
     3: from __future__ import annotations
     4: 
     5: import argparse
     6: import ast
     7: import importlib
     8: import json
     9: import os
    10: import re
    11: import subprocess
    12: import sys
    13: import tempfile
    14: import time
    15: import types
    16: from pathlib import Path
    17: 
    18: import torch
    19: 
    20: # dLLM-cache path resolution
    21: _HERE = Path(__file__).resolve().parent
    22: 
    23: def _find_dllm_cache_root() -> Path:
    24:     env = os.environ.get('DLLM_CACHE_DIR', '')
    25:     if env and (Path(env) / 'dllm_cache').is_dir():
    26:         return Path(env)
    27:     for p in [Path.cwd(), _HERE, _HERE.parent]:
    28:         if (p / 'dllm_cache').is_dir():
    29:             return p
    30:     raise RuntimeError('dLLM-cache not found. Set DLLM_CACHE_DIR env var.')
    31: 
    32: _DLLM_ROOT = _find_dllm_cache_root()
    33: if str(_DLLM_ROOT) not in sys.path:
    34:     sys.path.insert(0, str(_DLLM_ROOT))
    35: _MODEL_DIR = os.environ.get('LLADA_MODEL_DIR', 'LLaDA-8B-Instruct')
    36: _TASK_SLUG = 'dlm-dkv-policy'
    37: 
    38: REGIMES = {"final": {}}
    39: 
    40: WORKLOAD_CONFIGS = {
    41:     "math":      {"source_family": "MATH-500",      "num_steps": 256, "gen_length": 256, "block_length": 32},
    42:     "humaneval": {"source_family": "HumanEval",     "num_steps": 512, "gen_length": 512, "block_length": 32},
    43:     "lm_eval":   {"source_family": "ARC-Challenge", "num_steps": 64,  "gen_length": 64,  "block_length": 32},
    44: }
    45: 
    46: # Editable region: DLMRefreshPolicy. config.json must match this marker-delimited
    47: # span; baseline edit files fail fast if the configured window drifts.
    48: MASK_ID = 126336  # LLaDA mask token id
    49: 
    50: 
    51: 
    52: 
    53: class DLMRefreshPolicy:
    54:     """Default shared-hook policy: uncached LLaDA denoising rollout.
    55: 
    56:     The participant-facing surface is a cache-plan interface over one fixed
    57:     rollout, not a selector for paper-specific backend modules.
    58:     """
    59: 
    60:     policy_name = "vanilla_uncached"
    61: 
    62:     def block_schedule(self, request_meta):
    63:         wl = WORKLOAD_CONFIGS[request_meta["workload"]]
    64:         return {
    65:             "gen_length": wl["gen_length"],
    66:             "block_length": wl["block_length"],
    67:             "num_steps": wl["num_steps"],
    68:             "warmup_forward": False,
    69:         }
    70: 
    71:     def query_plan(self, step_meta, mask_state, cache_state):
    72:         return {
    73:             "query_scope": "full_sequence",
    74:             "query_positions": None,
    75:             "track_positions": [],
    76:             "masked_window": None,
    77:         }
    78: 
    79:     def cache_refresh_plan(self, layer_meta, step_meta, token_stats, cache_state):
    80:         return {
    81:             "use_feature_cache": False,
    82:             "prompt_refresh_interval": 1,
    83:             "gen_refresh_interval": 1,
    84:             "transfer_ratio": 0.0,
    85:             "row_selector": "none",
    86:             "kv_update": "full_refresh",
    87:             "layer_reset": None,
    88:         }
    89: 
    90:     def attention_probe_plan(self, layer_meta, step_meta):
    91:         return {
    92:             "need_attention_weights": False,
    93:             "rollout_p": 0.0,
    94:             "current_k": 0,
    95:             "gamma": None,
    96:             "track_num": 0,
    97:         }
    98: 
    99:     def token_transfer_plan(self, logits, mask_state, step_meta):
   100:         return {
   101:             "mode": "low_confidence",
   102:             "scope": "current_block",
   103:             "num_transfer_tokens": step_meta["default_num_transfer_tokens"],
   104:             "threshold": None,
   105:             "force_one": True,
   106:         }
   107: 
   108:     def after_step(self, step_meta, logits, attention_stats, transfer_state, cache_state):
   109:         return cache_state
   110: 
   111: 
   112: 
   113: 
   114: # end of editable region – evaluation code starts below.
   115: 
   116: 
   117: 
   118: 
   119: # ---------------------------------------------------------------------------
   120: # Evaluation infrastructure (do not edit below this line)
   121: # ---------------------------------------------------------------------------
   122: 
   123: 
   124: def _resolve_model_dir(configured: str) -> Path:
   125:     candidates = [Path(configured)] if configured else []
   126:     seen = set()
   127:     for candidate in candidates:
   128:         if not str(candidate):
   129:             continue
   130:         resolved = candidate.resolve()
   131:         if resolved in seen:
   132:             continue
   133:         seen.add(resolved)
   134:         has_config = (resolved / 'config.json').exists()
   135:         has_weights = (resolved / 'model.safetensors').exists() or any(resolved.glob('model-*.safetensors'))
   136:         if has_config and has_weights:
   137:             return resolved
   138:     return Path(configured)
   139: 
   140: 
   141: def _task_dir() -> Path | None:
   142:     raw = os.environ.get("MLSBENCH_TASK_DIR", "")
   143:     if not raw:
   144:         return None
   145:     path = Path(raw)
   146:     return path if path.exists() else None
   147: 
   148: 
   149: def _policy_initial_plan(policy: "DLMRefreshPolicy", workload_name: str = "math") -> tuple[dict, dict]:
   150:     request_meta = {"workload": workload_name, "step_budget": "final"}
   151:     schedule = dict(policy.block_schedule(request_meta))
   152:     step_meta = {
   153:         "step": 0,
   154:         "step_in_block": 0,
   155:         "block": 0,
   156:         "workload": workload_name,
   157:         "regime": "final",
   158:         "total_steps": _clamp_int(schedule.get("num_steps"), 1, WORKLOAD_CONFIGS[workload_name]["num_steps"]),
   159:         "prompt_len": 0,
   160:         "gen_length": _clamp_int(schedule.get("gen_length"), 1, WORKLOAD_CONFIGS[workload_name]["gen_length"]),
   161:         "block_length": _clamp_int(schedule.get("block_length"), 1, WORKLOAD_CONFIGS[workload_name]["block_length"]),
   162:         "default_num_transfer_tokens": 1,
   163:     }
   164:     refresh_plan = dict(policy.cache_refresh_plan(
   165:         {"layer_id": -1, "segment": "all"}, step_meta, [], {},
   166:     ))
   167:     return schedule, refresh_plan
   168: 
   169: 
   170: def _needs_active_query_model(policy: "DLMRefreshPolicy") -> bool:
   171:     _, refresh_plan = _policy_initial_plan(policy)
   172:     return refresh_plan.get("kv_update") == "active_q_mask"
   173: 
   174: 
   175: def _needs_tracked_window_model(policy: "DLMRefreshPolicy") -> bool:
   176:     _, refresh_plan = _policy_initial_plan(policy)
   177:     return refresh_plan.get("kv_update") == "tracked_window_layer_reset" or refresh_plan.get("layer_reset") == "attention_similarity"
   178: 
   179: 
   180: def _load_llada_class(package_name: str, package_path: Path):
   181:     runtime_pkg = sys.modules.get(package_name)
   182:     if runtime_pkg is None:
   183:         runtime_pkg = types.ModuleType(package_name)
   184:         runtime_pkg.__path__ = [str(package_path)]
   185:         runtime_pkg.__file__ = str(package_path / "__init__.py")
   186:         sys.modules[package_name] = runtime_pkg
   187:     importlib.invalidate_caches()
   188:     modeling_llada = importlib.import_module(f"{package_name}.modeling_llada")
   189:     llada_cls = modeling_llada.LLaDAModelLM
   190:     llada_cls._tied_weights_keys = {
   191:         "model.transformer.ff_out.weight": "model.transformer.wte.weight"
   192:     }
   193:     return llada_cls
   194: 
   195: 
   196: def _install_d2_cache_import_stub() -> None:
   197:     """Avoid importing d2Cache's CLI/frame stack when only model hooks are needed."""
   198: 
   199:     if "src.cache" in sys.modules:
   200:         return
   201: 
   202:     class _NullD2Cache:
   203:         def __init__(self, model_config):
   204:             self.model_config = model_config
   205: 
   206:         def model_forward(self, x):
   207:             return _D2ContextManager(_D2ModelForwardContext(x))
   208: 
   209:         def attention(self, layer_idx, x, attn_norm, q_proj, k_proj, v_proj, attention_mask=None, position_ids=None):
   210:             normed = attn_norm(x)
   211:             return _D2ContextManager(_D2AttentionContext(
   212:                 q=q_proj(normed),
   213:                 k=k_proj(normed),
   214:                 v=v_proj(normed),
   215:                 residual=x,
   216:                 attention_mask=attention_mask,
   217:                 q_position_ids=position_ids,
   218:                 kv_position_ids=position_ids,
   219:             ))
   220: 
   221:         def ffn(self, layer_idx, x):
   222:             return _D2ContextManager(_D2FFNContext(x))
   223: 
   224:     cache_module = types.ModuleType("src.cache")
   225:     cache_module.dCache = _NullD2Cache
   226:     cache_module.d2Cache = _NullD2Cache
   227:     sys.modules["src.cache"] = cache_module
   228: 
   229: 
   230: def _install_minimal_einops() -> None:
   231:     """Provide the two rearrange patterns used by the Elastic LLaDA adapter."""
   232: 
   233:     if "einops" in sys.modules or importlib.util.find_spec("einops") is not None:
   234:         return
   235: 
   236:     einops_module = types.ModuleType("einops")
   237: 
   238:     def rearrange(x, pattern: str, **axes):
   239:         normalized = " ".join(pattern.split())
   240:         if normalized == "b s three h d -> b h three s d":
   241:             return x.permute(0, 3, 2, 1, 4).contiguous()
   242:         if normalized == "b h s d -> b s (h d)":
   243:             bsz, heads, seq, dim = x.shape
   244:             return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq, heads * dim)
   245:         raise NotImplementedError(f"Unsupported fallback einops pattern: {pattern}")
   246: 
   247:     einops_module.rearrange = rearrange
   248:     sys.modules["einops"] = einops_module
   249: 
   250: 
   251: def _load_d2_llada_class(task_dir: Path):
   252:     source_root = task_dir / "third_party" / "official_dlm_cache_baselines" / "d2cache"
   253:     if not (source_root / "src" / "models" / "llada" / "modeling_llada.py").exists():
   254:         raise FileNotFoundError(f"Active-query LLaDA model class not found under {source_root}")
   255:     if str(source_root) not in sys.path:
   256:         sys.path.insert(0, str(source_root))
   257:     _install_d2_cache_import_stub()
   258:     importlib.invalidate_caches()
   259:     modeling_llada = importlib.import_module("src.models.llada.modeling_llada")
   260:     modeling_llada.d2Cache = _SharedD2Cache
   261:     llada_cls = modeling_llada.LLaDAModelLM
   262:     if not getattr(llada_cls, "_mlsbench_d2_hf_compat", False):
   263:         orig_tie_weights = llada_cls.tie_weights
   264: 
   265:         def compat_tie_weights(self, *args, **kwargs):
   266:             return orig_tie_weights(self)
   267: 
   268:         llada_cls.tie_weights = compat_tie_weights
   269:         llada_cls._mlsbench_d2_hf_compat = True
   270:     return llada_cls
   271: 
   272: 
   273: def _load_elastic_llada_class(task_dir: Path):
   274:     class_dir = task_dir / "third_party" / "official_dlm_cache_baselines" / "elastic_cache" / "llada" / "model"
   275:     if not (class_dir / "modeling_llada.py").exists():
   276:         raise FileNotFoundError(f"Tracked-window LLaDA model class not found under {class_dir}")
   277:     _install_minimal_einops()
   278:     return _load_llada_class("_mlsbench_llada_tracked_window", class_dir)
   279: 
   280: 
   281: def _patch_hf_compat(llada_cls) -> None:
   282:     if getattr(llada_cls, "_mlsbench_hf_compat", False):
   283:         return
   284:     orig_init = llada_cls.__init__
   285:     orig_tie_weights = llada_cls.tie_weights
   286: 
   287:     def compat_init(self, config, model=None, init_params=False):
   288:         if not hasattr(config, "use_cache"):
   289:             config.use_cache = False
   290:         if not hasattr(config, "return_dict"):
   291:             config.return_dict = True
   292:         orig_init(self, config, model=model, init_params=init_params)
   293:         if hasattr(self, "get_expanded_tied_weights_keys"):
   294:             self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
   295:         else:
   296:             self.all_tied_weights_keys = dict(getattr(self, "_tied_weights_keys", {}))
   297: 
   298:     def compat_tie_weights(self, missing_keys=None, recompute_mapping=True):
   299:         result = orig_tie_weights(self)
   300:         if hasattr(self, "get_expanded_tied_weights_keys"):
   301:             self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)
   302:         else:
   303:             self.all_tied_weights_keys = dict(getattr(self, "_tied_weights_keys", {}))
   304:         return result
   305: 
   306:     llada_cls.__init__ = compat_init
   307:     llada_cls.tie_weights = compat_tie_weights
   308:     llada_cls._mlsbench_hf_compat = True
   309: 
   310: 
   311: def _load_model_and_tokenizer(policy: "DLMRefreshPolicy" | None = None):
   312:     model_dir = _resolve_model_dir(_MODEL_DIR)
   313:     if str(model_dir) not in sys.path:
   314:         sys.path.insert(0, str(model_dir))
   315:     from transformers import AutoTokenizer
   316:     if not model_dir.exists():
   317:         raise FileNotFoundError(
   318:             f'LLaDA model not found: {_MODEL_DIR}. '
   319:             f'Set LLADA_MODEL_DIR to a prepared GSAI-ML/LLaDA-8B-Instruct directory.'
   320:         )
   321:     if policy is not None and _needs_active_query_model(policy):
   322:         task_dir = _task_dir()
   323:         if task_dir is None:
   324:             raise FileNotFoundError("MLSBENCH_TASK_DIR is required for active-query model compatibility files.")
   325:         llada_cls = _load_d2_llada_class(task_dir)
   326:     elif policy is not None and _needs_tracked_window_model(policy):
   327:         task_dir = _task_dir()
   328:         if task_dir is None:
   329:             raise FileNotFoundError("MLSBENCH_TASK_DIR is required for tracked-window model compatibility files.")
   330:         llada_cls = _load_elastic_llada_class(task_dir)
   331:     else:
   332:         llada_cls = _load_llada_class("_mlsbench_llada_runtime", model_dir)
   333:     if not _needs_active_query_model(policy) if policy is not None else True:
   334:         _patch_hf_compat(llada_cls)
   335: 
   336:     device = 'cuda' if torch.cuda.is_available() else 'cpu'
   337:     tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
   338:     model = llada_cls.from_pretrained(
   339:         str(model_dir), trust_remote_code=True, torch_dtype=torch.bfloat16
   340:     ).to(device).eval()
   341:     if not hasattr(model.config, "use_cache"):
   342:         model.config.use_cache = False
   343:     if not hasattr(model.config, "return_dict"):
   344:         model.config.return_dict = True
   345:     return model, tokenizer, device
   346: 
   347: 
   348: class _FeatureCacheRuntime:
   349:     """Shared runtime adapter for feature-level cache reuse hooks."""
   350: 
   351:     def __init__(self, model):
   352:         from dataclasses import asdict
   353:         from dllm_cache.cache import dLLMCache, dLLMCacheConfig
   354:         from dllm_cache.hooks import register_cache_LLaDA, logout_cache_LLaDA
   355: 
   356:         self.model = model
   357:         self._cache_cls = dLLMCache
   358:         self._config_cls = dLLMCacheConfig
   359:         self._asdict = asdict
   360:         self._register = register_cache_LLaDA
   361:         self._logout = logout_cache_LLaDA
   362:         self.cache = None
   363: 
   364:     def disable(self) -> None:
   365:         self._logout(self.model, 'model.transformer.blocks')
   366:         self.cache = None
   367: 
   368:     def enable(self, gen_interval: int, prompt_interval: int, transfer: float, prompt_len: int):
   369:         self.disable()
   370:         self._cache_cls.new_instance(**self._asdict(self._config_cls(
   371:             prompt_interval_steps=max(1, prompt_interval),
   372:             gen_interval_steps=max(1, gen_interval),
   373:             transfer_ratio=float(transfer),
   374:         )))
   375:         self._register(self.model, 'model.transformer.blocks')
   376:         self.cache = self._cache_cls()
   377:         self.cache.reset_cache(prompt_len)
   378:         return self.cache
   379: 
   380:     def update(self, gen_interval: int, prompt_interval: int, transfer: float) -> None:
   381:         if self.cache is None:
   382:             return
   383:         self.cache.gen_interval_steps = max(1, gen_interval)
   384:         self.cache.prompt_interval_steps = max(1, prompt_interval)
   385:         self.cache.transfer_ratio = float(transfer)
   386: 
   387: 
   388: class _D2ModelForwardContext:
   389:     def __init__(self, x: torch.Tensor):
   390:         self.x = x
   391:         self.logits = None
   392: 
   393: 
   394: class _D2AttentionContext:
   395:     def __init__(self, q, k, v, residual, attention_mask=None, q_position_ids=None, kv_position_ids=None):
   396:         self.q = q
   397:         self.k = k
   398:         self.v = v
   399:         self.residual = residual
   400:         self.attention_mask = attention_mask
   401:         self.q_position_ids = q_position_ids
   402:         self.kv_position_ids = kv_position_ids
   403:         self.o = None
   404:         self.attn_weight = None
   405: 
   406: 
   407: class _D2FFNContext:
   408:     def __init__(self, x: torch.Tensor):
   409:         self.x = x
   410:         self.residual = x
   411:         self.ffn_out = None
   412: 
   413: 
   414: class _D2ContextManager:
   415:     def __init__(self, context, after=None):
   416:         self.context = context
   417:         self.after = after
   418: 
   419:     def __enter__(self):
   420:         return self.context
   421: 
   422:     def __exit__(self, exc_type, exc, tb):
   423:         if exc_type is None and self.after is not None:
   424:             self.after(self.context)
   425:         return False
   426: 
   427: 
   428: def _d2_convert_attention_mask(attention_mask, dtype, query_length=None, key_value_length=None):
   429:     if attention_mask is None:
   430:         return None
   431:     if attention_mask.dim() == 2:
   432:         attention_mask = attention_mask[:, None, None, :].expand(
   433:             attention_mask.size(0),
   434:             1,
   435:             query_length or attention_mask.size(1),
   436:             key_value_length or attention_mask.size(1),
   437:         )
   438:     elif attention_mask.dim() != 4:
   439:         raise ValueError(f"Expected attention_mask rank 2 or 4, got {attention_mask.dim()}.")
   440:     return (1.0 - attention_mask.to(dtype)) * torch.finfo(dtype).min
   441: 
   442: 
   443: def _d2_select_position_ids(position_ids=None, q_mask=None, kv_mask=None):
   444:     q_position_ids, kv_position_ids = position_ids, position_ids
   445:     if position_ids is not None:
   446:         if q_mask is not None:
   447:             q_position_ids = position_ids[q_mask].view(q_mask.size(0), -1)
   448:         if kv_mask is not None:
   449:             kv_position_ids = position_ids[kv_mask].view(kv_mask.size(0), -1)
   450:     return q_position_ids, kv_position_ids
   451: 
   452: 
   453: def _d2_certainty_density(mask: torch.Tensor, sigma: float) -> torch.Tensor:
   454:     assert sigma > 0
   455:     batch, length = mask.shape
   456:     device = mask.device
   457:     float_mask = mask.float()
   458:     padded_mask = torch.nn.functional.pad(float_mask, (length, length), "constant", 1.0)
   459:     padded_mask[mask[:, -1] == False, 2 * length:] = 0.0
   460:     extended_len = 3 * length
   461:     padded_len = 2 * extended_len
   462:     dist = torch.cat((
   463:         torch.arange(extended_len, device=device),
   464:         torch.arange(-extended_len, 0, device=device),
   465:     ))
   466:     kernel_fft = torch.fft.fft(torch.exp(-(dist**2) / (2 * sigma**2)), n=padded_len)
   467:     weighted_sum_ext = torch.fft.ifft(
   468:         torch.fft.fft(torch.nn.functional.pad(padded_mask, (0, extended_len)), n=padded_len) * kernel_fft,
   469:         n=padded_len,
   470:     ).real
   471:     kernel_sum_ext = torch.fft.ifft(
   472:         torch.fft.fft(torch.ones(batch, extended_len * 2, device=device), n=padded_len) * kernel_fft,
   473:         n=padded_len,
   474:     ).real
   475:     return weighted_sum_ext[..., length:2 * length] / kernel_sum_ext[..., length:2 * length].clamp_min(1e-8)
   476: 
   477: 
   478: def _d2_nucleus_select(scores: torch.Tensor, top_p: float, min_k: int = 1, mask: torch.Tensor | None = None):
   479:     scores = torch.where(mask, scores, 0.0) if mask is not None else scores
   480:     probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
   481:     sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
   482:     cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
   483:     nucleus_mask = cumulative_probs <= top_p
   484:     top_k_mask = torch.arange(nucleus_mask.shape[-1], device=nucleus_mask.device) < min(min_k, scores.shape[-1])
   485:     combined_mask = nucleus_mask | top_k_mask
   486:     if mask is not None:
   487:         combined_mask &= torch.gather(mask, 1, sorted_indices)
   488:     return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, sorted_indices, combined_mask)
   489: 
   490: 
   491: def _d2_top_up_mask_(mask: torch.Tensor, target_count: int, scores: torch.Tensor):
   492:     num_selected = mask.sum(dim=-1)
   493:     num_to_pad = (target_count - num_selected).clamp(min=0)
   494:     if num_to_pad.sum() == 0:
   495:         return mask
   496:     max_pad = int(num_to_pad.max())
   497:     ranked_scores = torch.where(mask, -torch.inf, scores)
   498:     _, indices = torch.topk(ranked_scores, k=max_pad, dim=-1)
   499:     pad_indices = indices.masked_select(
   500:         torch.arange(max_pad, device=mask.device).expand(mask.shape[0], -1) < num_to_pad.unsqueeze(-1)

[truncated: showing at most 500 lines / 60000 bytes from dLLM-cache/custom_dlm_eval.py]
```

## Reference Baselines

The following are **read-only** reference implementations. Each shows what
the editable region of a strong baseline looks like, with a few lines of
surrounding context for orientation. Study them, but write your own
algorithm — repeating a baseline verbatim will be detected and scored as
a baseline reproduction.


### `vanilla_uncached` baseline — editable region  [READ-ONLY — reference implementation]

In `dLLM-cache/custom_dlm_eval.py`:

```python
Lines 53–105:
    50: 
    51: 
    52: 
    53: class DLMRefreshPolicy:
    54:     """No-cache control: full LLaDA denoising forward every step."""
    55: 
    56:     policy_name = "vanilla_uncached"
    57: 
    58:     def block_schedule(self, request_meta):
    59:         wl = WORKLOAD_CONFIGS[request_meta["workload"]]
    60:         return {
    61:             "gen_length": wl["gen_length"],
    62:             "block_length": wl["block_length"],
    63:             "num_steps": wl["num_steps"],
    64:             "warmup_forward": False,
    65:         }
    66: 
    67:     def query_plan(self, step_meta, mask_state, cache_state):
    68:         return {
    69:             "query_scope": "full_sequence",
    70:             "query_positions": None,
    71:             "track_positions": [],
    72:             "masked_window": None,
    73:         }
    74: 
    75:     def cache_refresh_plan(self, layer_meta, step_meta, token_stats, cache_state):
    76:         return {
    77:             "use_feature_cache": False,
    78:             "prompt_refresh_interval": 1,
    79:             "gen_refresh_interval": 1,
    80:             "transfer_ratio": 0.0,
    81:             "row_selector": "none",
    82:             "kv_update": "full_refresh",
    83:             "layer_reset": None,
    84:         }
    85: 
    86:     def attention_probe_plan(self, layer_meta, step_meta):
    87:         return {
    88:             "need_attention_weights": False,
    89:             "rollout_p": 0.0,
    90:             "current_k": 0,
    91:             "gamma": None,
    92:             "track_num": 0,
    93:         }
    94: 
    95:     def token_transfer_plan(self, logits, mask_state, step_meta):
    96:         return {
    97:             "mode": "low_confidence",
    98:             "scope": "current_block",
    99:             "num_transfer_tokens": step_meta["default_num_transfer_tokens"],
   100:             "threshold": None,
   101:             "force_one": True,
   102:         }
   103: 
   104:     def after_step(self, step_meta, logits, attention_stats, transfer_state, cache_state):
   105:         return cache_state
   106: # end of editable region – evaluation code starts below.
   107: 
   108: 
```

### `dllm_cache` baseline — editable region  [READ-ONLY — reference implementation]

In `dLLM-cache/custom_dlm_eval.py`:

```python
Lines 53–115:
    50: 
    51: 
    52: 
    53: class DLMRefreshPolicy:
    54:     """dLLM-Cache: interval feature reuse plus low-similarity row refresh."""
    55: 
    56:     policy_name = "dllm_cache"
    57:     _PRESETS = {
    58:         "math": (50, 7, 0.25),
    59:         "humaneval": (50, 8, 0.25),
    60:         "lm_eval": (50, 3, 0.25),
    61:     }
    62: 
    63:     def _preset(self, request_meta):
    64:         return self._PRESETS.get(request_meta["workload"], (50, 4, 0.25))
    65: 
    66:     def block_schedule(self, request_meta):
    67:         wl = WORKLOAD_CONFIGS[request_meta["workload"]]
    68:         return {
    69:             "gen_length": wl["gen_length"],
    70:             "block_length": wl["block_length"],
    71:             "num_steps": wl["num_steps"],
    72:             "warmup_forward": False,
    73:         }
    74: 
    75:     def query_plan(self, step_meta, mask_state, cache_state):
    76:         return {
    77:             "query_scope": "full_sequence",
    78:             "query_positions": None,
    79:             "track_positions": [],
    80:             "masked_window": None,
    81:         }
    82: 
    83:     def cache_refresh_plan(self, layer_meta, step_meta, token_stats, cache_state):
    84:         request_meta = {"workload": step_meta.get("workload", ""), "step_budget": "final"}
    85:         prompt_interval, gen_interval, transfer_ratio = self._preset(request_meta)
    86:         return {
    87:             "use_feature_cache": True,
    88:             "prompt_refresh_interval": prompt_interval,
    89:             "gen_refresh_interval": gen_interval,
    90:             "transfer_ratio": transfer_ratio,
    91:             "row_selector": "lowest_value_feature_similarity",
    92:             "kv_update": "scatter_refresh",
    93:             "layer_reset": None,
    94:         }
    95: 
    96:     def attention_probe_plan(self, layer_meta, step_meta):
    97:         return {
    98:             "need_attention_weights": False,
    99:             "rollout_p": 0.0,
   100:             "current_k": 0,
   101:             "gamma": None,
   102:             "track_num": 0,
   103:         }
   104: 
   105:     def token_transfer_plan(self, logits, mask_state, step_meta):
   106:         return {
   107:             "mode": "low_confidence",
   108:             "scope": "current_block",
   109:             "num_transfer_tokens": step_meta["default_num_transfer_tokens"],
   110:             "threshold": None,
   111:             "force_one": True,
   112:         }
   113: 
   114:     def after_step(self, step_meta, logits, attention_stats, transfer_state, cache_state):
   115:         return cache_state
   116: # end of editable region – evaluation code starts below.
   117: 
   118: 
```

### `d2cache` baseline — editable region  [READ-ONLY — reference implementation]

In `dLLM-cache/custom_dlm_eval.py`:

```python
Lines 53–107:
    50: 
    51: 
    52: 
    53: class DLMRefreshPolicy:
    54:     """d2Cache: active query rows plus attention-rollout top-up."""
    55: 
    56:     policy_name = "d2cache"
    57: 
    58:     def block_schedule(self, request_meta):
    59:         wl = WORKLOAD_CONFIGS[request_meta["workload"]]
    60:         return {
    61:             "gen_length": wl["gen_length"],
    62:             "block_length": wl["gen_length"],
    63:             "num_steps": wl["gen_length"],
    64:             "warmup_forward": False,
    65:         }
    66: 
    67:     def query_plan(self, step_meta, mask_state, cache_state):
    68:         return {
    69:             "query_scope": "full_sequence" if step_meta["step_in_block"] == 0 else "active_query_rows",
    70:             "query_positions": cache_state.get("active_q_mask"),
    71:             "track_positions": [],
    72:             "masked_window": (mask_state["block_start"], mask_state["block_end"]),
    73:         }
    74: 
    75:     def cache_refresh_plan(self, layer_meta, step_meta, token_stats, cache_state):
    76:         return {
    77:             "use_feature_cache": False,
    78:             "prompt_refresh_interval": 1,
    79:             "gen_refresh_interval": 1,
    80:             "transfer_ratio": 0.0,
    81:             "row_selector": "certainty_density_attention_rollout",
    82:             "kv_update": "active_q_mask",
    83:             "layer_reset": None,
    84:         }
    85: 
    86:     def attention_probe_plan(self, layer_meta, step_meta):
    87:         return {
    88:             "need_attention_weights": True,
    89:             "rollout_p": 0.1,
    90:             "current_k": 32,
    91:             "gamma": None,
    92:             "track_num": 0,
    93:             "sigma": 10.0,
    94:             "inflate_w": 0,
    95:         }
    96: 
    97:     def token_transfer_plan(self, logits, mask_state, step_meta):
    98:         return {
    99:             "mode": "low_confidence",
   100:             "scope": "current_block",
   101:             "num_transfer_tokens": step_meta["default_num_transfer_tokens"],
   102:             "threshold": None,
   103:             "force_one": True,
   104:         }
   105: 
   106:     def after_step(self, step_meta, logits, attention_stats, transfer_state, cache_state):
   107:         return cache_state
   108: # end of editable region – evaluation code starts below.
   109: 
   110: 
```

### `elastic_cache` baseline — editable region  [READ-ONLY — reference implementation]

In `dLLM-cache/custom_dlm_eval.py`:

```python
Lines 53–106:
    50: 
    51: 
    52: 
    53: class DLMRefreshPolicy:
    54:     """Elastic-Cache: tracked-token windows with attention-similarity reset."""
    55: 
    56:     policy_name = "elastic_cache"
    57: 
    58:     def block_schedule(self, request_meta):
    59:         wl = WORKLOAD_CONFIGS[request_meta["workload"]]
    60:         return {
    61:             "gen_length": wl["gen_length"],
    62:             "block_length": wl["block_length"],
    63:             "window_length": 16,
    64:             "num_steps": wl["num_steps"],
    65:             "warmup_forward": False,
    66:         }
    67: 
    68:     def query_plan(self, step_meta, mask_state, cache_state):
    69:         return {
    70:             "query_scope": "full_sequence" if step_meta["step"] == 0 else "tracked_window",
    71:             "query_positions": None,
    72:             "track_positions": cache_state.get("track_positions", []),
    73:             "masked_window": (mask_state["block_start"], mask_state["block_end"]),
    74:         }
    75: 
    76:     def cache_refresh_plan(self, layer_meta, step_meta, token_stats, cache_state):
    77:         return {
    78:             "use_feature_cache": False,
    79:             "prompt_refresh_interval": 1,
    80:             "gen_refresh_interval": 1,
    81:             "transfer_ratio": 0.0,
    82:             "row_selector": "tracked_tokens_and_masked_window",
    83:             "kv_update": "tracked_window_layer_reset",
    84:             "layer_reset": "attention_similarity",
    85:         }
    86: 
    87:     def attention_probe_plan(self, layer_meta, step_meta):
    88:         return {
    89:             "need_attention_weights": True,
    90:             "rollout_p": 0.0,
    91:             "current_k": 0,
    92:             "gamma": 0.9,
    93:             "track_num": 1,
    94:         }
    95: 
    96:     def token_transfer_plan(self, logits, mask_state, step_meta):
    97:         return {
    98:             "mode": "confidence_threshold",
    99:             "scope": "masked_window",
   100:             "num_transfer_tokens": 1,
   101:             "threshold": 0.9,
   102:             "force_one": True,
   103:         }
   104: 
   105:     def after_step(self, step_meta, logits, attention_stats, transfer_state, cache_state):
   106:         return cache_state
   107: # end of editable region – evaluation code starts below.
   108: 
   109: 
```


## Tips

- Keep the function/class signatures of the editable regions identical;
  evaluation imports them by name.
- Determinism matters: seeds are fixed; don't introduce hidden randomness.
- The baseline implementations above are deliberately strong. Aim for an
  *algorithmic* improvement — many hyperparameters are locked outside the
  editable surface anyway.

Good luck.
