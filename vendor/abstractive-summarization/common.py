"""Shared runtime for the abstractive-summarization MLS-Bench tasks.

The package provides an inference-only `transformers` + `rouge-score` pipeline
over three complete official summarization test splits. Frozen model paths,
dataset and model digests, row counts, tokenization, batching, source truncation,
and ROUGE scoring are supplied here so task-specific surfaces vary only the
exposed policy or decode configuration.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Fixed evaluation constants (shared by ALL summ-* tasks)
# ---------------------------------------------------------------------------
DEFAULT_MAX_INPUT_TOKENS = 512  # fixed uniform source-token cap for all ten tasks
GEN_BATCH_SIZE = 16        # fixed generation batch size
SEED = 42
PROTOCOL = "summ-full-official-test-v1"

# The THREE FIXED settings and the FROZEN model staged offline for each.
SETTINGS = ("xsum", "cnndm", "samsum")

DATASET_INVENTORY = {
    "xsum": {
        "rows": 11334,
        "sha256": "5b7854e6ee8b81493e38a5255ba794dec21735b30ba8f77280a394bf7d6dae53",
    },
    "cnndm": {
        "rows": 11490,
        "sha256": "525e5bf29cb49a1ea2c58939d11435b6d6422f9a59d9c15e43d226c464b33278",
    },
    "samsum": {
        "rows": 819,
        "sha256": "b9dd8165689b1c252b2db617b089b408ea068ef5bd6819a93c64e8ebde856b86",
    },
}
TOTAL_DOCS = sum(int(entry["rows"]) for entry in DATASET_INVENTORY.values())

_MODEL_DIRS = {
    "xsum":   "distilbart-xsum-12-6",
    "cnndm":  "distilbart-cnn-12-6",
    "samsum": "bart-large-cnn-samsum",
}

MODEL_INVENTORY = {
    "xsum": {
        "revision": "5b2e376c845c201ddc34ec0e55fd1ad9890ba5ee",
        "weights_sha256": "6e9ebfc94e474225457570ad33c225cf66ca26279c5cd1cbfb67e089a03a791b",
        "weights_bytes": 611201041,
        "parameter_count": 305510400,
    },
    "cnndm": {
        "revision": "a4f8f3ea906ed274767e9906dbaede7531d660ff",
        "weights_sha256": "3bac65d18c99463302d12ca75c2220ea714f9c81ce235f205fa818efe71df6ea",
        "weights_bytes": 1222317369,
        "parameter_count": 305510400,
    },
    "samsum": {
        "revision": "e49b3d60d923f12db22bdd363356f1a4c68532ad",
        "weights_sha256": "9f453aa6edef4dba1893723b7313b57b06b60214442d308a8acc3baa9583dd7b",
        "weights_bytes": 1625565295,
        "parameter_count": 406290432,
    },
}

# Per-setting FIXED length window (matches that domain's reference length). These
# are the "sensible domain default" the harness pins when the task is NOT about
# the length window. Values are the published task_specific_params for each model.
LEN_WINDOW = {
    "xsum":   {"min_length": 11, "max_length": 62,  "length_penalty": 1.0},
    "cnndm":  {"min_length": 56, "max_length": 142, "length_penalty": 2.0},
    "samsum": {"min_length": 10, "max_length": 80,  "length_penalty": 1.0},
}


def data_root() -> Path:
    return Path(os.environ.get("SUMM_DATA",
                               "/data/abstractive-summarization/data"))


def models_root() -> Path:
    return Path(os.environ.get("SUMM_MODELS",
                               "/data/abstractive-summarization/models"))


def model_path(setting: str) -> str:
    """FROZEN pretrained summarizer for `setting`, staged offline."""
    if setting not in SETTINGS:
        raise SystemExit(f"unknown setting {setting!r}; expected one of {SETTINGS}")
    return str(models_root() / _MODEL_DIRS[setting])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup(seed: int = SEED):
    """Pin device + seed and force offline HF caches."""
    import random

    import numpy as np
    import torch

    if seed != SEED:
        raise SystemExit(f"seed must be exactly {SEED}, got {seed}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise SystemExit("summarization verification requires one CUDA GPU")
    torch.cuda.manual_seed_all(seed)
    dev = torch.device("cuda:0")
    return dev


def load_dataset(setting: str):
    """Load and authenticate one complete official test split."""
    if setting not in SETTINGS:
        raise SystemExit(f"unknown setting {setting!r}; expected one of {SETTINGS}")

    path = data_root() / f"{setting}_test.jsonl"
    expected = DATASET_INVENTORY[setting]
    if not path.is_file():
        raise SystemExit(f"missing complete test split: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected["sha256"]:
        raise SystemExit(
            f"dataset digest mismatch for {setting}: {actual_sha256} != "
            f"{expected['sha256']}"
        )

    docs: List[str] = []
    refs: List[str] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"malformed JSONL for {setting} at line {line_number}: {exc}"
                ) from exc
            if set(record) != {"id", "document", "summary"}:
                raise SystemExit(
                    f"invalid record schema for {setting} at line {line_number}"
                )
            row_id = record["id"]
            document = record["document"]
            summary = record["summary"]
            # XSum's canonical test split contains one source with an empty
            # document (id 39563665). Preserve that official row rather than
            # silently shrinking or resampling the split; BART tokenization
            # handles it as an empty input. IDs and references must be nonempty.
            if (not isinstance(row_id, str) or not row_id.strip()
                    or not isinstance(document, str)
                    or not isinstance(summary, str) or not summary.strip()):
                raise SystemExit(
                    f"empty or non-string field for {setting} at line {line_number}"
                )
            if row_id in ids:
                raise SystemExit(f"duplicate row id for {setting}: {row_id}")
            ids.add(row_id)
            docs.append(document)
            refs.append(summary)

    if len(docs) != expected["rows"]:
        raise SystemExit(
            f"wrong full-split row count for {setting}: {len(docs)} != "
            f"{expected['rows']}"
        )
    print(
        f"SUMM_DATA setting={setting} n_docs={len(docs)} "
        f"sha256={actual_sha256}",
        flush=True,
    )
    return docs, refs


def load_model_and_tokenizer(setting: str, device):
    """FROZEN domain-matched summarizer for `setting`, eval mode, staged offline."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    mp = model_path(setting)
    expected = MODEL_INVENTORY[setting]
    weights = Path(mp) / "pytorch_model.bin"
    if not weights.is_file() or weights.stat().st_size != expected["weights_bytes"]:
        raise SystemExit(f"missing or wrong-sized pinned weights for {setting}: {weights}")
    actual_weights_sha256 = _sha256(weights)
    if actual_weights_sha256 != expected["weights_sha256"]:
        raise SystemExit(
            f"model weights digest mismatch for {setting}: "
            f"{actual_weights_sha256} != {expected['weights_sha256']}"
        )
    tok = AutoTokenizer.from_pretrained(mp, local_files_only=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        mp, local_files_only=True, torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != expected["parameter_count"]:
        raise SystemExit(
            f"model parameter-count mismatch for {setting}: "
            f"{parameter_count} != {expected['parameter_count']}"
        )
    print(
        f"SUMM_MODEL setting={setting} model={_MODEL_DIRS[setting]} "
        f"revision={expected['revision']} params={parameter_count} "
        f"dtype={str(dtype).replace('torch.', '')} "
        f"weights_sha256={actual_weights_sha256}",
        flush=True,
    )
    return model, tok


def load_surface(sol_path: str, attr: str):
    """Import the agent-editable callable `attr` from solution/<file>.py."""
    p = Path(sol_path)
    spec = importlib.util.spec_from_file_location("agent_surface", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p.parent))
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, attr):
        raise SystemExit(f"solution must define `{attr}(...)`")
    return getattr(mod, attr)


def require_surface_config(value, required, *, allowed=None, surface="config"):
    """Require an explicit, complete mapping from an editable surface."""
    if not isinstance(value, dict):
        print(f"SURFACE_ERROR {surface}: expected dict, got {type(value).__name__}",
              flush=True)
        raise TypeError(f"{surface} must return a dict")
    required = set(required)
    allowed = set(allowed if allowed is not None else required)
    missing = sorted(required - set(value))
    extra = sorted(set(value) - allowed)
    if missing or extra:
        print(f"SURFACE_ERROR {surface}: missing={missing} extra={extra}", flush=True)
        raise ValueError(f"invalid {surface} schema")
    return value


def require_surface_choice(value, name, allowed, *, surface):
    """Require a string selector without coercing another type to text."""
    if not isinstance(value, str):
        print(f"SURFACE_ERROR {surface}: {name} must be a string", flush=True)
        raise TypeError(f"{name} must be a string")
    value = value.strip().lower()
    if value not in allowed:
        print(f"SURFACE_ERROR {surface}: unsupported {name}={value!r}", flush=True)
        raise ValueError(f"unsupported {name}")
    return value


def require_surface_int(value, name, low, high, *, surface):
    """Require a built-in integer in range without truncation or clamping."""
    if isinstance(value, bool) or not isinstance(value, int):
        print(f"SURFACE_ERROR {surface}: {name} must be an integer", flush=True)
        raise TypeError(f"{name} must be an integer")
    if not low <= value <= high:
        print(f"SURFACE_ERROR {surface}: {name}={value} outside [{low}, {high}]",
              flush=True)
        raise ValueError(f"{name} outside allowed range")
    return value


def require_surface_number(value, name, low, high, *, surface, low_open=False):
    """Require a finite built-in number in range without string conversion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        print(f"SURFACE_ERROR {surface}: {name} must be numeric", flush=True)
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        print(f"SUMM_NONFINITE {name}={value}", flush=True)
        raise ValueError(f"non-finite {name}")
    valid_low = value > low if low_open else value >= low
    if not valid_low or value > high:
        print(f"SURFACE_ERROR {surface}: {name}={value} outside allowed range",
              flush=True)
        raise ValueError(f"{name} outside allowed range")
    return value


# ---------------------------------------------------------------------------
# Generation + scoring
# ---------------------------------------------------------------------------
def _sanitize_gen_kwargs(gen_kwargs: dict) -> dict:
    """Validate generation kwargs without repairing an invalid configuration."""
    allowed = {
        "num_beams", "min_length", "max_length", "length_penalty",
        "no_repeat_ngram_size", "repetition_penalty", "early_stopping",
        "num_beam_groups", "diversity_penalty",
        # sampling knobs (for the sampling-vs-beam / temperature surfaces)
        "do_sample", "top_k", "top_p", "temperature",
    }
    if not isinstance(gen_kwargs, dict):
        print("SURFACE_ERROR summarization generation config must be a dict", flush=True)
        raise TypeError("generation config must be a dict")
    out = {}
    for k, v in gen_kwargs.items():
        if k not in allowed:
            raise SystemExit(
                f"decode kwarg {k!r} not allowed; permitted: {sorted(allowed)}"
            )
        out[k] = v
    def _int(name, low, high):
        value = out[name]
        if isinstance(value, bool) or not isinstance(value, int):
            print(f"SURFACE_ERROR summarization {name} must be an integer", flush=True)
            raise TypeError(f"{name} must be an integer")
        if not low <= value <= high:
            print(f"SURFACE_ERROR summarization {name}={value} outside [{low}, {high}]",
                  flush=True)
            raise ValueError(f"{name} outside allowed range")

    def _float(name, low, high, *, low_open=False):
        value = out[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            print(f"SURFACE_ERROR summarization {name} must be numeric", flush=True)
            raise TypeError(f"{name} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            print(f"SUMM_NONFINITE {name}={value}", flush=True)
            raise ValueError(f"non-finite {name}")
        valid_low = value > low if low_open else value >= low
        if not valid_low or value > high:
            bracket = "(" if low_open else "["
            print(f"SURFACE_ERROR summarization {name}={value} outside {bracket}{low}, {high}]",
                  flush=True)
            raise ValueError(f"{name} outside allowed range")
        out[name] = value

    for name, low, high in (
        ("num_beams", 1, 12),
        ("min_length", 0, 200),
        ("max_length", 1, 200),
        ("no_repeat_ngram_size", 0, 20),
        ("top_k", 0, 1000),
        ("num_beam_groups", 1, 12),
    ):
        if name in out:
            _int(name, low, high)
    for name, low, high, low_open in (
        ("length_penalty", 0.0, 10.0, True),
        ("repetition_penalty", 0.0, 10.0, True),
        ("temperature", 0.0, 5.0, True),
        ("top_p", 0.0, 1.0, True),
        ("diversity_penalty", 0.0, 10.0, False),
    ):
        if name in out:
            _float(name, low, high, low_open=low_open)
    for name in ("do_sample", "early_stopping"):
        if name in out and not isinstance(out[name], bool):
            print(f"SURFACE_ERROR summarization {name} must be bool", flush=True)
            raise TypeError(f"{name} must be bool")
    if "min_length" in out and "max_length" in out:
        if out["min_length"] > out["max_length"]:
            print("SURFACE_ERROR summarization min_length exceeds max_length", flush=True)
            raise ValueError("min_length exceeds max_length")
    groups = out.get("num_beam_groups", 1)
    beams = out.get("num_beams", 1)
    if groups > beams or beams % groups:
        print("SURFACE_ERROR summarization beam groups must divide num_beams", flush=True)
        raise ValueError("invalid beam grouping")
    if "diversity_penalty" in out and (
        groups <= 1 or out["diversity_penalty"] <= 0.0
    ):
        print("SURFACE_ERROR summarization positive diversity_penalty requires grouped beams",
              flush=True)
        raise ValueError("positive diversity_penalty requires num_beam_groups > 1")
    return out


def generate_summaries(model, tok, docs: List[str], gen_kwargs: dict,
                       device, max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
                       setting: str = "unknown",
                       ) -> List[str]:
    """Decode summaries for every document with the agent's decode config.

    Source truncation (512 tokenizer tokens), batching, and the frozen model are
    fixed by the harness; only `gen_kwargs` (the agent's decode config) varies.
    """
    import torch

    if isinstance(max_input_tokens, bool) or max_input_tokens != DEFAULT_MAX_INPUT_TOKENS:
        print(
            f"SURFACE_ERROR summarization max_input_tokens must be "
            f"{DEFAULT_MAX_INPUT_TOKENS}",
            flush=True,
        )
        raise ValueError("invalid fixed source-token cap")
    gk = _sanitize_gen_kwargs(gen_kwargs)
    preds: List[str] = []
    for i in range(0, len(docs), GEN_BATCH_SIZE):
        batch = docs[i:i + GEN_BATCH_SIZE]
        enc = tok(
            batch,
            max_length=int(max_input_tokens),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                **gk,
            )
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))
        completed = min(i + len(batch), len(docs))
        if completed == len(docs) or completed % 1024 == 0:
            print(
                f"SUMM_PROGRESS setting={setting} completed={completed}/{len(docs)}",
                flush=True,
            )
    return preds


def score_rouge(preds: List[str], refs: List[str]) -> dict:
    """Mean per-example ROUGE-1/2/L **F1** via the `rouge_score` library
    (Google's implementation; the standard, non-gameable F-measure)."""
    from rouge_score import rouge_scorer

    if not preds or len(preds) != len(refs):
        print(f"SURFACE_ERROR summarization prediction/reference count "
              f"{len(preds)}/{len(refs)}", flush=True)
        raise ValueError("incomplete summarization predictions")
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = 0
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)  # (target=ref, prediction=pred)
        for k in agg:
            agg[k] += s[k].fmeasure
        n += 1
    scores = {k: v / n for k, v in agg.items()}
    if any(not math.isfinite(v) for v in scores.values()):
        print(f"SUMM_NONFINITE scores={scores}", flush=True)
        raise ValueError("non-finite ROUGE score")
    return scores


def mean_pred_len_words(preds: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(len(p.split()) for p in preds) / len(preds)


# ---------------------------------------------------------------------------
# Unified 3-setting driver (used by every surface harness)
# ---------------------------------------------------------------------------
def emit_metrics(setting: str, scores: dict, plen: float, n_docs: int) -> None:
    print(f"SUMM_METRICS setting={setting} "
          f"rougeL={scores['rougeL']:.6f} rouge1={scores['rouge1']:.6f} "
          f"rouge2={scores['rouge2']:.6f} plen={plen:.1f} n_docs={n_docs}",
          flush=True)


def run_over_settings(build_gen_for_setting, device,
                      max_input_tokens_for_setting=None,
                      preds_override_for_setting=None) -> None:
    """Loop over the THREE FIXED settings, decode with the surface config, and
    emit one SUMM_METRICS line per setting.

    build_gen_for_setting(setting) -> gen_kwargs dict (or None if preds are
        produced by preds_override_for_setting instead of model.generate).
    max_input_tokens_for_setting(setting) -> int (optional; must remain 512).
    preds_override_for_setting(setting, docs) -> list[str] (optional; for
        non-model source policies like copy/empty/lead).
    """
    print(
        f"SUMM_PROTOCOL version={PROTOCOL} settings={len(SETTINGS)} "
        f"total_docs={TOTAL_DOCS}",
        flush=True,
    )
    completed_settings = 0
    completed_docs = 0
    for setting in SETTINGS:
        docs, refs = load_dataset(setting)
        if preds_override_for_setting is not None:
            override = preds_override_for_setting(setting, docs)
        else:
            override = None
        if override is not None:
            preds = override
        else:
            model, tok = load_model_and_tokenizer(setting, device)
            gk = build_gen_for_setting(setting)
            mit = (max_input_tokens_for_setting(setting)
                   if max_input_tokens_for_setting else DEFAULT_MAX_INPUT_TOKENS)
            preds = generate_summaries(
                model, tok, docs, gk, device, mit, setting=setting
            )
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        scores = score_rouge(preds, refs)
        plen = mean_pred_len_words(preds)
        emit_metrics(setting, scores, plen, len(docs))
        completed_settings += 1
        completed_docs += len(docs)
        print(
            f"SUMM_SETTING_DONE setting={setting} generated={len(preds)} "
            f"expected={DATASET_INVENTORY[setting]['rows']}",
            flush=True,
        )
    print(
        f"SUMM_EVAL_DONE settings={completed_settings} total_docs={completed_docs}",
        flush=True,
    )
