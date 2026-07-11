"""Shared pipeline for the machine-translation (mt-*) MLS-Bench tasks.

A self-contained ``transformers`` + ``sacrebleu`` harness for INFERENCE-ONLY
neural machine translation. Every task loads a FROZEN small pretrained MT model
(a Helsinki-NLP OPUS-MT MarianMT Transformer, ~75M params) staged offline,
translates the complete official parallel test split into English, and scores corpus
**sacreBLEU** (and **chrF**) against the FIXED English references. Nothing is
trained. Each direction evaluates all 2000 official OPUS-100 pairs on one GPU.

Three DIRECTIONS / settings (all -> English so the sacreBLEU reference tokenizer
and metric are consistent across settings):

  * de_en  : German  -> English   (Helsinki-NLP/opus-mt-de-en)
  * fr_en  : French  -> English   (Helsinki-NLP/opus-mt-fr-en)
  * ru_en  : Russian -> English   (Helsinki-NLP/opus-mt-ru-en)

Each mt-* task aggregates its metric (geometric mean) over these THREE settings.
The weak->strong->SOTA decoding order (greedy < beam < beam+length-penalty tuned)
holds in ALL three directions, so the ordering is verified across settings — the
MLS-Bench "reproduce >=2 baselines + SOTA, preserve the partial-order over >=3
settings" acceptance bar.

Why THIS is a genuinely new direction: the registry has monolingual NLP
(NER / relation-extraction / dependency-parsing / abstractive-summarization /
embeddings) but NO cross-lingual translation. MT is a distinct task with a
distinct, standard, non-gameable metric (sacreBLEU) and a well-studied set of
INFERENCE-TIME decoding levers (Koehn & Knowles 2017; Wu et al. 2016 GNMT;
Stahlberg & Byrne 2019; Vijayakumar et al. 2016; Freitag & Al-Onaizan 2017).

The agent-editable surface never touches the model, the corpus, the references,
or the sacreBLEU evaluator. It controls ONLY the one DECODING component the task
is about (beam / length-normalization / repetition-block / coverage / sampling /
temperature / n-best rerank / tokenization+truncation / batch+max-length /
early-stopping / detok / decode strategy).

Why sacreBLEU (not a hand-rolled BLEU): sacreBLEU is the community-standard,
reproducible, tokenization-controlled corpus BLEU (Post 2018), so scores are
comparable and non-gameable. BLEU is a geometric mean of n-gram precisions with
a brevity penalty, so an empty output scores 0 and a copy-the-source (wrong
language) output scores ~0 — a real translation cannot be beaten by degenerate
tricks.

Everything runs offline (HF_HUB_OFFLINE=1) and deterministically for the scored
beam/greedy paths (sampling paths seed-fix RNG for reproducibility).
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Fixed evaluation constants (shared by ALL mt-* tasks / directions)
# ---------------------------------------------------------------------------
OFFICIAL_TEST_PAIRS = 2000 # complete OPUS-100 test split for every direction
MAX_INPUT_TOKENS = 128     # source truncation (OPUS-100 sentences are ~15-25 words)
MAX_NEW_TOKENS_CAP = 160   # hard cap on generated length
GEN_BATCH_SIZE = 16        # default generation batch size (a task may override)
SEED = 42

# The three directions this repo evaluates over. Each entry: (model_dir, test_file).
# All target English so the sacreBLEU reference side is one language.
DIRECTIONS = {
    "de_en": ("opus-mt-de-en", "de_en_test.jsonl"),
    "fr_en": ("opus-mt-fr-en", "fr_en_test.jsonl"),
    "ru_en": ("opus-mt-ru-en", "ru_en_test.jsonl"),
}
EXPECTED_TEST_SHA256 = {
    "de_en": "2e7a80586d269952371ff5e71f8840e26926416c399051e2371a3b14a1b0b6dc",
    "fr_en": "09477f8a19e67d3f7c09c320d076f5a32168ab4cde55d8f1d88ffb66c02f68a1",
    "ru_en": "c072e931f99d1ed04829ec4f63b18c34ebc9ead4b1b19b25fceec60720650eb0",
}

TGT_LANG = "en"


def _data_root() -> Path:
    return Path(os.environ.get("MT_DATA", "/data/machine-translation/data"))


def _models_root() -> Path:
    # MT_MODEL historically pointed at the single de-en model dir; derive the
    # models root from it so per-direction models resolve next to it.
    mm = os.environ.get("MT_MODEL")
    if mm:
        p = Path(mm)
        # if MT_MODEL is .../models/opus-mt-de-en, the root is its parent
        if p.name.startswith("opus-mt-"):
            return p.parent
        return p
    return Path("/data/machine-translation/models")


def direction() -> str:
    """Active direction key (env MT_DIR; default de_en)."""
    d = os.environ.get("MT_DIR", "de_en")
    if d not in DIRECTIONS:
        raise SystemExit(f"MT_DIR={d!r} not one of {sorted(DIRECTIONS)}")
    return d


def model_path(dir_key: str | None = None) -> str:
    """FROZEN small pretrained OPUS-MT model for the active direction."""
    dk = dir_key or direction()
    model_dir, _ = DIRECTIONS[dk]
    # de_en can also be resolved from the legacy MT_MODEL env directly.
    if dk == "de_en":
        mm = os.environ.get("MT_MODEL")
        if mm and Path(mm).name.startswith("opus-mt-de-en"):
            return mm
    return str(_models_root() / model_dir)


def data_root() -> Path:
    return _data_root()


def setup(seed: int = SEED):
    """Pin device + seed and force offline HF caches."""
    import random

    import numpy as np
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return dev


def load_dataset(dir_key: str | None = None):
    """Load the complete official OPUS-100 test split for one direction."""
    dk = dir_key or direction()
    if dk not in DIRECTIONS:
        raise ValueError(f"unknown direction {dk!r}")
    _, fname = DIRECTIONS[dk]
    fp = _data_root() / fname
    try:
        actual_sha256 = hashlib.sha256(fp.read_bytes()).hexdigest()
    except OSError as exc:
        raise FileNotFoundError(f"cannot read staged OPUS-100 split {fp}: {exc}") from exc
    if actual_sha256 != EXPECTED_TEST_SHA256[dk]:
        raise ValueError(
            f"official OPUS-100 split digest mismatch for {fp}: expected "
            f"{EXPECTED_TEST_SHA256[dk]}, got {actual_sha256}"
        )
    srcs: List[str] = []
    refs: List[str] = []
    with fp.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {fp}:{line_number}")
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {fp}:{line_number}: {exc}") from exc
            if not isinstance(rec, dict) or set(rec) != {"src", "ref"}:
                raise ValueError(
                    f"invalid record schema at {fp}:{line_number}; expected src/ref"
                )
            src, ref = rec["src"], rec["ref"]
            if not isinstance(src, str) or not src.strip():
                raise ValueError(f"empty source at {fp}:{line_number}")
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"empty reference at {fp}:{line_number}")
            srcs.append(src)
            refs.append(ref)
    if len(srcs) != OFFICIAL_TEST_PAIRS:
        raise ValueError(
            f"incomplete official split {fp}: expected {OFFICIAL_TEST_PAIRS} rows, "
            f"got {len(srcs)}"
        )
    return srcs, refs


def load_model_and_tokenizer(device, dir_key: str | None = None):
    """FROZEN OPUS-MT MarianMT for the active direction, eval mode, offline."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    mp = model_path(dir_key)
    model_dir = Path(mp)
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"missing staged OPUS-MT model: {model_dir}")
    tok = AutoTokenizer.from_pretrained(mp, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        mp, local_files_only=True, torch_dtype=torch.float32
    )
    model.to(device)
    model.eval()
    return model, tok


_SURFACE_SOURCE_BYTES = 64 * 1024
_SURFACE_AST_NODES = 256
_SURFACE_NAMES = {
    "build_beam_config",
    "build_divbeam_config",
    "build_early_stopping",
    "build_length_config",
    "build_max_new_tokens",
    "build_norep_config",
    "build_postproc",
    "build_reppen_config",
    "build_mode",
    "build_strategy",
    "build_temperature",
    "build_source_max_tokens",
}


def _surface_error(message: str) -> ValueError:
    return ValueError(f"unsafe machine-translation configuration: {message}")


def _validate_data_literal(value, *, path: str = "return") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _surface_error(f"{path} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_data_literal(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _surface_error(f"{path} has a non-string mapping key")
            _validate_data_literal(item, path=f"{path}[{key!r}]")
        return
    raise _surface_error(f"{path} has unsupported type {type(value).__name__}")


def load_surface_value(sol_path: str, attr: str):
    """Read one zero-argument literal-return surface without executing the file."""
    if attr not in _SURFACE_NAMES:
        raise _surface_error(f"unsupported surface name {attr!r}")
    path = Path(sol_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"solution file does not exist: {path}")
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _surface_error(f"cannot read solution: {exc}") from exc
    if len(source.encode("utf-8")) > _SURFACE_SOURCE_BYTES:
        raise _surface_error("source exceeds 64 KiB")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise _surface_error(f"source does not parse: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > _SURFACE_AST_NODES:
        raise _surface_error("AST exceeds 256 nodes")

    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == attr
    ]
    if len(functions) != 1:
        raise _surface_error(f"solution must define exactly one {attr}")
    function = functions[0]
    args = function.args
    if (
        function.decorator_list
        or args.posonlyargs
        or args.args
        or args.vararg is not None
        or args.kwonlyargs
        or args.kwarg is not None
        or args.defaults
        or args.kw_defaults
    ):
        raise _surface_error(f"{attr} must be undecorated and accept no arguments")
    if len(function.body) != 1 or not isinstance(function.body[0], ast.Return):
        raise _surface_error(f"{attr} body must contain exactly one return statement")

    for node in tree.body:
        if node is function:
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and [alias.name for alias in node.names] == ["annotations"]
        ):
            continue
        raise _surface_error("top-level executable statements and imports are forbidden")

    try:
        value = ast.literal_eval(function.body[0].value)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise _surface_error(f"{attr} must return one literal value") from exc
    _validate_data_literal(value)
    return value


def require_config(value, attr: str, required: set[str], optional: set[str] | None = None):
    """Validate an editable mapping without supplying verifier-side defaults."""
    if not isinstance(value, dict):
        raise TypeError(f"{attr} must return a mapping, got {type(value).__name__}")
    optional = optional or set()
    missing = required - set(value)
    unexpected = set(value) - required - optional
    if missing:
        raise ValueError(f"{attr} is missing required keys: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"{attr} has unsupported keys: {sorted(unexpected)}")
    return value


def require_int(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum},{maximum}], got {value}")
    return value


def require_real(value, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum},{maximum}], got {value}")
    return result


# ---------------------------------------------------------------------------
# Generation config sanitisation
# ---------------------------------------------------------------------------
def _sanitize_gen_kwargs(gen_kwargs: dict) -> dict:
    """Validate the agent's decode config: only known generation knobs allowed,
    and hard caps so nothing can blow up the compute budget or game the metric."""
    allowed = {
        "num_beams", "min_length", "max_new_tokens", "length_penalty",
        "no_repeat_ngram_size", "repetition_penalty", "early_stopping",
        "do_sample", "temperature", "top_k", "top_p", "num_beam_groups",
        "diversity_penalty", "num_return_sequences",
    }
    if not isinstance(gen_kwargs, dict):
        raise TypeError("generation configuration must be a mapping")
    out = {}
    for k, v in gen_kwargs.items():
        if k not in allowed:
            raise SystemExit(
                f"decode kwarg {k!r} not allowed; permitted: {sorted(allowed)}"
            )
        out[k] = v
    out["num_beams"] = require_int(out.get("num_beams", 1), "num_beams", 1, 12)
    out["max_new_tokens"] = require_int(
        out.get("max_new_tokens", MAX_NEW_TOKENS_CAP),
        "max_new_tokens",
        1,
        MAX_NEW_TOKENS_CAP,
    )
    out["min_length"] = require_int(
        out.get("min_length", 0), "min_length", 0, MAX_NEW_TOKENS_CAP
    )
    if out["min_length"] > out["max_new_tokens"]:
        raise ValueError("min_length cannot exceed max_new_tokens")
    if "length_penalty" in out:
        out["length_penalty"] = require_real(
            out["length_penalty"], "length_penalty", 0.0, 5.0
        )
    if "no_repeat_ngram_size" in out:
        out["no_repeat_ngram_size"] = require_int(
            out["no_repeat_ngram_size"], "no_repeat_ngram_size", 0, 10
        )
    if "repetition_penalty" in out:
        out["repetition_penalty"] = require_real(
            out["repetition_penalty"], "repetition_penalty", 0.1, 5.0
        )
    if "temperature" in out:
        out["temperature"] = require_real(
            out["temperature"], "temperature", 0.05, 5.0
        )
    if "top_k" in out:
        out["top_k"] = require_int(out["top_k"], "top_k", 0, 1000)
    if "top_p" in out:
        out["top_p"] = require_real(out["top_p"], "top_p", 0.0, 1.0)
    if "diversity_penalty" in out:
        out["diversity_penalty"] = require_real(
            out["diversity_penalty"], "diversity_penalty", 0.0, 5.0
        )
    if "do_sample" in out and not isinstance(out["do_sample"], bool):
        raise TypeError("do_sample must be a boolean")
    if "early_stopping" in out and out["early_stopping"] not in (True, False, "never"):
        raise ValueError("early_stopping must be True, False, or 'never'")
    if "num_return_sequences" in out:
        out["num_return_sequences"] = require_int(
            out["num_return_sequences"],
            "num_return_sequences",
            1,
            out["num_beams"],
        )
    if "num_beam_groups" in out:
        out["num_beam_groups"] = require_int(
            out["num_beam_groups"], "num_beam_groups", 1, out["num_beams"]
        )
        if out["num_beams"] % out["num_beam_groups"]:
            raise ValueError("num_beam_groups must divide num_beams")
    return out


# ---------------------------------------------------------------------------
# Generation + scoring
# ---------------------------------------------------------------------------
def translate(model, tok, sources: List[str], gen_kwargs: dict, device,
              max_input_tokens: int = MAX_INPUT_TOKENS,
              batch_size: int = GEN_BATCH_SIZE) -> List[str]:
    """Decode a translation for every source sentence with the agent's config.

    The frozen model is fixed; only `gen_kwargs`, the source truncation length
    (`max_input_tokens`) and the batch size may vary (some tasks expose those).
    """
    import torch

    if not sources:
        raise ValueError("translation source list is empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if (
        isinstance(max_input_tokens, bool)
        or not isinstance(max_input_tokens, int)
        or not 1 <= max_input_tokens <= MAX_INPUT_TOKENS
    ):
        raise ValueError(
            f"max_input_tokens must be an integer in [1,{MAX_INPUT_TOKENS}], "
            f"got {max_input_tokens!r}"
        )
    gk = _sanitize_gen_kwargs(gen_kwargs)
    preds: List[str] = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        enc = tok(
            batch,
            max_length=max_input_tokens,
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
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        if len(decoded) != len(batch) or any(not isinstance(item, str) for item in decoded):
            raise RuntimeError(
                f"decoder returned {len(decoded)} outputs for a batch of {len(batch)} sources"
            )
        preds.extend(decoded)
    if len(preds) != len(sources):
        raise RuntimeError(
            f"decoder returned {len(preds)} outputs for {len(sources)} sources"
        )
    return preds


def translate_nbest(model, tok, sources: List[str], gen_kwargs: dict, device,
                    n: int, max_input_tokens: int = MAX_INPUT_TOKENS,
                    batch_size: int = GEN_BATCH_SIZE) -> List[List[str]]:
    """Return the top-`n` beam hypotheses per source (for n-best reranking tasks).

    Each output row is a list of `n` candidate strings (best-first by model
    score). `num_beams` must be >= n; the harness sets num_return_sequences=n.
    """
    import torch

    n = require_int(n, "n", 1, 12)
    if not sources:
        raise ValueError("translation source list is empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if (
        isinstance(max_input_tokens, bool)
        or not isinstance(max_input_tokens, int)
        or not 1 <= max_input_tokens <= MAX_INPUT_TOKENS
    ):
        raise ValueError(
            f"max_input_tokens must be an integer in [1,{MAX_INPUT_TOKENS}], "
            f"got {max_input_tokens!r}"
        )
    gk = _sanitize_gen_kwargs(gen_kwargs)
    gk["num_beams"] = max(gk.get("num_beams", 1), n)
    gk["num_return_sequences"] = n
    gk.pop("do_sample", None)          # n-best is a beam concept
    all_lists: List[List[str]] = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        enc = tok(batch, max_length=max_input_tokens, truncation=True,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"],
                                  attention_mask=enc["attention_mask"], **gk)
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        if len(decoded) != len(batch) * n:
            raise RuntimeError(
                f"n-best decoder returned {len(decoded)} outputs for "
                f"{len(batch)} sources with n={n}"
            )
        # out is (batch*n) rows, grouped per source
        for b in range(len(batch)):
            all_lists.append(decoded[b * n:(b + 1) * n])
    if len(all_lists) != len(sources) or any(len(row) != n for row in all_lists):
        raise RuntimeError("n-best decoder output is incomplete")
    return all_lists


def score_bleu_chrf(preds: List[str], refs: List[str]) -> dict:
    """Corpus sacreBLEU + chrF over the slice via the `sacrebleu` library
    (Post 2018; the community-standard, tokenization-controlled, non-gameable
    corpus BLEU). Single reference -> wrap the ref list once: [refs].

    Returns {"bleu": <0-100>, "chrf": <0-100>}.
    """
    import sacrebleu

    if not preds or len(preds) != len(refs):
        raise ValueError(
            f"predictions/references must be non-empty and aligned, got "
            f"{len(preds)} and {len(refs)}"
        )
    if any(not isinstance(item, str) for item in preds + refs):
        raise TypeError("predictions and references must contain only strings")

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    if not math.isfinite(bleu) or not math.isfinite(chrf):
        raise ValueError(f"non-finite translation metrics: bleu={bleu}, chrf={chrf}")
    return {"bleu": float(bleu), "chrf": float(chrf)}


def mean_pred_len_words(preds: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(len(p.split()) for p in preds) / len(preds)
