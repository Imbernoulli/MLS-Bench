"""Shared pipeline for the text-simplification (simp-*) MLS-Bench tasks.

A self-contained ``transformers`` + a vendored **SARI** harness for INFERENCE-ONLY
sentence simplification. Every task loads a FROZEN small pretrained seq2seq
simplifier (``mrm8488/t5-base-finetuned-turk-text-simplification``, a T5-base
fine-tuned on the Wiki/Turk simplification data, ~220M params) staged offline,
rewrites a FIXED small test slice in EACH of THREE distinct simplification test
sets, and scores corpus **SARI** (Xu et al. 2016) against the FIXED
multi-reference set. Nothing is trained; the whole run is a few hundred short beam
decodes per setting on a single GPU (minute-scale; the model is tiny).

## Why THIS is a genuinely new direction

Text simplification REWRITES a complex sentence into a simpler, meaning-preserving
one (via lexical paraphrasing, deletion of non-essential content, AND sentence
splitting). It is distinct from every existing NLP package:
  * abstractive-summarization (summ-*) COMPRESSES a document (many-to-one, ROUGE).
  * machine-translation (mt-*) changes LANGUAGE (de->en, sacreBLEU).
  * grammar-error-correction (gec-*) makes MINIMAL grammatical edits (GLEU / F0.5).
Simplification is same-language, same-meaning, but must be EASIER to read, and its
canonical metric SARI is a distinct, standard, reference-based, **non-gameable**
metric: it compares the SOURCE, the SYSTEM output, AND multiple human REFERENCES
and rewards correct n-gram ADD / KEEP / DELETE edits:

    SARI = (F1_add + F1_keep + P_del) / 3   averaged over n = 1..4

So an output that just COPIES the source earns 0 ADD credit and 0 DELETE credit
and cannot beat a genuine simplification. The references are FIXED, so the metric
is not gameable (unlike BLEU/ROUGE, which reward copying the input).

## The THREE FIXED test settings (all from GEM/wiki_auto_asset_turk, staged offline)

  * asset : ASSET test split (359 sentences, 10 human refs each; Alva-Manchego
            et al. 2020). References feature the FULL range of rewriting ops
            (paraphrase + compression + splitting) -> the hardest, most
            discriminating setting.
  * turk  : TurkCorpus test split (359 sentences, 7-8 human refs each; Xu et al.
            2016). References focus on LEXICAL simplification / light editing ->
            copying the input is a stronger (but still beaten) baseline here.
  * wiki  : WikiAuto manual test split (720 sentences, 1 ref each). Real-world
            Wikipedia -> Simple-Wikipedia rewrites; longer sources -> the
            single-reference, harder-recall setting.

The task score is the geometric mean of the per-setting SARI, so a method must
simplify well across ALL THREE (a method that only helps on one setting is
penalised). The three settings have the SAME sources for asset/turk but DIFFERENT
reference styles, and wiki is a disjoint, longer, single-ref set.

## Agent-editable surface

The surface never touches the model, the corpora, the references, the tokenizer,
or the SARI evaluator. It controls ONLY one INFERENCE-TIME component:

  * policy  : the simplification SOURCE POLICY (the monotonicity / anti-gaming
              task) — a degenerate baseline (copy the complex input unchanged / a
              naive TRUNCATION that just deletes the tail) vs a real greedy vs a
              real tuned-beam T5 simplifier decode. Proves SARI is monotone and
              un-gameable across all three settings (copy < truncate < model).
  * beam    : the beam / repetition config of the FROZEN simplifier (num_beams /
              no_repeat_ngram_size / repetition_penalty) — greedy under-searches;
              a tuned beam is the standard strong simplification decode.
  * length  : the length / compression config (max_length / min_length /
              length_penalty). Simplification often SHORTENS a sentence, so
              length_penalty is a direct lever on the DELETE/ADD balance SARI
              measures: a runaway-long decode acts like copy-the-input (few
              deletes -> lower SARI), a sensibly compressive window recovers edits.

Everything runs offline (HF_HUB_OFFLINE=1) and deterministically (greedy and beam
search are deterministic given the frozen model; no sampling in the scored paths).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Fixed evaluation constants (shared by ALL simp-* tasks)
# ---------------------------------------------------------------------------
N_SENTS = 300              # deterministic head-slice per setting (minute-scale)
MAX_INPUT_TOKENS = 160     # source truncation (sentences are ~20-30 words)
MAX_NEW_TOKENS_CAP = 200   # hard cap on generated length (keeps it minute-scale)
GEN_BATCH_SIZE = 16        # fixed generation batch size
SEED = 42

# T5 simplification models are trained with a task prefix; FIXED here.
SRC_PREFIX = "simplify: "

# The three FIXED test settings, staged offline as JSONL {source, references}.
SETTINGS = ("asset", "turk", "wiki")

# Sources-only jsonl (no `references` field -- never contained the answer key)
# shipped directly alongside this file, agent-visible at all times. Produced
# host-side by vendor/data_scripts/text-simplification/prepare_data.py.
_SIMP_DATA_DIR = Path(__file__).resolve().parent / "_simp_data"


def model_path() -> str:
    """FROZEN small pretrained simplifier staged offline (t5-base-turk-simplify)."""
    return os.environ.get(
        "SIMP_MODEL",
        "/data/text-simplification/models/t5-base-finetuned-turk-text-simplification",
    )


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


def _refs_data_path(setting: str) -> Path:
    """Resolve the verifier-only held-out references jsonl for `setting`.

    Only exists under $TASK_DIR/data (set by test.sh / score_task.py's
    _install_task_meta_legacy_links to the per-task verifier meta dir) during
    scoring; absent during the agent's action session. Mirrors
    vendor/tpp-neural-hawkes/common.py::_test_data_path and
    vendor/ebm-langevin-cd/common.py::_test_data_path exactly.
    """
    task_dir = Path(os.environ.get("TASK_DIR", "/workspace/_task"))
    return task_dir / "data" / f"simp_{setting}_refs.jsonl"


def load_dataset(setting: str, n_sents: int = N_SENTS):
    """FIXED simplification test slice for ONE setting.

    Returns (sources, references): sources is a list[str] (the complex sentence);
    references is a list[list[str]] (each source paired with its human reference
    simplifications). The staged files are deterministic head-slices of the
    GEM/wiki_auto_asset_turk test_{asset,turk,wiki} splits, serialised offline as
    JSONL so the container needs no network.

    ## Held-out-reference hygiene (class-3 fix, 2026-07-05, PR #54 pattern)

    Sources are NOT secret (an agent must see the complex sentence to
    simplify it) and load from the agent-visible, vendor-checked-in
    ``vendor/text-simplification/_simp_data/simp_<setting>_src.jsonl`` (no
    ``references`` field at all -- that file never contained the answer key).

    The human reference simplifications ARE the held-out answer key SARI is
    scored against, so they must NOT sit anywhere under the agent-visible
    ``vendor/`` tree or in the Harbor base image the agent's shell can read.
    They load ONLY from the verifier-only
    ``$TASK_DIR/data/simp_<setting>_refs.jsonl`` (see ``_refs_data_path``),
    which Harbor's ``test.sh`` / native's ``score_task.py::
    _install_task_meta_legacy_links`` stage from each task's ``data/`` dir
    into ``tests/meta/data`` and symlink at ``$TASK_DIR/data`` ONLY at
    verification time -- see
    ``harbor_adapter/src/mls_bench/adapter.py::_stage_verifier_assets``. If
    that file is absent (i.e. this is an agent action session, not scoring),
    we SystemExit rather than silently falling back to any other source --
    there is no combined file anymore and no env-var indirection to a
    same-data-root copy. This exactly mirrors
    ``vendor/tpp-neural-hawkes/common.py::make_dataset`` and
    ``vendor/ebm-langevin-cd/common.py::make_dataset``.
    """
    import json

    if setting not in SETTINGS:
        raise SystemExit(f"unknown setting {setting!r}; expected one of {SETTINGS}")

    def _read_jsonl(fp: Path):
        with fp.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_sents:
                    break
                yield json.loads(line)

    src_fp = _SIMP_DATA_DIR / f"simp_{setting}_src.jsonl"
    if not src_fp.exists():
        raise SystemExit(
            f"no frozen source data for setting={setting!r}; expected "
            f"{src_fp}. Regenerate with "
            f"holdout/text-simplification/generate_data.py (host-side only)."
        )
    srcs: List[str] = [rec["source"] for rec in _read_jsonl(src_fp)]

    refs_fp = _refs_data_path(setting)
    if not refs_fp.exists():
        raise SystemExit(
            f"no held-out reference data for setting={setting!r}; expected "
            f"{refs_fp}. This file is staged verifier-side only "
            f"(tests/meta/data, mounted at $TASK_DIR/data) and is not "
            f"available during an agent action session -- SARI can only be "
            f"computed at scoring time, once the held-out human "
            f"simplifications are mounted."
        )
    refs: List[List[str]] = [
        [r for r in rec["references"] if r and r.strip()]
        for rec in _read_jsonl(refs_fp)
    ]

    if len(srcs) != len(refs):
        raise SystemExit(
            f"source/reference row-count mismatch for setting={setting!r}: "
            f"{len(srcs)} sources vs {len(refs)} reference rows"
        )
    return srcs, refs


def load_model_and_tokenizer(device):
    """FROZEN t5-base simplifier, eval mode, staged offline."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path(), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path(), local_files_only=True, torch_dtype=torch.float32
    )
    model.to(device)
    model.eval()
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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _sanitize_gen_kwargs(gen_kwargs: dict) -> dict:
    """Validate the agent's decode config: only known generation knobs allowed,
    and hard caps so nothing can blow up the compute budget.

    Extended (2026-07-05) beyond the original {num_beams, min_length, max_length,
    max_new_tokens, length_penalty, no_repeat_ngram_size, repetition_penalty,
    early_stopping} to also allow the sampling / diverse-beam knobs used by the
    NEW simp-* tasks (decoding-temperature, nucleus-topp, sampling-vs-beam,
    diverse-beam): do_sample, temperature, top_p, top_k, num_beam_groups,
    diversity_penalty. All ORIGINAL keys keep their EXACT original caps below
    (unchanged), so the 3 pre-existing tasks reproduce byte-identical."""
    allowed = {
        "num_beams", "min_length", "max_length", "max_new_tokens",
        "length_penalty", "no_repeat_ngram_size", "repetition_penalty",
        "early_stopping",
        # NEW knobs (all default to the ORIGINAL deterministic behaviour when
        # absent, so old callers that never set them are unaffected):
        "do_sample", "temperature", "top_p", "top_k",
        "num_beam_groups", "diversity_penalty",
    }
    out = {}
    for k, v in (gen_kwargs or {}).items():
        if k not in allowed:
            raise SystemExit(
                f"decode kwarg {k!r} not allowed; permitted: {sorted(allowed)}"
            )
        out[k] = v
    # hard caps (keep it minute-scale; cannot game the metric via runaway length)
    out["num_beams"] = int(min(max(int(out.get("num_beams", 1)), 1), 12))
    if "max_new_tokens" in out:
        out["max_new_tokens"] = int(min(int(out["max_new_tokens"]), MAX_NEW_TOKENS_CAP))
    out["max_length"] = int(min(int(out.get("max_length", MAX_NEW_TOKENS_CAP)),
                                MAX_NEW_TOKENS_CAP))
    out["min_length"] = int(max(int(out.get("min_length", 0)), 0))
    if out["min_length"] > out["max_length"]:
        out["min_length"] = out["max_length"]

    # NEW knob caps (sane ranges; nothing pathological/slow, and sampling can
    # never be exploited to reproduce a reference verbatim since it's the
    # SAME frozen model / SAME fixed inputs, only re-weighting its own logits).
    if "do_sample" in out:
        out["do_sample"] = bool(out["do_sample"])
    if "temperature" in out:
        out["temperature"] = float(min(max(float(out["temperature"]), 0.05), 2.5))
    if "top_p" in out:
        out["top_p"] = float(min(max(float(out["top_p"]), 0.01), 1.0))
    if "top_k" in out:
        out["top_k"] = int(min(max(int(out["top_k"]), 0), 200))
    if "num_beam_groups" in out:
        out["num_beam_groups"] = int(min(max(int(out["num_beam_groups"]), 1), out["num_beams"]))
        # transformers requires num_beams % num_beam_groups == 0.
        while out["num_beam_groups"] > 1 and out["num_beams"] % out["num_beam_groups"] != 0:
            out["num_beam_groups"] -= 1
    if "diversity_penalty" in out:
        out["diversity_penalty"] = float(min(max(float(out["diversity_penalty"]), 0.0), 5.0))
        if out.get("num_beam_groups", 1) <= 1:
            # diversity_penalty is only valid (and only has effect) in
            # group-beam-search; drop it rather than let generate() raise.
            out.pop("diversity_penalty")
    return out


def simplify(model, tok, sources: List[str], gen_kwargs: dict, device) -> List[str]:
    """Decode a simplification for every source with the agent's config.

    The FIXED task prefix, source truncation (MAX_INPUT_TOKENS), batching, and the
    frozen model are all fixed; only `gen_kwargs` (the agent's decode config) varies.
    """
    import torch

    gk = _sanitize_gen_kwargs(gen_kwargs)
    preds: List[str] = []
    for i in range(0, len(sources), GEN_BATCH_SIZE):
        batch = [SRC_PREFIX + s for s in sources[i:i + GEN_BATCH_SIZE]]
        enc = tok(
            batch,
            max_length=MAX_INPUT_TOKENS,
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
    return [p.strip() for p in preds]


# ---------------------------------------------------------------------------
# Degenerate reference policies (used by the policy task's baselines / anchors;
# NOT the agent's real model decode)
# ---------------------------------------------------------------------------
def copy_input(sources: List[str]) -> List[str]:
    """IDENTITY / copy-input baseline: return the complex sentence unchanged.

    Makes NO edits -> 0 on SARI's Add and Delete components (only Keep is high) ->
    LOW SARI. The standard low-effort simplification reference the literature
    reports (~20 SARI on ASSET)."""
    return [s.strip() for s in sources]


def truncate_tail(sources: List[str], keep_ratio: float = 0.75) -> List[str]:
    """TRUNCATION baseline: keep only the first `keep_ratio` of the words.

    Deletes the sentence tail -> non-zero SARI Delete, but cannot paraphrase or
    add, and deletes arbitrary content -> lands ABOVE copy-input but well BELOW a
    real meaning-preserving simplifier. The classic "truncation gets a
    surprisingly-OK SARI" baseline."""
    out = []
    for s in sources:
        w = s.split()
        k = max(1, int(round(len(w) * keep_ratio)))
        out.append(" ".join(w[:k]))
    return out


# ---------------------------------------------------------------------------
# Scoring (SARI is the primary metric; BLEU + length ratio are diagnostics)
# ---------------------------------------------------------------------------
def score_sari(sources: List[str], preds: List[str],
               references: List[List[str]]) -> float:
    """Corpus SARI (0-100, higher is better) via the vendored SARI implementation
    (Xu et al. 2016; faithful port of the HuggingFace ``evaluate`` metric)."""
    from sari import corpus_sari

    return float(corpus_sari(sources, preds, references))


def bleu_corpus(preds: List[str], references: List[List[str]]) -> float:
    """Self-contained corpus BLEU-4 (0-100) — an ADEQUACY / meaning-preservation
    diagnostic (a good simplification stays close to a reference). NOT the primary
    score (SARI is), reported only as a sanity trace so over-deletion is visible."""
    import math
    import re
    from collections import Counter

    def tok(s: str):
        return re.findall(r"\w+|[^\w\s]", s.lower())

    def ng(t, n):
        return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))

    if not preds:
        return 0.0
    total = 0.0
    for h, rs in zip(preds, references):
        ht = tok(h)
        rr = [tok(r) for r in rs] or [ht]
        if not ht:
            continue
        log_sum = 0.0
        ok = True
        for n in range(1, 5):
            hn = ng(ht, n)
            hd = sum(hn.values())
            if hd == 0:
                ok = False
                break
            mx = Counter()
            for r in rr:
                for g, c in ng(r, n).items():
                    if c > mx.get(g, 0):
                        mx[g] = c
            clip = sum(min(c, mx.get(g, 0)) for g, c in hn.items())
            if clip <= 0:
                ok = False
                break
            log_sum += math.log(clip / hd)
        if not ok:
            continue
        c = len(ht)
        r = min((len(x) for x in rr), key=lambda rl: (abs(rl - c), rl))
        bp = 1.0 if c > r else math.exp(1 - r / max(1, c))
        total += bp * math.exp(log_sum / 4)
    return 100.0 * total / max(1, len(preds))


def mean_pred_len_words(preds: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(len(p.split()) for p in preds) / len(preds)


def length_ratio(sources: List[str], preds: List[str]) -> float:
    """Mean output/input word-count ratio (diagnostic trace, NOT scored)."""
    if not sources:
        return 0.0
    tot = 0.0
    for s, p in zip(sources, preds):
        ns = max(1, len(s.split()))
        tot += len(p.split()) / ns
    return tot / len(sources)
