"""Long-context LLM inference eval with Qwen2.5-1.5B-Instruct + sparse attention.

Three sub-environments selected by ``--env``:

  niah_8k                     Needle-In-A-Haystack at 8K context. Inserts a
                              "magic number" sentence at varying depths
                              inside an 8K filler; measures retrieval acc.

  longbench_qasper            Single-doc QA (LongBench Qasper). Reads a
                              4K-16K paper context, generates an answer,
                              scores token-overlap F1 vs the reference
                              answers (capped at 8K context for memory).

  longbench_multifieldqa_en   Long-document multi-field QA (LongBench
                              MultiFieldQA-EN). 4-8K context, generation-
                              based, scored by token-overlap F1 (LongBench
                              ``qa_f1_score`` convention).

In all 3 envs the agent's ``SparseAttention`` is monkey-patched into every
Qwen2Attention layer. All prompts are wrapped with the model's chat
template (Qwen2.5-Instruct expects this for instruction following).

Qwen2.5-1.5B-Instruct's native context is 32K, so no RoPE scaling is
needed at the 8K target.
"""

import argparse
import json
import math
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/workspace/sparse-attn-eval")

from custom_sparse_attn import SparseAttention
from harness import (
    patch_model, enforce_budget, density_window, reset_density,
    apply_ntk_rope_scaling,
)


# ── shared helpers ────────────────────────────────────────────────────────────

def load_filler_text(path):
    p = Path(path)
    if p.exists():
        return p.read_text()
    # last-ditch fallback inside /data/longctx
    cache = Path("/data/longctx")
    for cand in ("wikitext103_test.txt", "wikitext2_test.txt"):
        f = cache / cand
        if f.exists():
            return f.read_text()
    raise FileNotFoundError(f"no filler text at {path} or /data/longctx")


def cuda_device_summary():
    if not torch.cuda.is_available():
        return "cpu"
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return f"{props.name} cc={props.major}.{props.minor} cuda={torch.version.cuda}"


def configure_cuda_backend():
    if not torch.cuda.is_available():
        print("CUDA_DIAGNOSTIC device=cpu dtype=float32", flush=True)
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception as exc:
        print(f"CUDA_DIAGNOSTIC tf32_config_failed={exc}", flush=True)
    for name, enabled in (
        ("enable_flash_sdp", True),
        ("enable_math_sdp", True),
        ("enable_mem_efficient_sdp", False),
        ("enable_cudnn_sdp", False),
    ):
        setter = getattr(torch.backends.cuda, name, None)
        if setter is None:
            continue
        try:
            setter(enabled)
        except Exception as exc:
            print(f"CUDA_DIAGNOSTIC {name}_failed={exc}", flush=True)
    print(f"CUDA_DIAGNOSTIC device={cuda_device_summary()} backend=sdpa flash=on math=on mem_efficient=off cudnn=off", flush=True)


def select_model_dtype():
    # Default fp16 to stay numerically consistent with the recorded baselines,
    # which were all computed in fp16. The H20-specific SIGFPE on the fp16
    # prefill path is opt-out via SPARSE_ATTN_DTYPE=bf16 (or auto). NOTE: changing
    # dtype changes metric values -- if you switch, re-baseline the whole task so
    # the leaderboard stays self-consistent.
    if not torch.cuda.is_available():
        return torch.float32
    choice = os.environ.get("SPARSE_ATTN_DTYPE", "fp16").lower()
    if choice in ("bf16", "bfloat16"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("CUDA_DIAGNOSTIC bf16_requested_but_unsupported=true fallback_dtype=float16", flush=True)
        return torch.float16
    if choice == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16


def sync_cuda(label):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"CUDA_SYNC label={label}", flush=True)


def build_model(model_path, target_ctx, dtype=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    configure_cuda_backend()
    dtype = dtype or select_model_dtype()
    attn_impl = os.environ.get("SPARSE_ATTN_IMPL", "sdpa")
    print(
        f"Loading {model_path} (target_ctx={target_ctx}, dtype={dtype}, "
        f"attn_impl={attn_impl}, device={cuda_device_summary()})...",
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Qwen2.5 native max_position_embeddings is 32768 — 8K target is well
    # within the supported range, so no RoPE scaling.
    native_ctx = int(getattr(model.config, "max_position_embeddings", 32768))
    if native_ctx <= 0:
        print(f"NUMERIC_GUARD label=native_ctx value={native_ctx} fallback={target_ctx}", flush=True)
        native_ctx = max(target_ctx, 1)
    if target_ctx > native_ctx:
        scale = target_ctx / max(native_ctx, 1)
        apply_ntk_rope_scaling(model, scale)
        model.config.max_position_embeddings = max(target_ctx, native_ctx)
    print("[run_llm] model load complete", flush=True)
    return model, tok


def make_factory(args):
    def factory(head_dim, num_heads):
        return SparseAttention(
            head_dim=head_dim, num_heads=num_heads, block_size=64,
            density_budget=args.density_budget,
        )
    return factory


def _format_chat(tok, user_msg):
    """Render a single-turn user message via the tokenizer's chat template
    and return input_ids (1, L) on CPU. The template adds the assistant
    generation prefix so the model directly produces the answer.
    """
    messages = [{"role": "user", "content": user_msg}]
    return tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    )


def _truncate_ids_for_chat(tok, user_msg, max_len):
    """Render the chat-templated prompt; if it exceeds max_len, truncate the
    user message text from the right (drop tail tokens of the body) until it
    fits, then re-render. Used for LongBench Qasper where the paper context
    can exceed the 8K window."""
    ids = _format_chat(tok, user_msg)
    if ids.size(1) <= max_len:
        return ids
    # Token-level truncation of the user message body. We binary-search a
    # safe character cut so the chat template re-fits.
    body = user_msg
    lo, hi = 0, len(body)
    best = ""
    # Estimate ~3 chars/token for English; iterate down.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cand = body[:mid]
        cand_ids = _format_chat(tok, cand)
        if cand_ids.size(1) <= max_len:
            best = cand
            lo = mid
        else:
            hi = mid - 1
    return _format_chat(tok, best)


# ── env: niah_8k ──────────────────────────────────────────────────────────────

def _build_niah_prompt_ids(tok, filler_token_pool, needle_text, depth_pct,
                            question, target_total_len, instr_text):
    """Build a NIAH prompt at exactly ``target_total_len`` tokens (or shorter
    if the filler pool is smaller).

    Approach:
      1. Tokenize the static parts (instruction header + question + answer
         prefix) once. Tokenize the needle.
      2. Render the chat template with a marker placeholder where the body
         goes, measure the chat-template scaffolding overhead.
      3. Solve for ``body_tokens = target - scaffold - instr - needle - question``.
      4. Splice ``body_tokens`` tokens from ``filler_token_pool`` with the
         needle inserted at ``depth_pct`` of the body.
      5. Decode body to text, assemble final user_text, render chat template.

    This keeps the question + answer prefix intact at every depth — fixing
    the bug where right-side chat-template truncation was removing the
    question and the model was just continuing the passage.
    """
    needle_ids = tok(needle_text, add_special_tokens=False,
                     return_tensors="pt").input_ids[0]
    instr_ids = tok(instr_text, add_special_tokens=False,
                    return_tensors="pt").input_ids[0]
    qa_text = f"\n\nQuestion: {question}\nAnswer concisely:"
    qa_ids = tok(qa_text, add_special_tokens=False,
                 return_tensors="pt").input_ids[0]
    pre_text = "\n\nPassage:\n"
    pre_ids = tok(pre_text, add_special_tokens=False,
                  return_tensors="pt").input_ids[0]

    # Measure chat-template scaffolding overhead by rendering with an empty body.
    empty_user = instr_text + pre_text + qa_text
    empty_ids = _format_chat(tok, empty_user)[0]
    # Body budget = target - empty rendered length
    body_budget = target_total_len - empty_ids.size(0) - needle_ids.size(0)
    body_budget = max(0, body_budget)
    body_budget = min(body_budget, filler_token_pool.size(0))

    # Splice needle at depth.
    insert_at = int(round(max(0.0, min(1.0, depth_pct)) * body_budget))
    insert_at = max(0, min(insert_at, body_budget))
    body_ids = torch.cat([
        filler_token_pool[:insert_at],
        needle_ids,
        filler_token_pool[insert_at:body_budget],
    ], dim=0)
    body_text = tok.decode(body_ids, skip_special_tokens=True)

    user_text = instr_text + pre_text + body_text + qa_text
    return _format_chat(tok, user_text)


@torch.no_grad()
def env_niah(args):
    cases = [json.loads(l) for l in Path(args.niah_cases).read_text().splitlines() if l.strip()]
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    text = load_filler_text(args.filler_text)

    model, tok = build_model(args.model_path, args.context_len)
    patch_model(model, "llm", make_factory(args))

    # Tokenize a long stretch of filler once. Need at least ``context_len``
    # tokens; oversample 2× to be safe and tile if the corpus is short.
    target_tokens = args.context_len * 2
    char_oversample = target_tokens * 8  # ~8 chars/token upper bound
    if len(text) < char_oversample:
        text = text * (char_oversample // max(len(text), 1) + 1)
    filler_token_pool = tok(
        text[:char_oversample], add_special_tokens=False, return_tensors="pt",
    ).input_ids[0]

    instr_text = (
        "You are given a long passage. Read it carefully and then answer "
        "the question at the end based only on information in the passage."
    )

    correct = 0
    total = 0
    reset_density()
    t0 = time.time()
    with density_window():
        for ci, case in enumerate(cases):
            needle = case["needle"]
            question = case["question"]
            answer_str = case["answer"]
            depth = float(case["depth_pct"])
            ids = _build_niah_prompt_ids(
                tok, filler_token_pool, needle, depth, question,
                args.context_len - 32, instr_text,
            ).cuda()

            if ci == 0:
                print(
                    "FIRST_FORWARD_DIAGNOSTIC "
                    f"env=niah_8k prompt_tokens={ids.size(1)} dtype={next(model.parameters()).dtype} "
                    f"device={next(model.parameters()).device}",
                    flush=True,
                )
                sync_cuda("niah_before_first_generate")
            out = model.generate(
                ids,
                max_new_tokens=24,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                use_cache=False,  # we don't support KV cache in custom attn
            )
            gen_ids = out[0, ids.size(1):]
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)
            ok = answer_str in gen_text
            correct += int(ok)
            total += 1
            if (ci + 1) % 5 == 0 or ci == len(cases) - 1:
                print(f"TRAIN_METRICS case={ci+1}/{len(cases)} "
                      f"acc={correct/total:.4f} last_match={int(ok)}",
                      flush=True)
    elapsed = time.time() - t0
    acc = correct / max(total, 1)
    stats = enforce_budget("niah_8k", args.density_budget,
                           allow_dense=args.allow_dense)
    print(f"TEST_METRICS niah_acc={acc:.4f} "
          f"niah_density={stats['mean']:.4f} niah_time={elapsed:.1f}",
          flush=True)


# ── env: longbench_qasper ─────────────────────────────────────────────────────

def _normalize_answer(s):
    """Lower / strip punctuation / whitespace (LongBench convention)."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(s.split())


def _f1(pred, gold):
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def env_qasper(args):
    examples = [json.loads(l) for l in Path(args.qasper_jsonl).read_text().splitlines()
                if l.strip()]
    if args.max_cases > 0:
        examples = examples[:args.max_cases]

    model, tok = build_model(args.model_path, args.max_context_len)
    patch_model(model, "llm", make_factory(args))

    f1_sum = 0.0
    n = 0
    reset_density()
    t0 = time.time()
    with density_window():
        for ei, ex in enumerate(examples):
            ctx = ex["context"]
            q = ex["input"]
            golds = ex["answers"] if isinstance(ex["answers"], list) else [ex["answers"]]
            golds = [g for g in golds if g]
            if not golds:
                continue

            # Prompt format follows LongBench Qasper: long context, then Q+A.
            # We truncate the *paper context* from the right so the question
            # and answer prefix always stay in the prompt — naive whole-prompt
            # truncation would silently drop the question and the model
            # would just continue the paper.
            instr_text = (
                "You are given a scientific paper. Answer the question based "
                "on the paper. Be brief."
            )
            qa_text = f"\n\nQuestion: {q}\nAnswer:"
            pre_text = "\n\nPaper:\n"
            target_len = args.max_context_len - args.max_new_tokens
            # Measure scaffold (chat template + instr + qa) with empty body.
            scaffold_ids = _format_chat(tok, instr_text + pre_text + qa_text)[0]
            body_budget = max(0, target_len - scaffold_ids.size(0))
            ctx_ids = tok(ctx, add_special_tokens=False,
                          return_tensors="pt").input_ids[0]
            ctx_ids = ctx_ids[:body_budget]
            ctx_truncated = tok.decode(ctx_ids, skip_special_tokens=True)
            user_text = instr_text + pre_text + ctx_truncated + qa_text
            ids = _format_chat(tok, user_text).cuda()
            if ei == 0:
                print(
                    "FIRST_FORWARD_DIAGNOSTIC "
                    f"env=longbench_qasper prompt_tokens={ids.size(1)} dtype={next(model.parameters()).dtype} "
                    f"device={next(model.parameters()).device}",
                    flush=True,
                )
                sync_cuda("qasper_before_first_generate")
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                use_cache=False,
            )
            gen = tok.decode(out[0, ids.size(1):], skip_special_tokens=True).strip()
            # Truncate at first newline (model often continues with another Q/A).
            gen = gen.split("\n")[0].strip()
            best_f1 = max(_f1(gen, g) for g in golds)
            f1_sum += best_f1
            n += 1
            if (ei + 1) % 5 == 0 or ei == len(examples) - 1:
                print(f"TRAIN_METRICS case={ei+1}/{len(examples)} "
                      f"f1={f1_sum/max(n,1):.4f} last_pred={gen[:60]!r}",
                      flush=True)
    elapsed = time.time() - t0
    f1 = f1_sum / max(n, 1)
    stats = enforce_budget("longbench_qasper", args.density_budget,
                           allow_dense=args.allow_dense)
    print(f"TEST_METRICS qasper_f1={f1:.4f} "
          f"qasper_density={stats['mean']:.4f} qasper_time={elapsed:.1f}",
          flush=True)


# ── env: longbench_multifieldqa_en ────────────────────────────────────────────

@torch.no_grad()
def env_multifieldqa_en(args):
    """LongBench MultiFieldQA-EN: long-document multi-field QA, F1 metric.

    Prompt template follows the LongBench ``pred.py`` format for
    ``multifieldqa_en``. Scoring uses the same token-level F1 helper as
    Qasper (lowercase + strip punctuation/articles, then unigram F1) — this
    is functionally equivalent to LongBench's ``qa_f1_score``.
    """
    examples = [json.loads(l) for l in Path(args.multifieldqa_jsonl).read_text().splitlines()
                if l.strip()]
    if args.max_cases > 0:
        examples = examples[:args.max_cases]

    model, tok = build_model(args.model_path, args.max_context_len)
    patch_model(model, "llm", make_factory(args))

    f1_sum = 0.0
    n = 0
    reset_density()
    t0 = time.time()
    with density_window():
        for ei, ex in enumerate(examples):
            ctx = ex["context"]
            q = ex["input"]
            golds = ex["answers"] if isinstance(ex["answers"], list) else [ex["answers"]]
            golds = [g for g in golds if g]
            if not golds:
                continue

            # LongBench-style prompt template for multifieldqa_en. We
            # split it into a prefix / context / suffix so we can truncate
            # the *context* from the right while preserving the question
            # and answer prefix at every length.
            prefix = (
                "You are an expert in answering questions based on context "
                "provided. Read the following context and answer the "
                "question:\n\n"
            )
            suffix = f"\n\nQuestion: {q}\n\nAnswer:"
            target_len = args.max_context_len - args.max_new_tokens
            scaffold_ids = _format_chat(tok, prefix + suffix)[0]
            body_budget = max(0, target_len - scaffold_ids.size(0))
            ctx_ids = tok(ctx, add_special_tokens=False,
                          return_tensors="pt").input_ids[0]
            ctx_ids = ctx_ids[:body_budget]
            ctx_truncated = tok.decode(ctx_ids, skip_special_tokens=True)
            user_text = prefix + ctx_truncated + suffix
            ids = _format_chat(tok, user_text).cuda()
            if ei == 0:
                print(
                    "FIRST_FORWARD_DIAGNOSTIC "
                    f"env=longbench_multifieldqa_en prompt_tokens={ids.size(1)} dtype={next(model.parameters()).dtype} "
                    f"device={next(model.parameters()).device}",
                    flush=True,
                )
                sync_cuda("multifieldqa_before_first_generate")
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                use_cache=False,
            )
            gen = tok.decode(out[0, ids.size(1):], skip_special_tokens=True).strip()
            gen = gen.split("\n")[0].strip()
            best_f1 = max(_f1(gen, g) for g in golds)
            f1_sum += best_f1
            n += 1
            if (ei + 1) % 5 == 0 or ei == len(examples) - 1:
                print(f"TRAIN_METRICS case={ei+1}/{len(examples)} "
                      f"f1={f1_sum/max(n,1):.4f} last_pred={gen[:60]!r}",
                      flush=True)
    elapsed = time.time() - t0
    f1 = f1_sum / max(n, 1)
    stats = enforce_budget("longbench_multifieldqa_en", args.density_budget,
                           allow_dense=args.allow_dense)
    print(f"TEST_METRICS multifieldqa_f1={f1:.4f} "
          f"multifieldqa_density={stats['mean']:.4f} "
          f"multifieldqa_time={elapsed:.1f}",
          flush=True)


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True,
                   choices=["niah_8k", "longbench_qasper",
                            "longbench_multifieldqa_en"])
    p.add_argument("--model-path", default="/data/qwen2.5-1.5b-instruct")
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    p.add_argument("--density-budget", type=float, default=0.25)
    p.add_argument("--allow-dense", action="store_true")
    # niah
    p.add_argument("--niah-cases", default="/data/niah/cases.jsonl")
    p.add_argument("--filler-text", default="/data/longctx/wikitext103_test.txt")
    p.add_argument("--context-len", type=int, default=8192)
    # qasper / multifieldqa
    p.add_argument("--qasper-jsonl", default="/data/longbench-qasper/qasper.jsonl")
    p.add_argument("--multifieldqa-jsonl",
                   default="/data/longbench-qasper/multifieldqa_en.jsonl")
    p.add_argument("--max-context-len", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-cases", type=int, default=50)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.env == "niah_8k":
        env_niah(args)
    elif args.env == "longbench_qasper":
        env_qasper(args)
    elif args.env == "longbench_multifieldqa_en":
        env_multifieldqa_en(args)


if __name__ == "__main__":
    main()
