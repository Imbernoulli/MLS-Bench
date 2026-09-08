"""Downstream task evaluation (MATH, HumanEval) for masked diffusion LMs.

Following the KLASS evaluation protocol (Kim et al., NeurIPS 2025):
  https://github.com/shkim0116/KLASS
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_CONFIGS = {
    "llada": {"path": os.environ.get("LLADA_INSTRUCT_PATH", "/data/llada-instruct"),
              "mask_id": 126336},
    "dream": {"path": "Dream-org/Dream-v0-Instruct-7B", "mask_id": None},
}


def load_instruct_model(name: str, device: str = "cuda"):
    from transformers import AutoModel, AutoTokenizer
    cfg = MODEL_CONFIGS[name]
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    mdl = AutoModel.from_pretrained(cfg["path"], trust_remote_code=True,
                                    torch_dtype=torch.bfloat16).to(device).eval()
    mid = cfg["mask_id"] or getattr(mdl.config, "mask_token_id", None) \
                        or getattr(tok, "mask_token_id", None)
    assert mid is not None
    return mdl, tok, int(mid)


def get_num_transfer_tokens(mask_index, steps):
    """Uniform schedule: mask_num / steps tokens per step (+1 remainder)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps,
                      device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, :remainder[i]] += 1
    return out


# ====================================================================
# EDITABLE REGION — DemaskDecoder
# ====================================================================



class DemaskDecoder:
    """Masked-diffusion decoding strategy. Reference: KLASS (Kim et al.,
    NeurIPS 2025).

    Performs semi-autoregressive decoding in blocks of length `block_length`.
    Within each block, at each step it decides which masked positions to
    unmask based on confidence and stability (KL divergence across steps).

    Params (set in __init__):
      mask_id            : mask token id (LLaDA=126336, Dream=from tokenizer)
      temperature        : Gumbel-max sampling temperature (0 = argmax)
      conf_threshold     : min top-1 prob for a position to be "confident"
      kl_threshold       : max KL over history for a position to be "stable"
      history_length     : # recent steps to require stability over

    decode() returns (x_output [1, prompt_len+gen_len], used_steps).
    """

    def __init__(self, mask_id: int, temperature: float = 0.0,
                 conf_threshold: float = 0.9, kl_threshold: float = 0.01,
                 history_length: int = 2):
        self.mask_id = mask_id
        self.temperature = temperature
        self.conf_threshold = conf_threshold
        self.kl_threshold = kl_threshold
        self.history_length = history_length

    @torch.no_grad()
    def decode(self, model, input_ids, gen_length: int, steps: int,
               block_length: int):
        mid = self.mask_id
        x = torch.full((1, input_ids.shape[1] + gen_length), mid,
                       dtype=torch.long, device=model.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        V = model.lm_head.out_features if hasattr(model, "lm_head") \
                                       else model.config.vocab_size
        kl_hist = torch.zeros((1, x.shape[1], self.history_length),
                              dtype=torch.float64, device=x.device)
        p_prev = torch.zeros((1, x.shape[1], V), dtype=torch.float64,
                             device=x.device)
        used = 0

        for b in range(num_blocks):
            bs = input_ids.shape[1] + b * block_length
            be = bs + block_length
            num_xfer = get_num_transfer_tokens(
                (x[:, bs:be] == mid), steps_per_block)

            for step in range(steps_per_block):
                mask_idx = (x == mid)
                block_m = torch.zeros_like(mask_idx)
                block_m[:, bs:be] = True
                mask_idx = mask_idx & block_m
                if not mask_idx.any():
                    break

                logits = model(x).logits
                p_curr = F.softmax(logits.to(torch.float64), dim=-1)
                x0 = torch.argmax(p_curr, dim=-1)
                conf = torch.gather(p_curr, -1, x0.unsqueeze(-1)).squeeze(-1)

                eps = 1e-12
                kl = (p_curr * (torch.log(p_curr + eps)
                                - torch.log(p_prev + eps))).sum(-1)
                kl_hist = torch.roll(kl_hist, -1, dims=-1)
                kl_hist[..., -1] = kl
                p_prev = p_curr.clone()

                # KLASS: ready = stable ∩ confident ∩ still-masked
                if step >= self.history_length - 1:
                    stable = torch.all(kl_hist < self.kl_threshold, dim=-1)
                else:
                    stable = torch.zeros_like(conf, dtype=torch.bool)
                ready = stable & (conf > self.conf_threshold) & mask_idx

                xfer = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(ready.shape[0]):
                    rdy = torch.where(ready[j])[0]
                    if len(rdy) > 0:
                        xfer[j, rdy] = True
                    else:
                        c = conf[j].clone()
                        c[~mask_idx[j]] = -float("inf")
                        _, topk = torch.topk(c, int(num_xfer[j, step].item()))
                        xfer[j, topk] = True
                x = torch.where(xfer, x0, x)
                used += 1
        return x, used


# ====================================================================
# END OF EDITABLE REGION
# ====================================================================


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_math(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_humaneval(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# MATH evaluation (uses klass_utils extract_math_answer + compare_answers)
# ---------------------------------------------------------------------------

def _import_klass_utils():
    """Import klass_utils from task data dir (mounted at /workspace/_task)."""
    task_dir = os.environ.get("TASK_DIR", "/workspace/_task")
    sys.path.insert(0, os.path.join(task_dir, "data"))
    import klass_utils as ku
    return ku


def eval_math(model, tokenizer, decoder: DemaskDecoder, problems: list[dict],
              gen_length: int, steps: int, block_length: int):
    ku = _import_klass_utils()
    sys_msg = ("Your task is to answer the question below. Give step by step "
               "reasoning before you answer, and when you're ready to answer, "
               "please use the format 'The final answer is'.")
    correct = 0
    total_steps = 0
    for i, ex in enumerate(problems):
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": ex["problem"]}]
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                               tokenize=False)
        input_ids = torch.tensor(tokenizer(prompt)["input_ids"],
                                 device=model.device).unsqueeze(0)
        gt = ku.extract_math_answer(ex["problem"], ex["solution"])
        x_out, used = decoder.decode(model, input_ids, gen_length, steps,
                                     block_length)
        gen_text = tokenizer.batch_decode(
            x_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        pred = ku.extract_math_answer(ex["problem"], gen_text)
        is_correct = ku.compare_answers(ex["problem"], gt, pred)
        if i < 2:
            print(f"[DEBUG] math example {i}:\n"
                  f"  problem: {ex['problem'][:150]}\n"
                  f"  gt={gt}\n"
                  f"  gen (first 400 chars): {gen_text[:400]}\n"
                  f"  pred={pred} correct={is_correct}", flush=True)
        if is_correct:
            correct += 1
        total_steps += used
        if (i + 1) % 10 == 0:
            print(f"TRAIN_METRICS: math {i+1}/{len(problems)} "
                  f"acc={correct/(i+1):.3f} "
                  f"avg_steps={total_steps/(i+1):.1f}", flush=True)
    return correct / max(len(problems), 1), total_steps / max(len(problems), 1)


# ---------------------------------------------------------------------------
# HumanEval evaluation (uses klass_utils evaluate_task)
# ---------------------------------------------------------------------------

def _run_humaneval(code: str, test: str, entry_point: str) -> bool:
    """Exec code + test + check(entry_point) in fresh namespace."""
    try:
        ns: dict = {}
        exec(code + "\n" + test + f"\ncheck({entry_point})\n", ns)
        return True
    except Exception:
        return False


def check_humaneval_code(code: str, problem: dict, timeout: float = 3.0) -> bool:
    import multiprocessing
    entry = problem["entry_point"]
    # If generated code lacks the function def, prepend problem prompt
    # (which provides function signature + docstring).
    if f"def {entry}" not in code:
        code = problem["prompt"] + code
    with multiprocessing.Pool(processes=1) as pool:
        res = pool.apply_async(_run_humaneval, (code, problem["test"], entry))
        try:
            return bool(res.get(timeout=timeout))
        except Exception:
            return False


def eval_humaneval(model, tokenizer, decoder: DemaskDecoder,
                   problems: list[dict], gen_length: int, steps: int,
                   block_length: int):
    passed = 0
    total_steps = 0
    for i, p in enumerate(problems):
        msgs = [{"role": "system", "content": "You complete only Python code."},
                {"role": "user", "content": p["prompt"]}]
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                               tokenize=False)
        input_ids = torch.tensor(tokenizer(prompt)["input_ids"],
                                 device=model.device).unsqueeze(0)
        x_out, used = decoder.decode(model, input_ids, gen_length, steps,
                                     block_length)
        gen_text = tokenizer.batch_decode(
            x_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        eos = tokenizer.eos_token or ""
        if eos:
            gen_text = gen_text.split(eos)[0]
        m = re.search(r"```(?:python)?\n(.*?)(?:```|$)", gen_text, re.DOTALL)
        code = m.group(1).strip() if m else gen_text.strip()
        if i < 2:
            print(f"[DEBUG] humaneval {p['entry_point']}:\n"
                  f"gen (first 300 chars): {gen_text[:300]}\n"
                  f"code (first 200 chars): {code[:200]}", flush=True)
        ok = check_humaneval_code(code, p, timeout=3)
        if ok:
            passed += 1
        total_steps += used
        if (i + 1) % 10 == 0:
            print(f"TRAIN_METRICS: humaneval {i+1}/{len(problems)} "
                  f"pass@1={passed/(i+1):.3f} "
                  f"avg_steps={total_steps/(i+1):.1f}", flush=True)
    return passed / max(len(problems), 1), total_steps / max(len(problems), 1)


# ---------------------------------------------------------------------------
# Open-ended text generation evaluation (gen_ppl, MAUVE, entropy, rep2)
# ---------------------------------------------------------------------------

def _truncate_at_eos(text: str, eos_tokens=("</s>", "<|endoftext|>", "<|im_end|>")):
    for eos in eos_tokens:
        idx = text.find(eos)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


def compute_conditional_gen_ppl(prefix_texts, gen_texts, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import math as _m
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
    mdl = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2-large").to(device).eval()
    total_loss, total_tokens = 0.0, 0
    for prefix, gen in zip(prefix_texts, gen_texts):
        if not gen.strip():
            continue
        p_ids = tok.encode(prefix, add_special_tokens=False)
        g_ids = tok.encode(gen, add_special_tokens=False)
        all_ids = (p_ids + g_ids)[:1024]
        if len(p_ids) >= len(all_ids):
            continue
        ids = torch.tensor([all_ids], device=device)
        with torch.no_grad():
            logits = mdl(ids).logits[:, :-1, :]
        labels = ids[:, 1:]
        start = max(len(p_ids) - 1, 0)
        loss = F.cross_entropy(
            logits[:, start:, :].reshape(-1, logits.shape[-1]),
            labels[:, start:].reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += labels[:, start:].numel()
    del mdl
    torch.cuda.empty_cache()
    return _m.exp(total_loss / total_tokens) if total_tokens else float("inf")


def compute_mauve(gen_texts, ref_texts):
    try:
        import mauve
        r = mauve.compute_mauve(p_text=ref_texts, q_text=gen_texts,
                                device_id=0 if torch.cuda.is_available() else -1,
                                max_text_length=512, verbose=False,
                                featurize_model_name="openai-community/gpt2-large")
        return float(r.mauve)
    except Exception as e:
        print(f"[WARN] MAUVE failed: {e}", flush=True)
        return 0.0


def compute_entropy_rep2(texts):
    from collections import Counter
    import math as _m
    all_bigrams = []
    rep_ratios = []
    for t in texts:
        words = t.split()
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        all_bigrams.extend(bigrams)
        if bigrams:
            rep_ratios.append(1.0 - len(set(bigrams)) / len(bigrams))
        else:
            rep_ratios.append(0.0)
    ent = 0.0
    if all_bigrams:
        c = Counter(all_bigrams)
        tot = sum(c.values())
        for v in c.values():
            p = v / tot
            if p > 0:
                ent -= p * _m.log2(p)
    rep2 = sum(rep_ratios) / max(len(rep_ratios), 1)
    return ent, rep2


def eval_text(model, tokenizer, decoder: DemaskDecoder, raw_texts: list[str],
              prefix_len: int, gen_length: int, steps: int, block_length: int,
              n_samples: int, seed: int):
    """Prefix-conditioned C4 continuation. Reports gen_ppl/MAUVE/entropy/rep2."""
    import random as _r
    rng = _r.Random(seed)
    if len(raw_texts) > n_samples:
        raw_texts = rng.sample(raw_texts, n_samples)

    # Build prefix prompts (raw, no chat template — Dream-Instruct as a base LM)
    prefix_ids_list, prefix_texts, valid_refs = [], [], []
    for txt in raw_texts:
        ids = tokenizer.encode(txt, add_special_tokens=False)
        if len(ids) >= prefix_len + gen_length:
            pids = ids[:prefix_len]
            prefix_ids_list.append(pids)
            prefix_texts.append(tokenizer.decode(pids, skip_special_tokens=True))
            ref_ids = ids[prefix_len:prefix_len + gen_length]
            valid_refs.append(tokenizer.decode(ref_ids, skip_special_tokens=True))
    print(f"[INFO] kept {len(prefix_ids_list)}/{len(raw_texts)} texts "
          f"long enough for prefix={prefix_len}+gen={gen_length}", flush=True)

    gen_texts, total_used = [], 0
    for i, pids in enumerate(prefix_ids_list):
        ids = torch.tensor([pids], dtype=torch.long, device=model.device)
        x_out, used = decoder.decode(model, ids, gen_length, steps, block_length)
        gen = tokenizer.decode(x_out[0, ids.shape[1]:].tolist(),
                               skip_special_tokens=True)
        gen = _truncate_at_eos(gen)
        gen_texts.append(gen)
        total_used += used
        if (i + 1) % 10 == 0:
            print(f"TRAIN_METRICS: text {i+1}/{len(prefix_ids_list)} "
                  f"avg_steps={total_used/(i+1):.1f}", flush=True)

    avg_steps = total_used / max(len(prefix_ids_list), 1)
    print("[INFO] unloading gen model, computing GPT-2 ppl...", flush=True)
    del model
    torch.cuda.empty_cache()

    ppl = compute_conditional_gen_ppl(prefix_texts, gen_texts, "cuda")
    mauve = compute_mauve(gen_texts, valid_refs)
    entropy, rep2 = compute_entropy_rep2(gen_texts)
    return ppl, mauve, entropy, rep2, avg_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["math", "humaneval", "text"],
                        required=True)
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--conf-threshold", type=float, default=0.9)
    parser.add_argument("--kl-threshold", type=float, default=0.01)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--n-samples", type=int, default=0,
                        help="0 = use all problems")
    parser.add_argument("--prefix-len", type=int, default=32,
                        help="Prefix length (text task)")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading {args.model}...", flush=True)
    model, tokenizer, mask_id = load_instruct_model(args.model, device)

    decoder = DemaskDecoder(
        mask_id=mask_id,
        temperature=args.temperature,
        conf_threshold=args.conf_threshold,
        kl_threshold=args.kl_threshold,
        history_length=args.history_length,
    )

    print(f"[INFO] task={args.task} steps={args.steps} "
          f"gen_length={args.gen_length} block_length={args.block_length}",
          flush=True)

    if args.task == "math":
        problems = load_math(args.data_path)
        if args.n_samples > 0:
            problems = problems[:args.n_samples]
        acc, avg_steps = eval_math(
            model, tokenizer, decoder, problems,
            args.gen_length, args.steps, args.block_length)
        print(f"TEST_METRICS: accuracy={acc:.4f} avg_steps={avg_steps:.2f} "
              f"n_samples={len(problems)}", flush=True)
    elif args.task == "humaneval":
        problems = load_humaneval(args.data_path)
        if args.n_samples > 0:
            problems = problems[:args.n_samples]
        acc, avg_steps = eval_humaneval(
            model, tokenizer, decoder, problems,
            args.gen_length, args.steps, args.block_length)
        print(f"TEST_METRICS: accuracy={acc:.4f} avg_steps={avg_steps:.2f} "
              f"n_samples={len(problems)}", flush=True)
    else:  # text
        with open(args.data_path) as f:
            texts = json.load(f)
        n = args.n_samples if args.n_samples > 0 else 256
        ppl, mauve, ent, rep2, avg_steps = eval_text(
            model, tokenizer, decoder, texts,
            args.prefix_len, args.gen_length, args.steps, args.block_length,
            n_samples=n, seed=args.seed)
        print(f"TEST_METRICS: gen_ppl={ppl:.4f} mauve={mauve:.4f} "
              f"entropy={ent:.4f} rep2={rep2:.4f} avg_steps={avg_steps:.2f} "
              f"n_samples={n}", flush=True)


if __name__ == "__main__":
    main()
