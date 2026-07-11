#!/usr/bin/env python3
"""Run-wide demonstration-prefix evaluation with a fixed generation pipeline."""
from __future__ import annotations
import argparse, time
import common


TASK_ID = "codegen-fewshot-priming"
DEMO_TOKEN_CAP = 256


def _token_count(tok, text):
    encoded = tok(text, add_special_tokens=False)
    input_ids = getattr(encoded, "input_ids", None)
    if not isinstance(input_ids, list) or any(type(token) is not int for token in input_ids):
        raise RuntimeError("tokenizer returned malformed demonstration token ids")
    return len(input_ids)


def _base(prob):
    return ("Now complete the following Python function. Return ONLY the complete "
            "function inside a single ```python code block, with no example "
            "usage and no explanation.\n\n" + prob["prompt"])


def _compiles(src):
    try:
        compile(src, "<p>", "exec"); return bool(src.strip())
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=common.EXPECTED_PROBLEMS)
    args = ap.parse_args()
    common.set_seeds(args.seed); t0 = time.time()
    common.start_executor()
    tok, _ = common.load_model()
    problems = common.load_problems(args.n)
    common.seal_private_data()
    fewshot = common.load_surface(args.solution, "fewshot")
    demonstrations = common.call_policy_without_generation(fewshot)
    if not isinstance(demonstrations, str):
        raise TypeError("fewshot() must return text (empty text is allowed)")
    demo_tokens = _token_count(tok, demonstrations)
    if demo_tokens > DEMO_TOKEN_CAP:
        raise ValueError(f"fewshot() exceeds the fixed {DEMO_TOKEN_CAP}-token cap")
    common.emit_protocol(TASK_ID, args.seed, problems)
    n = len(problems); n_hidden = 0; n_parse = 0
    for i, prob in enumerate(problems):
        setup = prob.get("test_setup", "")
        base_prompt = _base(prob)
        prompt = demonstrations + "\n\n" + base_prompt if demonstrations else base_prompt
        raw = common.generate(common.GenConfig(
            prompt=prompt, n_samples=1, do_sample=False,
            max_new_tokens=512), seed=args.seed)
        prog = common.extract_code(raw[0])
        parse_ok = _compiles(prog or "")
        hid_ok = common.passes_all(prog or "", prob["hidden_tests"], setup)
        n_parse += int(parse_ok); n_hidden += int(hid_ok)
        common.emit_item(i + 1, hid_ok, parsed=parse_ok)
        if i < 3:
            print(f"CG_SAMPLE i={i} id={prob['task_id']} demo_tokens={demo_tokens} "
                  f"parse_ok={parse_ok} hid_ok={hid_ok}", flush=True)
        if (i + 1) % 20 == 0:
            common.emit_progress(i + 1, n_hidden)
    dt = time.time() - t0
    print(f"CG_METRICS task={TASK_ID} pass_at_1={n_hidden/n:.6f} "
          f"parse_rate={n_parse/n:.6f} n={n} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
