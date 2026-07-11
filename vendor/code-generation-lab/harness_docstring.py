#!/usr/bin/env python3
"""Prompt-design evaluation with a fixed model, decoder, and extractor."""
from __future__ import annotations
import argparse, time
import common


TASK_ID = "codegen-docstring-design"
INPUT_TOKEN_CAP = 1024


def _prompt_view(prob):
    """Return the complete and exact policy-visible problem view."""
    return {"prompt": prob["prompt"], "entry_point": prob["entry_point"]}


def _chat_token_count(tok, prompt):
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tok(rendered, add_special_tokens=False)
    input_ids = getattr(encoded, "input_ids", None)
    if not isinstance(input_ids, list) or any(type(token) is not int for token in input_ids):
        raise RuntimeError("tokenizer returned malformed prompt token ids")
    return len(input_ids)


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
    build_prompt = common.load_surface(args.solution, "build_prompt")
    common.emit_protocol(TASK_ID, args.seed, problems)
    n = len(problems); n_hidden = 0; n_parse = 0
    for i, prob in enumerate(problems):
        setup = prob.get("test_setup", "")
        prompt = common.call_policy_without_generation(
            build_prompt, _prompt_view(prob)
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("build_prompt() must return non-empty text")
        input_tokens = _chat_token_count(tok, prompt)
        if input_tokens > INPUT_TOKEN_CAP:
            raise ValueError(
                f"build_prompt() exceeds the fixed {INPUT_TOKEN_CAP}-token input cap"
            )
        raw = common.generate(common.GenConfig(
            prompt=prompt, n_samples=1, do_sample=False,
            max_new_tokens=512), seed=args.seed)
        prog = common.extract_code(raw[0])
        parse_ok = _compiles(prog or "")
        hid_ok = common.passes_all(prog or "", prob["hidden_tests"], setup)
        n_parse += int(parse_ok); n_hidden += int(hid_ok)
        common.emit_item(i + 1, hid_ok, parsed=parse_ok)
        if i < 3:
            print(f"CG_SAMPLE i={i} id={prob['task_id']} parse_ok={parse_ok} "
                  f"reserved_ok={hid_ok} input_tokens={input_tokens}", flush=True)
        if (i + 1) % 20 == 0:
            common.emit_progress(i + 1, n_hidden)
    dt = time.time() - t0
    print(f"CG_METRICS task={TASK_ID} pass_at_1={n_hidden/n:.6f} "
          f"parse_rate={n_parse/n:.6f} n={n} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
