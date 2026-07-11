#!/usr/bin/env python3
"""Output-extraction evaluation with a fixed prompt and greedy decoder."""
from __future__ import annotations
import argparse, time
import common


TASK_ID = "codegen-output-extract"


def _prompt(prob):
    return ("Complete the following Python function. Return ONLY the complete "
            "function definition inside a single ```python code block, with no "
            "example usage and no explanation.\n\n" + prob["prompt"])


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
    extract = common.load_surface(args.solution, "extract")
    common.emit_protocol(TASK_ID, args.seed, problems)
    n = len(problems); n_hidden = 0; n_parse = 0
    for i, prob in enumerate(problems):
        setup = prob.get("test_setup", "")
        raw = common.generate(common.GenConfig(
            prompt=_prompt(prob), n_samples=1, do_sample=False,
            max_new_tokens=512), seed=args.seed)
        raw_text = raw[0]
        prog = common.call_policy_without_generation(
            extract, raw_text, prob["entry_point"]
        )
        if not isinstance(prog, str):
            raise TypeError("extract() must return source text")
        parse_ok = _compiles(prog or "")
        hid_ok = common.passes_all(prog or "", prob["hidden_tests"], setup)
        n_parse += int(parse_ok); n_hidden += int(hid_ok)
        common.emit_item(i + 1, hid_ok, parsed=parse_ok)
        if i < 3:
            print(f"CG_SAMPLE i={i} id={prob['task_id']} parse_ok={parse_ok} "
                  f"hid_ok={hid_ok}", flush=True)
        if (i + 1) % 20 == 0:
            common.emit_progress(i + 1, n_hidden)
    dt = time.time() - t0
    print(f"CG_METRICS task={TASK_ID} pass_at_1={n_hidden/n:.6f} "
          f"parse_rate={n_parse/n:.6f} n={n} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
