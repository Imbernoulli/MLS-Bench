#!/usr/bin/env python3
"""codegen-self-repair harness (fixed pipeline).

Greedy-generate one program per problem and run the PROVIDED tests. After a
failure, the editable `build_repair_prompt(...)` supplies only a prompt. The
verifier owns one fixed greedy generation per round, for at most two rounds.
Scoring uses the RESERVED assertions.
"""
from __future__ import annotations
import argparse, time
import common


TASK_ID = "codegen-self-repair"
MAX_REPAIR_INPUT_TOKENS = 1536


def _instruction(prob):
    return ("Complete the following Python function. Return ONLY the complete "
            "function inside a single ```python code block.\n\n" + prob["prompt"])


def _repair_prompt(policy, tok, *args):
    prompt = common.call_policy_without_generation(policy, *args)
    if not isinstance(prompt, str) or not prompt.strip():
        raise TypeError("build_repair_prompt() must return non-empty text")
    token_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if len(token_ids) > MAX_REPAIR_INPUT_TOKENS:
        raise ValueError(
            f"repair prompt exceeds {MAX_REPAIR_INPUT_TOKENS} input tokens"
        )
    return prompt


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
    build_repair_prompt = common.load_surface(args.solution, "build_repair_prompt")
    common.emit_protocol(TASK_ID, args.seed, problems)
    n = len(problems); n_hidden = 0; n_vis = 0; n_help = 0
    for i, prob in enumerate(problems):
        sp = common.safe_problem(prob); setup = prob.get("test_setup", "")
        raw = common.generate(common.GenConfig(
            prompt=_instruction(prob), n_samples=1, do_sample=False,
            max_new_tokens=512), seed=args.seed)
        prog = common.extract_code(raw[0])
        res = common.run_tests(prog, prob["visible_tests"], setup)
        failed_vis = not res["ok"]
        rounds = 0
        while not res["ok"] and rounds < 2:
            prompt = _repair_prompt(
                build_repair_prompt,
                tok,
                sp,
                prog,
                str(res.get("err", "")),
                rounds,
            )
            raw_repair = common.generate(common.GenConfig(
                prompt=prompt, n_samples=1, do_sample=False,
                max_new_tokens=512), seed=args.seed)
            prog = common.extract_code(raw_repair[0])
            res = common.run_tests(prog, prob["visible_tests"], setup)
            rounds += 1
        vis_ok = res["ok"]
        hid_ok = common.passes_all(prog, prob["hidden_tests"], setup)
        n_vis += int(vis_ok); n_hidden += int(hid_ok)
        helped = failed_vis and vis_ok
        n_help += int(helped)
        common.emit_item(i + 1, hid_ok, visible=vis_ok, helped=helped)
        if i < 3:
            print(f"CG_SAMPLE i={i} id={prob['task_id']} rounds={rounds} "
                  f"vis_ok={vis_ok} hid_ok={hid_ok}", flush=True)
        if (i + 1) % 20 == 0:
            common.emit_progress(i + 1, n_hidden)
    dt = time.time() - t0
    print(f"CG_METRICS task={TASK_ID} pass_at_1={n_hidden/n:.6f} "
          f"visible_solve_rate={n_vis/n:.6f} "
          f"repair_help_rate={n_help/n:.6f} n={n} elapsed={dt:.1f}", flush=True)


if __name__ == "__main__":
    main()
