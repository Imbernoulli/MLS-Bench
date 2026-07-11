#!/usr/bin/env python3
"""Prompt/postprocess interaction evaluation under three output conditions."""
from __future__ import annotations

import argparse
import time

import common


TASK_ID = "codegen-prompt-postprocess"
CONDITIONS = ("direct", "fenced_wrapper", "trailing_text")
INPUT_TOKEN_CAP = 1024


def _prompt_view(prob):
    """Return the complete and exact prompt-policy view."""
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


def _condition_views(raw_text):
    """Construct deterministic views without exposing which view is being scored."""
    if not isinstance(raw_text, str):
        raise TypeError("generation must return text")
    return {
        "direct": raw_text,
        "fenced_wrapper": (
            "A generated answer follows.\n"
            "```python\n"
            f"{raw_text}\n"
            "```\n"
            "End of generated answer."
        ),
        "trailing_text": (
            f"{raw_text}\n\n"
            "The code above is the generated answer.\n"
            "Example usage is intentionally omitted."
        ),
    }


def _compiles(src):
    try:
        compile(src, "<prog>", "exec")
        return bool(src.strip())
    except Exception:
        return False


def _emit_item(index, outcomes):
    fields = [f"CG_ITEM i={index}"]
    for condition in CONDITIONS:
        passed, parsed = outcomes[condition]
        fields.extend(
            (f"{condition}_pass={int(passed)}", f"{condition}_parse={int(parsed)}")
        )
    print(" ".join(fields), flush=True)


def _emit_progress(completed, passed):
    fields = [f"CG_PROGRESS completed={completed}", f"total={common.EXPECTED_PROBLEMS}"]
    fields.extend(f"{condition}_passed={passed[condition]}" for condition in CONDITIONS)
    print(" ".join(fields), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=common.EXPECTED_PROBLEMS)
    args = ap.parse_args()

    common.set_seeds(args.seed)
    t0 = time.time()
    common.start_executor()
    tok, _ = common.load_model()
    problems = common.load_problems(args.n)
    common.seal_private_data()
    build_prompt = common.load_surface(args.solution, "build_prompt")
    postprocess = common.load_surface(args.solution, "postprocess")
    common.emit_protocol(TASK_ID, args.seed, problems)

    passed = {condition: 0 for condition in CONDITIONS}
    parsed = {condition: 0 for condition in CONDITIONS}
    for i, prob in enumerate(problems):
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
        raw = common.generate(
            common.GenConfig(
                prompt=prompt,
                n_samples=1,
                do_sample=False,
                max_new_tokens=512,
            ),
            seed=args.seed,
        )
        views = _condition_views(raw[0])
        outcomes = {}
        for condition in CONDITIONS:
            program = common.call_policy_without_generation(
                postprocess, views[condition], prob["entry_point"]
            )
            if not isinstance(program, str):
                raise TypeError("postprocess() must return source text")
            parse_ok = _compiles(program)
            reserved_ok = common.passes_all(
                program, prob["hidden_tests"], prob.get("test_setup", "")
            )
            parsed[condition] += int(parse_ok)
            passed[condition] += int(reserved_ok)
            outcomes[condition] = (reserved_ok, parse_ok)
        _emit_item(i + 1, outcomes)
        if i < 3:
            summary = " ".join(
                f"{condition}_parse={int(outcomes[condition][1])} "
                f"{condition}_pass={int(outcomes[condition][0])}"
                for condition in CONDITIONS
            )
            print(
                f"CG_SAMPLE i={i} id={prob['task_id']} input_tokens={input_tokens} {summary}",
                flush=True,
            )
        if (i + 1) % 20 == 0:
            _emit_progress(i + 1, passed)

    n = len(problems)
    fields = ["CG_METRICS", f"task={TASK_ID}"]
    for condition in CONDITIONS:
        fields.extend(
            (
                f"pass_at_1_{condition}={passed[condition] / n:.6f}",
                f"parse_rate_{condition}={parsed[condition] / n:.6f}",
            )
        )
    fields.extend((f"n={n}", f"elapsed={time.time() - t0:.1f}"))
    print(" ".join(fields), flush=True)


if __name__ == "__main__":
    main()
