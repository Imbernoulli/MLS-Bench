#!/usr/bin/env python3
"""Single-pass vLLM math evaluation on 3 benchmarks.

Loads the trained model once and runs greedy generation for GSM8K, MATH-500,
and AIME-2024 (MathRuler's bundled splits). Grading uses MathRuler's sympy
+ mathd grader — judge-free.

Prints lines that the task parser picks up:
    EVAL_RESULT benchmark=<name> correct=<int> total=<int> accuracy=<pct>
"""

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

from mathruler.grader import extract_boxed_content, grade_answer


# (display_name, splits_subdir, jsonl_filename)
BENCHMARKS = [
    ("gsm8k",     "gsm8k_splits", "test.jsonl"),
    ("math500",   "math_splits",  "test.jsonl"),
    ("aime2024",  "aime_splits",  "aime_2024.jsonl"),
]

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def load_problems(jsonl_path: Path):
    problems = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
    return problems


def build_prompts(problems, tokenizer):
    prompts = []
    for sample in problems:
        question = sample.get("problem") or sample.get("question") or ""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
    return prompts


def grade_predictions(problems, completions):
    correct = 0
    for sample, completion in zip(problems, completions):
        predicted = extract_boxed_content(completion)
        if grade_answer(predicted, sample["answer"]):
            correct += 1
    return correct


def main():
    model_path = os.environ.get("OUTPUT_DIR")
    if not model_path or not Path(model_path).is_dir():
        print(f"ERROR: model directory not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Splits live in MathRuler's package dir under data/{aime,gsm8k,math}_splits/
    # The wrapping shell does `cd /workspace/MathRuler` so cwd-relative works.
    splits_root = Path("data").resolve()
    if not splits_root.is_dir():
        splits_root = Path("/workspace/MathRuler/data")

    print(f"Evaluating model: {model_path}")
    print(f"Splits root: {splits_root.resolve()}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    gen_cfg = GenerationConfig.from_pretrained(model_path)
    eos = gen_cfg.eos_token_id
    stop_token_ids = list(eos) if isinstance(eos, list) else [eos]

    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048,
        stop_token_ids=stop_token_ids,
    )

    engine = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
    )

    summary = {}
    for name, subdir, fname in BENCHMARKS:
        jsonl_path = splits_root / subdir / fname
        if not jsonl_path.is_file():
            print(f"  {name}: split file missing ({jsonl_path}), skipping",
                  file=sys.stderr)
            continue

        problems = load_problems(jsonl_path)
        prompts = build_prompts(problems, tokenizer)
        outputs = engine.generate(prompts, sampling)
        completions = [o.outputs[0].text for o in outputs]

        correct = grade_predictions(problems, completions)
        total = len(problems)
        acc = 100.0 * correct / max(total, 1)
        summary[name] = (correct, total, acc)
        print(f"EVAL_RESULT benchmark={name} correct={correct} total={total} "
              f"accuracy={acc:.2f}")

    print("\n=== Summary ===")
    for name, (correct, total, acc) in summary.items():
        print(f"  {name}: {correct}/{total} = {acc:.2f}%")


if __name__ == "__main__":
    main()
