"""
This code is partially borrowed from: https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py
"""

import json
import os
import re
from typing import Tuple

import fire


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion: str) -> Tuple[str, str]:
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        completion = completion.replace(f"#### {match_str}", "").strip()
        match_str = match_str.replace(",", "").strip()
        return completion, match_str
    else:
        raise ValueError(f"Cannot extract answer in {completion}.")


def convert_gsm8k_to_math(data_folder: str):
    """
    Convert the GSM8K dataset to the MATH dataset format.

    Args:
        data_folder (str): Path to https://github.com/openai/grade-school-math/tree/master/grade_school_math/data
    """
    os.makedirs(os.path.join("data", "gsm8k_splits"), exist_ok=True)
    for filename in ["train.jsonl", "test.jsonl"]:
        samples = []
        with open(os.path.join(data_folder, filename), encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                solution, answer = extract_answer(sample["answer"])
                if solution[-1] != ".":
                    solution = f"{solution}."

                solution = rf"{solution} The answer is $\boxed{{{answer}}}$."
                samples.append(
                    {
                        "problem": sample["question"],
                        "solution": solution,
                        "answer": answer,
                    }
                )

        with open(os.path.join("data", "gsm8k_splits", filename), "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(convert_gsm8k_to_math)
