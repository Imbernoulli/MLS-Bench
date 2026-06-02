import json
import os

from datasets import load_dataset


def process_aime24():
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    with open(os.path.join("data", "aime_splits", "aime_2024.jsonl"), "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def process_aime25():
    dataset = load_dataset("math-ai/aime25", split="test")
    with open(os.path.join("data", "aime_splits", "aime_2025.jsonl"), "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    os.makedirs(os.path.join("data", "aime_splits"), exist_ok=True)
    process_aime24()
    process_aime25()


if __name__ == "__main__":
    main()
