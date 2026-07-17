"""
This code is partially borrowed from: https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/examples.py
"""

import json
import os
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Any, Dict, Generator, List, Literal

import torch
from datasets import Dataset
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams

from .demo import DEMOS


@dataclass
class Processor:
    """
    A class to process the data samples.
    """

    tokenizer: "PreTrainedTokenizer"
    n_shot: int
    demo_split: Literal["math", "gsm8k"]
    system: str

    def __post_init__(self):
        if self.n_shot != 0 and self.demo_split not in ["math", "gsm8k"]:
            raise ValueError("`demo_split` must be either `math` or `gsm8k`.")

        if self.n_shot > len(DEMOS[self.demo_split]):
            raise ValueError(f"`n_shot` must be less than or equal to {len(DEMOS[self.demo_split])}.")

    def encode_sample(self, problem: str) -> List[int]:
        """
        Tokenizes sample and returns its token ids.
        """
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})

        for i in range(self.n_shot):
            problem = f"{DEMOS[self.demo_split][i][0]}\n{DEMOS[self.demo_split][i][1]}\n\n{problem}"

        messages.append({"role": "user", "content": problem})
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def process_file(self, input_file: "TextIOWrapper") -> Generator[Dict[str, Any], None, None]:
        """
        Reads the samples from file.
        """
        for line in input_file:
            sample = json.loads(line)
            input_ids = self.encode_sample(sample["problem"])
            yield {"sample": sample, "model_inputs": {"prompt_token_ids": input_ids}}

    def collect_results(self, results: List["RequestOutput"]) -> Generator[Dict[str, List[List[int]]], None, None]:
        """
        Collects predicted token ids from vllm outputs.
        """
        for result in results:
            predicts = [output.token_ids for output in result.outputs]
            yield {"predicts": predicts}

    def decode_sample(self, outputs: Dict[str, List[List[int]]]) -> Dict[str, List[str]]:
        """
        Detokenize the token ids of predictions.
        """
        predicts = []
        for predict in outputs["predicts"]:
            predicts.append(self.tokenizer.decode(predict, skip_special_tokens=True))

        return {"predicts": predicts}


def vllm_generate(
    model: str,
    json_path: str = "data/math_splits/test.jsonl",
    save_path: str = "predicts/test.jsonl",
    n_shot: int = 0,
    demo_split: str = "math",
    system: str = r"Please reason step by step, and put your final answer within \boxed{}.",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    sample_num: int = 1,
):
    tokenizer = AutoTokenizer.from_pretrained(model)
    processor = Processor(tokenizer, n_shot, demo_split, system)
    with open(json_path, encoding="utf-8") as f:
        inputs = Dataset.from_generator(processor.process_file, gen_kwargs={"input_file": f})

    print("Input example:")
    print("=" * 50)
    print(tokenizer.decode(inputs[0]["model_inputs"]["prompt_token_ids"]))
    print("=" * 50)

    generation_conig = GenerationConfig.from_pretrained(model)
    if isinstance(generation_conig.eos_token_id, list):
        stop_token_ids = generation_conig.eos_token_id
    else:
        stop_token_ids = [generation_conig.eos_token_id]

    vllm_engine = LLM(model=model, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(
        n=sample_num,
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=stop_token_ids,
        max_tokens=max_tokens,
        detokenize=False,
    )
    results = vllm_engine.generate(inputs["model_inputs"], sampling_params)
    outputs = Dataset.from_generator(processor.collect_results, gen_kwargs={"results": results})
    outputs = outputs.map(processor.decode_sample, batched=False, num_proc=16, desc="Decode outputs")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for sample, predicts in zip(inputs["sample"], outputs["predicts"]):
            sample["predicts"] = predicts
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("=" * 50)
    print(f"Generated results have been saved at `{save_path}`.")
    print(f"Use `mathruler eval {save_path}` to evaluate them.")
    print("=" * 50)
