"""Full-attention KV selection replay harness.

The scaffold owns the model, data, decode loop, and fixed cache budget. Policies
only describe how to score already-computed KV tokens after a standard
full-attention prefill. The selected tokens are then used for greedy decoding on
the same public text workloads as llm-kv-adaptive-quantization.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import math
import os
import re
import string
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, set_seed


@dataclass
class WorkloadSpec:
    name: str
    source_family: str
    examples: list[dict]
    max_new_tokens: int


class SelectionPolicy:
    """Editable semantic hook for KV token retention after full-attention prefill."""

    method_name = "streamingllm"
    rerotate_selected_keys = True

    def retention_plan(self, layer_id, request_meta, cache_meta):
        return {
            "method": self.method_name,
            "sink_tokens": 4,
            "compression_ratio": cache_meta["compression_ratio"],
        }

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        k_len = int(keys.shape[2])
        n_sink = int(plan.get("sink_tokens", 4))
        ratio = float(plan["compression_ratio"])
        assert k_len > n_sink, f"Input should contain more tokens than sink_tokens={n_sink}"
        n_pruned = k_len - int(k_len * (1.0 - ratio))
        scores = torch.ones_like(keys[..., 0])
        scores[:, :, n_sink : n_sink + n_pruned] = 0
        return scores

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def rerotate_cache_keys(self, module, indices, keys):
        bsz, num_key_value_heads, n_kept = indices.shape
        device = indices.device
        device_type = keys.device.type
        dtype = keys.dtype
        inv_freq = module.rotary_emb.inv_freq[None, None, :, None].float().expand(
            bsz, num_key_value_heads, -1, 1
        )
        new_positions = torch.arange(0, n_kept, device=device).unsqueeze(0)[:, None, :].float()
        new_positions = new_positions.expand(bsz, num_key_value_heads, n_kept)
        delta_pos = (new_positions - indices.float()).unsqueeze(2)
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (delta_pos.float() * inv_freq.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().contiguous()
            sin = emb.sin().contiguous()
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        gathered = keys.gather(2, gather_idx).contiguous()
        return (gathered * cos) + (self.rotate_half(gathered) * sin)

    def select_cache(self, module, keys, values, scores, n_kept):
        indices = scores.topk(n_kept, dim=-1).indices
        if self.rerotate_selected_keys:
            indices = torch.sort(indices, dim=2).values
            selected_keys = self.rerotate_cache_keys(module, indices, keys)
        else:
            gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
            selected_keys = keys.gather(2, gather_idx).contiguous()
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
        selected_values = values.gather(2, gather_idx).contiguous()
        return selected_keys, selected_values


def resolve_task_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here,
        here.parent,
        Path(os.environ.get("MLSBENCH_TASK_DIR", "")),
        Path(os.environ.get("TASK_DIR", "")),
        Path.cwd() / "_task",
        Path.cwd().parent / "_task",
    ]
    seen = set()
    for candidate in candidates:
        if not candidate or str(candidate) == ".":
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "task_description.md").exists():
            return resolved
    raise FileNotFoundError(f"Unable to locate task directory from {here}")


TASK_DIR = resolve_task_dir()
DEFAULT_MAX_EXAMPLES = int(os.environ.get("SELECTION_KV_MAX_EXAMPLES", "0"))
DEFAULT_MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
DEFAULT_COMPRESSION_RATIO = float(os.environ.get("SELECTION_KV_COMPRESSION_RATIO", "0.8"))
DEFAULT_MAX_PROMPT_TOKENS = int(os.environ.get("SELECTION_KV_MAX_PROMPT_TOKENS", "0"))

LONG_BENCH_TEMPLATES = {
    "hotpotqa_e": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Question: {input}\nAnswer:"
    ),
    "passage_retrieval_en_e": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
        "The answer is: "
    ),
    "repobench-p_e": "Please complete the code given below.\n{context}{input}Next line of code:\n",
}

LONG_BENCH_V2_TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "{context}\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n"
    "Format your response as follows: \"The correct answer is (insert answer here)\"."
)

WORKLOAD_CONFIGS = {
    "longbench_hotpotqa": {
        "source_family": "THUDM/LongBench hotpotqa_e",
        "dataset_name": "hotpotqa_e",
        "max_new_tokens": 32,
    },
    "longbench_passage_retrieval": {
        "source_family": "THUDM/LongBench passage_retrieval_en_e",
        "dataset_name": "passage_retrieval_en_e",
        "max_new_tokens": 32,
    },
    "longbench_repobench": {
        "source_family": "THUDM/LongBench repobench-p_e",
        "dataset_name": "repobench-p_e",
        "max_new_tokens": 64,
    },
    "longbench_v2": {
        "source_family": "THUDM/LongBench-v2 train split",
        "max_new_tokens": 128,
    },
    "gsm8k": {
        "source_family": "openai/gsm8k main test split",
        "max_new_tokens": 256,
    },
}
WORKLOADS = WORKLOAD_CONFIGS


def load_hf_dataset(repo: str, config: str | None = None, split: str = "test"):
    from datasets import DownloadConfig, load_dataset

    download_config = DownloadConfig(local_files_only=True)
    if config is None:
        dataset = load_dataset(repo, split=split, download_config=download_config)
    else:
        dataset = load_dataset(repo, config, split=split, download_config=download_config)
    if hasattr(dataset, "keys"):
        if split not in dataset:
            raise RuntimeError(f"Cached dataset {repo}/{config or ''} does not contain required split {split!r}")
        return dataset[split]
    return dataset


def load_cached_gsm8k_test():
    from datasets import Dataset
    from datasets import config as datasets_config

    cache_roots = []
    if os.environ.get("HF_DATASETS_CACHE"):
        cache_roots.append(Path(os.environ["HF_DATASETS_CACHE"]))
    if os.environ.get("HF_HOME"):
        cache_roots.append(Path(os.environ["HF_HOME"]) / "datasets")
        cache_roots.append(Path(os.environ["HF_HOME"]))
    if os.environ.get("HF_HUB_CACHE"):
        cache_roots.append(Path(os.environ["HF_HUB_CACHE"]).parent / "datasets")
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        cache_roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]).parent / "datasets")
    if os.environ.get("MODEL_ID"):
        model_path = Path(os.environ["MODEL_ID"]).expanduser()
        if model_path.exists():
            for parent in model_path.resolve().parents:
                cache_roots.append(parent / "datasets")
                if parent.name == "hub":
                    cache_roots.append(parent.parent / "datasets")
    if getattr(datasets_config, "HF_DATASETS_CACHE", None):
        cache_roots.append(Path(datasets_config.HF_DATASETS_CACHE))
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "datasets")

    seen = set()
    for root in cache_roots:
        root = root.expanduser()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for pattern in (
            "openai___gsm8k/main/**/gsm8k-test.arrow",
            "**/openai___gsm8k/main/**/gsm8k-test.arrow",
            "**/gsm8k-test.arrow",
        ):
            matches = sorted(root.glob(pattern))
            if matches:
                return Dataset.from_file(str(matches[0]))
    raise FileNotFoundError("Unable to locate cached openai/gsm8k main test split")


def hf_dataset_file(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_files_only=True))


def limit_reached(examples: list[dict], max_examples: int) -> bool:
    return max_examples > 0 and len(examples) >= max_examples


def read_jsonl_lines(lines, max_examples: int = 0) -> list[dict]:
    rows = []
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
        if limit_reached(rows, max_examples):
            break
    return rows


def load_longbench_rows(dataset_name: str, max_examples: int) -> list[dict]:
    archive_path = hf_dataset_file("THUDM/LongBench", "data.zip")
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            name for name in archive.namelist()
            if name.endswith(f"{dataset_name}.jsonl") or name.endswith(f"{dataset_name}.json")
        ]
        if not candidates:
            raise FileNotFoundError(f"Unable to find {dataset_name} inside THUDM/LongBench data.zip")
        with archive.open(candidates[0]) as fh:
            return read_jsonl_lines(fh, max_examples)


def build_longbench_examples(workload_name: str, max_examples: int) -> list[dict]:
    dataset_name = WORKLOAD_CONFIGS[workload_name]["dataset_name"]
    template = LONG_BENCH_TEMPLATES[dataset_name]
    examples = []
    for raw in load_longbench_rows(dataset_name, max_examples):
        answers = raw.get("answers") or raw.get("outputs") or raw.get("answer") or []
        if isinstance(answers, str):
            answers = [answers]
        examples.append(
            {
                "example_id": raw.get("_id", f"{dataset_name}-{len(examples)}"),
                "dataset": dataset_name,
                "prompt": template.format(context=raw["context"], input=raw["input"]),
                "answers": [str(answer) for answer in answers],
            }
        )
        if limit_reached(examples, max_examples):
            return examples
    return examples


def load_longbench_v2_rows(max_examples: int) -> list[dict]:
    errors = []
    for repo in ("THUDM/LongBench-v2", "zai-org/LongBench-v2"):
        try:
            dataset = load_hf_dataset(repo, split="train")
        except Exception as exc:
            errors.append(exc)
            continue
        rows = []
        for raw in dataset:
            rows.append(dict(raw))
            if limit_reached(rows, max_examples):
                break
        return rows
    raise RuntimeError("Unable to load LongBench v2 from Hugging Face.") from errors[-1]


def build_longbench_v2_examples(max_examples: int) -> list[dict]:
    examples = []
    for raw in load_longbench_v2_rows(max_examples):
        examples.append(
            {
                "example_id": raw.get("_id", f"longbench-v2-{len(examples)}"),
                "dataset": "longbench_v2",
                "difficulty": raw.get("difficulty", ""),
                "length": raw.get("length", ""),
                "domain": raw.get("domain", ""),
                "prompt": LONG_BENCH_V2_TEMPLATE.format(
                    context=str(raw["context"]).strip(),
                    question=str(raw["question"]).strip(),
                    choice_A=str(raw["choice_A"]).strip(),
                    choice_B=str(raw["choice_B"]).strip(),
                    choice_C=str(raw["choice_C"]).strip(),
                    choice_D=str(raw["choice_D"]).strip(),
                ),
                "answers": [str(raw["answer"]).strip().upper()],
            }
        )
        if limit_reached(examples, max_examples):
            return examples
    return examples


def extract_boxed_answer(text: str) -> str:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return ""
    i = start + len(marker)
    depth = 1
    chars = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(ch)
        i += 1
    return ""


def normalize_math_answer(text: str) -> str:
    answer = extract_boxed_answer(text) or text
    answer = answer.strip().replace("$", "")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
    answer = re.sub(r"\s+", "", answer)
    answer = answer.rstrip(".;,").replace(",", "")
    return answer.lower()


def extract_math_answer(text: str) -> str:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    numeric_matches = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text.replace(",", ""))
    if numeric_matches:
        return numeric_matches[-1]
    return text.strip()


def build_gsm8k_examples(max_examples: int) -> list[dict]:
    try:
        dataset = load_hf_dataset("openai/gsm8k", "main", split="test")
    except Exception:
        dataset = load_cached_gsm8k_test()
    examples = []
    for raw in dataset:
        answer = extract_math_answer(raw.get("answer", ""))
        prompt = (
            "Solve the following grade-school math word problem carefully. "
            "Show your reasoning and end with the final answer wrapped in \\\\boxed{}.\n\n"
            f"Problem: {raw['question']}\n\nSolution:"
        )
        examples.append(
            {
                "example_id": f"gsm8k-{len(examples)}",
                "dataset": "gsm8k",
                "prompt": prompt,
                "answers": [answer],
            }
        )
        if limit_reached(examples, max_examples):
            return examples
    return examples


def load_workload(name: str, max_examples: int = DEFAULT_MAX_EXAMPLES) -> WorkloadSpec:
    if name.startswith("longbench_"):
        if name == "longbench_v2":
            examples = build_longbench_v2_examples(max_examples)
        else:
            examples = build_longbench_examples(name, max_examples)
    elif name == "gsm8k":
        examples = build_gsm8k_examples(max_examples)
    else:
        raise ValueError(f"Unsupported workload: {name}")
    return WorkloadSpec(
        name=name,
        source_family=WORKLOAD_CONFIGS[name]["source_family"],
        examples=examples,
        max_new_tokens=WORKLOAD_CONFIGS[name]["max_new_tokens"],
    )


def normalize_text(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    value = text.lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = remove_articles(value)
    return " ".join(value.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def retrieval_score(prediction: str, ground_truth: str) -> float:
    match = re.search(r"Paragraph (\d+)", ground_truth)
    if not match:
        return 0.0
    gold_id = match.group(1)
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(1.0 for number in numbers if number == gold_id) / len(numbers)


def code_similarity_score(prediction: str, ground_truth: str) -> float:
    candidate = ""
    for line in prediction.lstrip("\n").splitlines():
        if "`" not in line and "#" not in line and "//" not in line:
            candidate = line
            break
    return int(round(100 * difflib.SequenceMatcher(None, candidate, ground_truth).ratio())) / 100.0


def extract_choice_answer(text: str) -> str:
    response = text.replace("*", "")
    patterns = (
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
        r"answer is \(([A-D])\)",
        r"answer is ([A-D])",
        r"\(([A-D])\)",
        r"\b([A-D])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def score_prediction(workload_name: str, example: dict, prediction: str) -> float:
    answers = [str(answer) for answer in example.get("answers", []) if str(answer)]
    if not prediction or not answers:
        return 0.0
    dataset = example.get("dataset", "")
    if dataset == "hotpotqa_e":
        return 100.0 * max(token_f1(prediction, answer) for answer in answers)
    if dataset == "passage_retrieval_en_e":
        return 100.0 * max(retrieval_score(prediction, answer) for answer in answers)
    if dataset == "repobench-p_e":
        return 100.0 * max(code_similarity_score(prediction, answer) for answer in answers)
    if workload_name == "longbench_v2":
        prediction_answer = extract_choice_answer(prediction)
        gold_answers = {answer.strip().upper() for answer in answers}
        return 100.0 if prediction_answer and prediction_answer in gold_answers else 0.0
    if workload_name == "gsm8k":
        prediction_answer = normalize_math_answer(extract_math_answer(prediction))
        gold_answers = {normalize_math_answer(answer) for answer in answers}
        return 100.0 if prediction_answer and prediction_answer in gold_answers else 0.0
    raise ValueError(f"Unsupported workload/example dataset: workload={workload_name}, dataset={dataset}")


class PrefillSelectionCompressor:
    def __init__(self, policy: SelectionPolicy, compression_ratio: float, request_meta: dict):
        if not 0.0 <= compression_ratio < 1.0:
            raise ValueError("compression_ratio must be in [0, 1)")
        self.policy = policy
        self.compression_ratio = float(compression_ratio)
        self.request_meta = request_meta
        self.layer_retained_fractions: list[float] = []
        self.rerotate_decode_positions = False
        self.method_names: list[str] = []

    def _cache_layer(self, cache, layer_idx: int):
        if not hasattr(cache, "layers"):
            raise RuntimeError("Selection task expects the Transformers DynamicCache layer API.")
        return cache.layers[layer_idx]

    def _extract_kv(self, cache_layer):
        keys = getattr(cache_layer, "keys", None)
        values = getattr(cache_layer, "values", None)
        if keys is None or values is None:
            raise RuntimeError("Unable to access keys/values from DynamicCache layer.")
        return keys, values

    def forward_hook(self, module: nn.Module, inputs, kwargs: dict, output):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values") or kwargs.get("past_key_value")
        if cache is None:
            return output
        cache_layer = self._cache_layer(cache, int(module.layer_idx))
        keys, values = self._extract_kv(cache_layer)
        if keys.shape[2] <= 1:
            return output
        cache_meta = {
            "sequence_length": int(keys.shape[2]),
            "num_kv_heads": int(keys.shape[1]),
            "head_dim": int(keys.shape[-1]),
            "compression_ratio": self.compression_ratio,
        }
        raw_plan = self.policy.retention_plan(int(module.layer_idx), dict(self.request_meta), dict(cache_meta))
        if not isinstance(raw_plan, dict):
            raise TypeError("retention_plan must return a dict")
        plan = dict(raw_plan)
        method_name = str(plan.get("method", getattr(self.policy, "method_name", "selection_policy"))).lower()
        self.method_names.append(method_name)
        if bool(plan.get("disable_compression", False)) or self.compression_ratio == 0.0:
            self.layer_retained_fractions.append(1.0)
            return output
        plan["compression_ratio"] = self.compression_ratio
        scores = self.policy.score_tokens(module, hidden_states, keys, values, kwargs, plan)
        if scores is None:
            self.layer_retained_fractions.append(1.0)
            return output
        if scores.shape[-1] != keys.shape[2]:
            raise RuntimeError(f"Score length {scores.shape[-1]} does not match cache length {keys.shape[2]}")
        n_kept = int(keys.shape[2] * (1.0 - self.compression_ratio))
        selected_keys, selected_values = self.policy.select_cache(module, keys, values, scores, n_kept)
        self.rerotate_decode_positions = (
            self.rerotate_decode_positions or bool(getattr(self.policy, "rerotate_selected_keys", False))
        )
        cache_layer.keys = selected_keys
        cache_layer.values = selected_values
        if hasattr(cache_layer, "cumulative_length"):
            cache_layer.cumulative_length = int(selected_keys.shape[2])
        self.layer_retained_fractions.append(float(n_kept) / float(keys.shape[2]))
        return output

    @contextlib.contextmanager
    def apply(self, model):
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        hooks = []
        try:
            for layer in language_model.layers:
                layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield self
        finally:
            for hook in hooks:
                hook.remove()

    def method_name(self) -> str:
        return self.method_names[0] if self.method_names else "unknown"


def decode_prediction(tokenizer, generated_tokens: torch.Tensor, workload_name: str) -> str:
    text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
    if workload_name == "gsm8k":
        return text
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def infer_prompt_token_limit(model, tokenizer, max_new_tokens: int) -> int:
    if DEFAULT_MAX_PROMPT_TOKENS > 0:
        return DEFAULT_MAX_PROMPT_TOKENS
    candidates = []
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and 0 < value < 10_000_000:
            candidates.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10_000_000:
        candidates.append(tokenizer_limit)
    if not candidates:
        return 0
    return max(min(candidates) - int(max_new_tokens) - 1, 1)


def tokenize_prompt_for_model(model, tokenizer, prompt: str, max_new_tokens: int, device: str):
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    original_len = int(input_ids.shape[1])
    limit = infer_prompt_token_limit(model, tokenizer, max_new_tokens)
    truncated = False
    if limit > 0 and original_len > limit:
        head = limit // 2
        tail = limit - head
        input_ids = torch.cat((input_ids[:, :head], input_ids[:, -tail:]), dim=1)
        encoded = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        truncated = True
    else:
        encoded = dict(encoded)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return encoded, original_len, truncated


def maybe_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def generate_with_selection(model, tokenizer, example, workload_name, max_new_tokens, policy, compression_ratio, device):
    inputs, original_prompt_len, prompt_truncated = tokenize_prompt_for_model(
        model, tokenizer, example["prompt"], max_new_tokens, device
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    cache = DynamicCache(config=model.config)
    request_meta = {
        "workload": workload_name,
        "dataset": example.get("dataset", ""),
        "example_id": example.get("example_id", ""),
        "prompt_tokens": prompt_len,
        "prompt_tokens_original": original_prompt_len,
        "prompt_truncated": prompt_truncated,
        "prompt_chars": len(example["prompt"]),
    }
    compressor = PrefillSelectionCompressor(policy, compression_ratio, request_meta)
    maybe_sync()
    with compressor.apply(model):
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=True, past_key_values=cache, logits_to_keep=1)
    maybe_sync()
    current_cache = outputs.past_key_values
    decode_start_position = (
        int(current_cache.get_seq_length()) if compressor.rerotate_decode_positions else prompt_len
    )
    step_input = outputs.logits[:, -1:].argmax(dim=-1).detach()
    generated = [step_input.detach().clone()]
    del outputs

    for offset in range(max_new_tokens - 1):
        if tokenizer.eos_token_id is not None and int(step_input.item()) == int(tokenizer.eos_token_id):
            break
        cache_position = torch.tensor([decode_start_position + offset], dtype=torch.long, device=device)
        maybe_sync()
        with torch.inference_mode():
            outputs = model(
                input_ids=step_input,
                use_cache=True,
                past_key_values=current_cache,
                cache_position=cache_position,
                logits_to_keep=1,
            )
        maybe_sync()
        step_input = outputs.logits[:, -1:].argmax(dim=-1).detach()
        current_cache = outputs.past_key_values
        generated.append(step_input.detach().clone())
        del outputs

    generated_tokens = torch.cat(generated, dim=-1)
    trace = {
        "prompt_tokens": prompt_len,
        "prompt_tokens_original": original_prompt_len,
        "prompt_truncated": prompt_truncated,
        "generated_tokens": int(generated_tokens.shape[-1]),
        "retained_fraction": mean(compressor.layer_retained_fractions) if compressor.layer_retained_fractions else 1.0,
        "method": compressor.method_name(),
    }
    return decode_prediction(tokenizer, generated_tokens, workload_name), trace


def maybe_write_output_artifacts(workload: str, trace: dict, metrics: dict) -> None:
    output_dir = os.environ.get("OUTPUT_DIR")
    if not output_dir:
        return
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = workload.replace("-", "_")
    (out_path / f"{stem}_trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    (out_path / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def safe_mean(values) -> float:
    return mean(values) if values else 0.0


def run_real_eval(args):
    eval_start = time.perf_counter()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    workload = load_workload(args.workload, args.max_examples)
    policy = SelectionPolicy()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    ).to(device)
    model.eval()

    final_scores = []
    latencies = []
    retained_fractions = []
    method_names = []
    for example in workload.examples:
        start = time.perf_counter()
        prediction, trace = generate_with_selection(
            model,
            tokenizer,
            example,
            args.workload,
            workload.max_new_tokens,
            policy,
            args.compression_ratio,
            device,
        )
        latencies.append(time.perf_counter() - start)
        retained_fractions.append(float(trace["retained_fraction"]))
        method_names.append(str(trace.get("method", "unknown")))
        final_scores.append(score_prediction(args.workload, example, prediction))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed_s = time.perf_counter() - eval_start
    metrics = {
        "final_score": safe_mean(final_scores),
        "mean_latency_s": safe_mean(latencies),
        "mean_retained_fraction": safe_mean(retained_fractions),
        "runtime_seconds": elapsed_s,
    }
    trace = {
        "workload": args.workload,
        "examples": len(workload.examples),
        "model": args.model_id,
        "method": method_names[0] if method_names else "unknown",
        "compression_ratio": args.compression_ratio,
        "source_family": workload.source_family,
        **metrics,
    }
    maybe_write_output_artifacts(args.workload, trace, metrics)
    print(
        "TRACE_METRICS: "
        f"task=llm-kv-selection-budgeting workload={args.workload} "
        f"examples={len(workload.examples)} model={args.model_id} method={trace['method']} "
        f"compression_ratio={args.compression_ratio:.4f}"
    )
    print(
        "TEST_METRICS: "
        f"final_score={metrics['final_score']:.6f} "
        f"mean_retained_fraction={metrics['mean_retained_fraction']:.6f} "
        f"runtime_seconds={metrics['runtime_seconds']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--compression-ratio", type=float, default=DEFAULT_COMPRESSION_RATIO)
    parser.add_argument("--attn-implementation", default=os.environ.get("SELECTION_KV_ATTN_IMPL", "sdpa"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    args = parser.parse_args()

    run_real_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
