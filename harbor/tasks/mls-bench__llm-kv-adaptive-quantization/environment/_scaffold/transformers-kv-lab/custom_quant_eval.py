"""Tensor-level KV-cache quantization replay harness.

This scaffold replays deterministic decode steps on top of Hugging Face
Transformers. Instead of collapsing the policy into one global
QuantizedCacheConfig, it snapshots real KV tensors, quantizes them with
source-backed overlap rules, and replays the next decode step with the
quantized cache.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import string
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, set_seed


@dataclass
class WorkloadSpec:
    name: str
    source_family: str
    examples: list[dict]
    max_new_tokens: int

    @property
    def prompts(self) -> list[str]:
        return [example["prompt"] for example in self.examples]


class AdaptiveKVQuantizer:
    """Editable KV-cache quantizer.

    The fixed harness supplies real key/value tensors from a Hugging Face
    DynamicCache and calls this class for the actual algorithm. Participants
    may rewrite the quantization math, residual policy, optional prefill
    observation, and memory accounting here without changing the benchmark
    datasets, model, or decode loop.
    """

    def __init__(self):
        self.bits = 4
        self.key_group_size = 32
        self.value_group_size = 32
        self.key_residual_length = 128
        self.value_residual_length = 128

    def reset_request(self, request_meta: dict, budget_state: dict):
        self.bits = min(4, int(budget_state.get("budget_bits", 4)))
        workload = str(request_meta.get("workload", ""))
        residual = 128 if workload.startswith("longbench_") else 32
        self.key_residual_length = residual
        self.value_residual_length = residual

    def needs_prefill_qkv_observer(self) -> bool:
        return False

    def observe_prefill_qkv(
        self,
        layer_id: int,
        query_states: torch.Tensor | None,
        key_states: torch.Tensor | None,
        value_states: torch.Tensor | None,
        attention_meta: dict,
    ) -> None:
        return None

    def query_observation_position(self) -> str:
        return "post_rope"

    def _residual_keep_length(self, seq_len: int, residual_length: int, residual_policy: str = "tail") -> int:
        residual_length = max(0, min(seq_len, int(residual_length)))
        if residual_length == 0 or residual_policy in {"none", ""}:
            return 0
        if residual_policy == "block_modulo":
            return seq_len % residual_length
        if residual_policy == "tail":
            return residual_length
        raise ValueError(f"Unsupported residual_policy={residual_policy}")

    def _minmax_quantize_last_dim(self, data: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
        if data.numel() == 0 or bits >= FP_BITS - 0.5:
            return data
        max_int = max(1, int(2**int(bits)) - 1)
        trailing = data.shape[-1]
        group_size = trailing if int(group_size) <= 0 else int(group_size)
        padded = math.ceil(trailing / group_size) * group_size
        work = data
        if padded != trailing:
            work = torch.nn.functional.pad(work, (0, padded - trailing))
        grouped = work.reshape(*work.shape[:-1], padded // group_size, group_size)
        gmin = grouped.amin(dim=-1, keepdim=True)
        gmax = grouped.amax(dim=-1, keepdim=True)
        scale = (gmax - gmin).clamp(min=1e-5) / max_int
        quant = torch.round((grouped - gmin) / scale).clamp(0, max_int)
        dequant = quant.mul(scale).add(gmin)
        return dequant.reshape(*work.shape[:-1], padded)[..., :trailing]

    def _quantize_grouped_minmax(
        self,
        layer_tensor: torch.Tensor,
        *,
        axis: str,
        bits: int,
        group_size: int,
        residual_length: int,
        residual_policy: str = "tail",
    ) -> tuple[torch.Tensor, float]:
        work = layer_tensor.float().clone()
        batch, heads, seq_len, head_dim = work.shape
        residual = self._residual_keep_length(seq_len, residual_length, residual_policy)
        quant_end = seq_len - residual
        if quant_end <= 0 or bits >= FP_BITS - 0.5:
            return work.to(layer_tensor.dtype), FP_BITS

        quant_slice = work[:, :, :quant_end, :]
        if axis == "channel":
            quant_len = quant_slice.shape[-2]
            group_size = quant_len if int(group_size) <= 0 else int(group_size)
            usable = quant_len - (quant_len % group_size)
            main = quant_slice[:, :, :usable, :]
            tail = quant_slice[:, :, usable:, :]
            if usable > 0:
                main = main.transpose(2, 3).reshape(batch, heads, head_dim, usable // group_size, group_size)
                main = self._minmax_quantize_last_dim(main, bits, group_size)
                work[:, :, :usable, :] = main.reshape(batch, heads, head_dim, usable).transpose(2, 3)
            if tail.numel() > 0:
                work[:, :, usable:quant_end, :] = tail
            fp_tokens = residual + (quant_len - usable)
            avg_bits = (usable * bits + fp_tokens * FP_BITS) / max(seq_len, 1)
        else:
            flat = quant_slice.transpose(1, 2).reshape(batch, quant_slice.shape[-2], heads * head_dim)
            flat = self._minmax_quantize_last_dim(flat, bits, group_size)
            work[:, :, :quant_end, :] = flat.reshape(batch, quant_slice.shape[-2], heads, head_dim).transpose(1, 2)
            avg_bits = (quant_end * bits + residual * FP_BITS) / max(seq_len, 1)
        return work.to(layer_tensor.dtype), float(avg_bits)

    def quantize_key(self, layer_id: int, key_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize_grouped_minmax(
            key_states,
            axis="channel",
            bits=self.bits,
            group_size=self.key_group_size,
            residual_length=self.key_residual_length,
            residual_policy="tail",
        )

    def quantize_value(self, layer_id: int, value_states: torch.Tensor, cache_meta: dict) -> tuple[torch.Tensor, float]:
        return self._quantize_grouped_minmax(
            value_states,
            axis="token",
            bits=self.bits,
            group_size=self.value_group_size,
            residual_length=self.value_residual_length,
            residual_policy="tail",
        )

    def estimate_bits(self, layer_id: int, kv_kind: str, seq_len: int, head_dim: int, cache_meta: dict) -> float:
        residual = self.key_residual_length if kv_kind == "key" else self.value_residual_length
        residual = self._residual_keep_length(seq_len, residual, "tail")
        quant_tokens = max(0, seq_len - residual)
        return float((quant_tokens * self.bits + residual * FP_BITS) / max(seq_len, 1))


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
FP_BITS = float(torch.finfo(torch.float16).bits)
DEFAULT_MAX_EXAMPLES = int(os.environ.get("ADAPTIVE_KV_MAX_EXAMPLES", "0"))

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

LONG_BENCH_DATASETS = ("hotpotqa_e", "passage_retrieval_en_e", "repobench-p_e")
NEEDLE_SENTENCE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
NEEDLE_QUESTION = "The best thing to do in San Francisco is: "
NEEDLE_DEPTHS = tuple(
    float(x)
    for x in os.environ.get("ADAPTIVE_KV_NEEDLE_DEPTHS", "0.10,0.50,0.90").split(",")
    if x.strip()
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
    "needlebench_niah": {
        "source_family": "opencompass/NeedleBench English haystacks with the shared RULER/NeedleBench-style NIAH probe",
        "max_new_tokens": 32,
    },
    "gsm8k": {
        "source_family": "openai/gsm8k main test split",
        "max_new_tokens": 256,
    },
}


def load_hf_dataset(repo: str, config: str | None = None, split: str = "test"):
    from datasets import load_dataset

    errors = []
    for candidate_split in (split, "train", None):
        try:
            if config is None:
                dataset = load_dataset(repo, split=candidate_split) if candidate_split else load_dataset(repo)
            else:
                dataset = (
                    load_dataset(repo, config, split=candidate_split)
                    if candidate_split
                    else load_dataset(repo, config)
                )
        except Exception as exc:
            errors.append(exc)
            continue
        if hasattr(dataset, "keys"):
            for key in (split, "test", "validation", "train"):
                if key in dataset:
                    return dataset[key]
            first_key = next(iter(dataset.keys()))
            return dataset[first_key]
        return dataset
    raise RuntimeError(f"Unable to load Hugging Face dataset {repo}/{config or ''}") from errors[-1]


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
        if root in seen or not root.exists():
            continue
        seen.add(root)
        patterns = (
            "openai___gsm8k/main/**/gsm8k-test.arrow",
            "**/openai___gsm8k/main/**/gsm8k-test.arrow",
            "**/gsm8k-test.arrow",
        )
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return Dataset.from_file(str(matches[0]))
    raise FileNotFoundError("Unable to locate cached openai/gsm8k main test split")


def hf_dataset_file(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset"))


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


def load_needle_haystack_rows(max_examples: int = 0) -> list[dict]:
    path = hf_dataset_file("opencompass/NeedleBench", "PaulGrahamEssays.jsonl")
    with path.open() as fh:
        return read_jsonl_lines(fh, max_examples)


def first_string_field(row: dict, preferred: tuple[str, ...]) -> str:
    for key in preferred:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def limit_reached(examples: list[dict], max_examples: int) -> bool:
    return max_examples > 0 and len(examples) >= max_examples


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


def build_needlebench_examples(max_examples: int) -> list[dict]:
    haystacks = load_needle_haystack_rows()
    haystack_texts = [
        first_string_field(row, ("text", "content", "English", "english"))
        for row in haystacks
    ]
    haystack_texts = [text for text in haystack_texts if text]
    if not haystack_texts:
        raise RuntimeError("Unable to build NIAH examples from opencompass/NeedleBench haystacks.")

    examples = []
    for essay_idx, haystack in enumerate(haystack_texts):
        for depth_idx, depth in enumerate(NEEDLE_DEPTHS):
            split = int(len(haystack) * min(max(depth, 0.0), 1.0))
            context = haystack[:split] + "\n" + NEEDLE_SENTENCE + "\n" + haystack[split:]
            prompt = (
                "A single relevant sentence is hidden in the following long document. "
                "Read the document carefully and answer the retrieval question with the exact phrase.\n\n"
                f"{context}\n\nQuestion: {NEEDLE_QUESTION}\nAnswer:"
            )
            examples.append(
                {
                    "example_id": f"niah-{essay_idx}-{depth:.2f}",
                    "dataset": "needlebench_niah",
                    "prompt": prompt,
                    "answers": [NEEDLE_SENTENCE],
                }
            )
            if limit_reached(examples, max_examples):
                return examples
    return examples


def build_gsm8k_examples(max_examples: int) -> list[dict]:
    try:
        dataset = load_hf_dataset("openai/gsm8k", "main")
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
        examples = build_longbench_examples(name, max_examples)
    elif name == "needlebench_niah":
        examples = build_needlebench_examples(max_examples)
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


WORKLOADS = WORKLOAD_CONFIGS


def maybe_write_output_artifacts(workload: str, trace: dict, metrics: dict) -> None:
    output_dir = os.environ.get("OUTPUT_DIR")
    if not output_dir:
        return
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = workload.replace("-", "_")
    (out_path / f"{stem}_trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    (out_path / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def snapshot_cache(cache: DynamicCache) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    snapshot = []
    for layer_idx, layer_entry in enumerate(cache):
        if len(layer_entry) == 2:
            keys, values = layer_entry
            sliding = None
        elif len(layer_entry) == 3:
            keys, values, sliding = layer_entry
        else:
            raise RuntimeError(
                "Unsupported DynamicCache layer entry shape at layer "
                f"{layer_idx}: expected 2 or 3 elements, got {len(layer_entry)}. "
                "This task expects the Transformers DynamicCache API used by the "
                "pinned transformers-kv-lab package."
            )
        snapshot.append(
            (
                keys.detach().clone(),
                values.detach().clone(),
                sliding.detach().clone() if sliding is not None else None,
            )
        )
    return snapshot


def restore_cache(model, cache_snapshot) -> DynamicCache:
    try:
        cache = DynamicCache(config=model.config)
    except TypeError as exc:
        raise RuntimeError(
            "DynamicCache constructor mismatch. This task expects the DynamicCache "
            "API provided by the pinned transformers-kv-lab package."
        ) from exc
    if not hasattr(cache, "layers") or len(cache.layers) < len(cache_snapshot):
        ddp_cache_data = []
        for keys, values, sliding in cache_snapshot:
            ddp_cache_data.append((keys, values) if sliding is None else (keys, values, sliding))
        return DynamicCache(ddp_cache_data=ddp_cache_data, config=model.config)
    for layer_idx, (keys, values, sliding) in enumerate(cache_snapshot):
        layer = cache.layers[layer_idx]
        if hasattr(layer, "lazy_initialization") and not getattr(layer, "is_initialized", False):
            layer.lazy_initialization(keys, values)
        layer.keys = keys
        layer.values = values
        layer.is_initialized = True
        if sliding is not None and hasattr(layer, "_sliding_window_tensor"):
            layer._sliding_window_tensor = sliding
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = int(keys.shape[-2])
    return cache


def stack_snapshot(cache_snapshot, kv_kind: str) -> torch.Tensor:
    index = 0 if kv_kind == "key" else 1
    per_layer = [entry[index].transpose(1, 2) for entry in cache_snapshot]
    return torch.stack(per_layer, dim=2)


def unstack_snapshot(cache_snapshot, kv_kind: str, stacked: torch.Tensor):
    updated = list(cache_snapshot)
    for layer_id in range(stacked.shape[2]):
        keys, values, sliding = updated[layer_id]
        tensor = stacked[:, :, layer_id].transpose(1, 2).contiguous()
        if kv_kind == "key":
            updated[layer_id] = (tensor, values, sliding)
        else:
            updated[layer_id] = (keys, tensor, sliding)
    return updated


def safe_mean(values) -> float:
    return mean(values) if values else 0.0


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
    # Match LongBench/FastKV's fuzz.ratio semantics without adding another
    # runtime dependency to the tensor replay package.
    return int(round(100 * difflib.SequenceMatcher(None, candidate, ground_truth).ratio())) / 100.0


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
    if workload_name == "needlebench_niah":
        normalized_prediction = normalize_text(prediction)
        return 100.0 if any(normalize_text(answer) in normalized_prediction for answer in answers) else 0.0
    if workload_name == "gsm8k":
        prediction_answer = normalize_math_answer(extract_math_answer(prediction))
        gold_answers = {normalize_math_answer(answer) for answer in answers}
        return 100.0 if prediction_answer and prediction_answer in gold_answers else 0.0
    raise ValueError(f"Unsupported workload/example dataset: workload={workload_name}, dataset={dataset}")


def decode_prediction(tokenizer, generated_tokens: torch.Tensor, workload_name: str) -> str:
    text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
    if workload_name == "gsm8k":
        return text
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_to_query(query_states: torch.Tensor, position_embeddings) -> torch.Tensor:
    if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
        return query_states
    cos, sin = position_embeddings
    if cos is None or sin is None or cos.shape[-1] != query_states.shape[-1]:
        return query_states
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    while cos.dim() < query_states.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos[..., -query_states.shape[-2] :, :].to(device=query_states.device, dtype=query_states.dtype)
    sin = sin[..., -query_states.shape[-2] :, :].to(device=query_states.device, dtype=query_states.dtype)
    return (query_states * cos) + (rotate_half(query_states) * sin)


def normalize_quantizer_result(result, original_tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    if isinstance(result, tuple) and len(result) == 2:
        quantized, avg_bits = result
    else:
        quantized, avg_bits = result, FP_BITS
    if not isinstance(quantized, torch.Tensor):
        raise TypeError("quantize_key/quantize_value must return a tensor or (tensor, avg_bits)")
    if tuple(quantized.shape) != tuple(original_tensor.shape):
        raise ValueError(f"Quantized tensor shape {tuple(quantized.shape)} does not match {tuple(original_tensor.shape)}")
    return quantized.to(original_tensor.dtype), float(avg_bits)


def quantize_snapshot(cache_snapshot, quantizer: AdaptiveKVQuantizer, request_meta: dict, budget_state: dict):
    quantized_snapshot = list(cache_snapshot)
    kv_bits = []

    for kv_kind in ("key", "value"):
        tensor_index = 0 if kv_kind == "key" else 1
        for layer_id, entry in enumerate(quantized_snapshot):
            keys, values, sliding = entry
            layer_tensor = keys if tensor_index == 0 else values
            cache_meta = {
                "kv_kind": kv_kind,
                "seq_len": int(layer_tensor.shape[-2]),
                "heads": int(layer_tensor.shape[1]),
                "head_dim": int(layer_tensor.shape[-1]),
                "request_meta": request_meta,
                "budget_state": budget_state,
            }
            if tensor_index == 0:
                result = quantizer.quantize_key(layer_id, layer_tensor, cache_meta)
            else:
                result = quantizer.quantize_value(layer_id, layer_tensor, cache_meta)
            quantized_layer, avg_bits = normalize_quantizer_result(result, layer_tensor)
            kv_bits.append(float(avg_bits))
            if tensor_index == 0:
                quantized_snapshot[layer_id] = (quantized_layer.contiguous(), values, sliding)
            else:
                quantized_snapshot[layer_id] = (keys, quantized_layer.contiguous(), sliding)

    effective_bits = safe_mean(kv_bits)
    return quantized_snapshot, {
        "effective_kv_bits": effective_bits,
        "kv_compression_ratio": FP_BITS / max(effective_bits, 1e-6),
    }


def estimate_policy_efficiency(quantizer: AdaptiveKVQuantizer, workload_name: str, budget_bits: int, num_layers: int) -> dict:
    budget_state = {"budget_bits": budget_bits}
    request_meta = {"workload": workload_name}
    quantizer.reset_request(request_meta, budget_state)
    kv_bits = []
    reference_span = 4096
    for layer_id in range(num_layers):
        for kv_kind in ("key", "value"):
            cache_meta = {
                "kv_kind": kv_kind,
                "seq_len": reference_span,
                "heads": 8,
                "head_dim": 128,
                "request_meta": request_meta,
                "budget_state": budget_state,
            }
            kv_bits.append(float(quantizer.estimate_bits(layer_id, kv_kind, reference_span, 128, cache_meta)))
    effective_bits = safe_mean(kv_bits)
    return {
        "effective_kv_bits": effective_bits,
        "kv_compression_ratio": FP_BITS / max(effective_bits, 1e-6),
    }


def register_prefill_observation_hooks(model, quantizer: AdaptiveKVQuantizer):
    if not bool(quantizer.needs_prefill_qkv_observer()):
        return [], {}
    captured = {}
    handles = []
    pre_kwargs = {}
    modules = [
        module
        for _, module in model.named_modules()
        if hasattr(module, "q_proj") and module.__class__.__name__.lower().endswith("attention")
    ]
    if not modules:
        modules = [
            module
            for _, module in model.named_modules()
            if hasattr(module, "q_proj") and hasattr(module, "num_heads")
    ]
    if not modules:
        raise RuntimeError("Prefill observer could not find attention modules with q_proj")

    def make_pre_hook(layer_id):
        def hook(_module, _args, kwargs):
            pre_kwargs[layer_id] = kwargs

        return hook

    def make_q_hook(layer_id, attn_module):
        def hook(_module, _inputs, output):
            raw = output[0] if isinstance(output, (tuple, list)) else output
            if raw is None or raw.dim() != 3:
                return
            num_heads = int(getattr(attn_module, "num_heads", getattr(model.config, "num_attention_heads", 0)))
            if num_heads <= 0 or raw.shape[-1] % num_heads != 0:
                return
            head_dim = int(getattr(attn_module, "head_dim", raw.shape[-1] // num_heads))
            query_states = raw.reshape(raw.shape[0], raw.shape[1], num_heads, head_dim).transpose(1, 2).contiguous()
            if str(quantizer.query_observation_position()) == "post_rope":
                query_states = apply_rotary_to_query(query_states, pre_kwargs.get(layer_id, {}).get("position_embeddings"))
            kv_heads = int(getattr(attn_module, "num_key_value_heads", getattr(model.config, "num_key_value_heads", num_heads)))
            captured[layer_id] = True
            quantizer.observe_prefill_qkv(
                layer_id,
                query_states.detach(),
                None,
                None,
                {"kv_heads": kv_heads, "position_embeddings": pre_kwargs.get(layer_id, {}).get("position_embeddings")},
            )

        return hook

    for layer_id, module in enumerate(modules):
        try:
            handles.append(module.register_forward_pre_hook(make_pre_hook(layer_id), with_kwargs=True))
        except TypeError:
            handles.append(module.register_forward_pre_hook(lambda _module, _args: None))
        handles.append(module.q_proj.register_forward_hook(make_q_hook(layer_id, module)))
    return handles, captured


def prefill_prompt(model, tokenizer, prompt: str, device: str, quantizer: AdaptiveKVQuantizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    cache = DynamicCache(config=model.config)
    hooks, _ = register_prefill_observation_hooks(model, quantizer)
    maybe_sync()
    try:
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=True, past_key_values=cache, output_attentions=False, logits_to_keep=1)
    finally:
        for handle in hooks:
            handle.remove()
    maybe_sync()
    first_token = outputs.logits[:, -1:].argmax(dim=-1).detach()
    current_cache = outputs.past_key_values
    del outputs
    return first_token, current_cache


def generate_with_quantized_cache(
    model,
    tokenizer,
    example: dict,
    workload_name: str,
    max_new_tokens: int,
    quantizer: AdaptiveKVQuantizer,
    budget_state: dict,
    device: str,
) -> tuple[str, dict]:
    prompt = example["prompt"]
    request_meta = {
        "workload": workload_name,
        "prompt_chars": len(prompt),
        "example_id": example.get("example_id", ""),
        "dataset": example.get("dataset", ""),
    }
    quantizer.reset_request(request_meta, budget_state)
    step_input, current_cache = prefill_prompt(model, tokenizer, prompt, device, quantizer)
    generated = [step_input.detach().clone()]
    effective_kv_bits = []
    kv_compression_ratios = []

    for _ in range(max_new_tokens - 1):
        cache_snapshot = snapshot_cache(current_cache)
        quantized_snapshot, efficiency = quantize_snapshot(cache_snapshot, quantizer, request_meta, budget_state)
        effective_kv_bits.append(efficiency["effective_kv_bits"])
        kv_compression_ratios.append(efficiency["kv_compression_ratio"])
        quantized_cache = restore_cache(model, quantized_snapshot)
        del cache_snapshot, quantized_snapshot
        maybe_sync()
        with torch.inference_mode():
            outputs = model(
                input_ids=step_input,
                use_cache=True,
                past_key_values=quantized_cache,
                logits_to_keep=1,
            )
        maybe_sync()
        step_input = outputs.logits[:, -1:].argmax(dim=-1).detach()
        current_cache = outputs.past_key_values
        generated.append(step_input.detach().clone())
        del quantized_cache, outputs
        if tokenizer.eos_token_id is not None and int(step_input.item()) == int(tokenizer.eos_token_id):
            break

    generated_tokens = torch.cat(generated, dim=-1)
    return decode_prediction(tokenizer, generated_tokens, workload_name), {
        "effective_kv_bits": safe_mean(effective_kv_bits),
        "kv_compression_ratio": safe_mean(kv_compression_ratios),
    }


def maybe_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def simulate_metrics(quantizer: AdaptiveKVQuantizer, workload_name: str, budget_bits: int) -> dict:
    efficiency = estimate_policy_efficiency(quantizer, workload_name, budget_bits, num_layers=36)
    compression_gain = 1.0 - efficiency["effective_kv_bits"] / FP_BITS
    return {"final_score": 50.0 + 10.0 * compression_gain, **efficiency}


def run_real_eval(args):
    import time

    runtime_start = time.perf_counter()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    workload = load_workload(args.workload, args.max_examples)
    quantizer = AdaptiveKVQuantizer()
    budget_state = {"budget_bits": args.budget_bits}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    final_scores = []
    effective_kv_bits = []
    kv_compression_ratios = []

    for example in workload.examples:
        prediction, efficiency = generate_with_quantized_cache(
            model,
            tokenizer,
            example,
            args.workload,
            workload.max_new_tokens,
            quantizer,
            budget_state,
            device,
        )
        final_scores.append(score_prediction(args.workload, example, prediction))
        if efficiency["effective_kv_bits"] > 0:
            effective_kv_bits.append(efficiency["effective_kv_bits"])
            kv_compression_ratios.append(efficiency["kv_compression_ratio"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    num_layers = int(getattr(model.config, "num_hidden_layers", 36))
    efficiency = estimate_policy_efficiency(quantizer, args.workload, args.budget_bits, num_layers)

    metrics = {"final_score": safe_mean(final_scores), **efficiency, "runtime_seconds": time.perf_counter() - runtime_start}
    maybe_write_output_artifacts(args.workload, {}, metrics)
    print(
        "TEST_METRICS: "
        f"final_score={metrics['final_score']:.6f} "
        f"effective_kv_bits={metrics['effective_kv_bits']:.6f} "
        f"kv_compression_ratio={metrics['kv_compression_ratio']:.6f} "
        f"runtime_seconds={metrics['runtime_seconds']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--budget-bits", type=int, required=True)
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    args = parser.parse_args()

    quantizer = AdaptiveKVQuantizer()
    if args.mock:
        import time

        runtime_start = time.perf_counter()
        metrics = simulate_metrics(quantizer, args.workload, args.budget_bits)
        metrics["runtime_seconds"] = time.perf_counter() - runtime_start
        maybe_write_output_artifacts(args.workload, {}, metrics)
        print(
            "TEST_METRICS: "
            f"final_score={metrics['final_score']:.6f} "
            f"effective_kv_bits={metrics['effective_kv_bits']:.6f} "
            f"kv_compression_ratio={metrics['kv_compression_ratio']:.6f} "
            f"runtime_seconds={metrics['runtime_seconds']:.6f}"
        )
        return

    run_real_eval(args)


if __name__ == "__main__":
    main()
