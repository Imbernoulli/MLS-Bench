"""Trusted full-scale inference and scoring for the ``qa-*`` task family."""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import math
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Any


PROTOCOL = "qa-official-full-v2"
SEED = 42
MAX_SEQ_LEN = 384
DOC_STRIDE = 128
MAX_QUERY_LENGTH = 64
N_BEST = 20
MAX_ANSWER_LENGTH = 30
BATCH_SIZE = 16
EXAMPLE_CHUNK_SIZE = 64

MODEL_REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
MODEL_MANIFEST = {
    "config.json": "64fa58495a722d57609c22f199824bfe98c19be068136a70c268214a08cb8060",
    "model.safetensors": "ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "vocab.json": "06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab",
    "tokenizer_config.json": "7a33226d4265e3989cc6341666af179d0cc710136f4059aae0dd8c0797cba556",
    "special_tokens_map.json": "c611b1f7d416eb001ee4f293d903ea8c88e703463f1d403f1866a0352743fd00",
}
MODEL_WEIGHT_SHA256 = MODEL_MANIFEST["model.safetensors"]
MODEL_MANIFEST_SHA256 = hashlib.sha256(
    json.dumps(MODEL_MANIFEST, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
MODEL_PARAMETER_COUNT = 124_056_578

DATASET_MANIFEST = {
    "mrqa_squad_validation.jsonl": {
        "sha256": "64ab3a4c69574a258c934044a63605b15d98e1608fa9fb5b244868c5d0af89aa",
        "n": 10_507,
        "n_ans": 10_507,
        "n_noans": 0,
        "kind": "mrqa",
    },
    "mrqa_newsqa_validation.jsonl": {
        "sha256": "87b31cff3db4cb8276ddc58c94b03ca3ca500a72af95b8b9e2c63c9266ded7ad",
        "n": 4_212,
        "n_ans": 4_212,
        "n_noans": 0,
        "kind": "mrqa",
    },
    "mrqa_hotpotqa_validation.jsonl": {
        "sha256": "a335e1778d3c2de3a99b00e8eeaa3fc6e9b611386afadcc54532c2f33d3d95ad",
        "n": 5_901,
        "n_ans": 5_901,
        "n_noans": 0,
        "kind": "mrqa",
    },
    "mrqa_naturalquestions_validation.jsonl": {
        "sha256": "705717e225fc972d9a1df01737ab11d59a2c573a6ba9e7018b5ace4c34de6952",
        "n": 12_836,
        "n_ans": 12_836,
        "n_noans": 0,
        "kind": "mrqa",
    },
    "squad2_validation_part0.jsonl": {
        "sha256": "bdb7f256bf8893edef347623c6698a16320608d5ddf31c774de8e8234598f5b9",
        "n": 3_958,
        "n_ans": 1_988,
        "n_noans": 1_970,
        "kind": "squad2",
    },
    "squad2_validation_part1.jsonl": {
        "sha256": "4159c7c652415873aa565af317a8c0d460164b5f80b185a35b9cbe6dac40f327",
        "n": 3_958,
        "n_ans": 1_956,
        "n_noans": 2_002,
        "kind": "squad2",
    },
    "squad2_validation_part2.jsonl": {
        "sha256": "4b8fff6cb1dd3370416e1cf36cb7d8ba846ef61fd2cb086ccd02ad80a97ce651",
        "n": 3_957,
        "n_ans": 1_984,
        "n_noans": 1_973,
        "kind": "squad2",
    },
}

_ROW_KEYS = {"id", "question", "context", "answers", "is_impossible"}
_PUNCTUATION = set(string.punctuation)


def data_root() -> Path:
    return Path(os.environ.get("QA_DATA", "/data/extractive-qa/data"))


def model_path() -> Path:
    return Path(
        os.environ.get("QA_MODEL", "/data/extractive-qa/models/roberta-base-squad2")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup(seed: int = SEED):
    """Set deterministic inference state and require one CUDA device."""
    import random

    import numpy as np
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        return torch.device("cuda:0")
    if os.environ.get("QA_ALLOW_CPU") == "1":
        return torch.device("cpu")
    raise RuntimeError("QA_RUNTIME_ERROR CUDA device is required")


def verify_model_files() -> None:
    root = model_path()
    if not root.is_dir():
        raise RuntimeError("QA_MODEL_ERROR model directory is missing")
    allowed_entries = set(MODEL_MANIFEST) | {".cache"}
    unexpected_entries = sorted(path.name for path in root.iterdir() if path.name not in allowed_entries)
    if unexpected_entries:
        raise RuntimeError(
            "QA_MODEL_ERROR unexpected model directory entries: "
            + ", ".join(unexpected_entries)
        )
    for filename, expected_sha in MODEL_MANIFEST.items():
        path = root / filename
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"QA_MODEL_ERROR missing {filename}")
        observed_sha = _sha256(path)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"QA_MODEL_ERROR hash mismatch for {filename}: {observed_sha}"
            )


def load_model_and_tokenizer(device):
    """Load the exact frozen checkpoint after verifying every required file."""
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    verify_model_files()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path(), local_files_only=True, use_fast=True
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_path(), local_files_only=True, torch_dtype=torch.float32
        )
    except Exception as exc:
        raise RuntimeError(f"QA_MODEL_ERROR checkpoint load failed: {exc}") from exc
    if not tokenizer.is_fast:
        raise RuntimeError("QA_MODEL_ERROR a fast tokenizer is required for offsets")
    if tokenizer.padding_side != "right":
        raise RuntimeError("QA_MODEL_ERROR tokenizer padding_side must be right")
    observed_parameters = sum(parameter.numel() for parameter in model.parameters())
    if observed_parameters != MODEL_PARAMETER_COUNT:
        raise RuntimeError(
            "QA_MODEL_ERROR parameter count mismatch: "
            f"observed={observed_parameters} expected={MODEL_PARAMETER_COUNT}"
        )
    model.to(device)
    model.eval()
    return model, tokenizer


def load_dataset(filename: str) -> list[dict]:
    """Load one complete fixture and verify its bytes, schema, and cardinality."""
    if filename not in DATASET_MANIFEST or Path(filename).name != filename:
        raise RuntimeError(f"QA_DATA_ERROR unknown dataset {filename!r}")
    spec = DATASET_MANIFEST[filename]
    path = data_root() / filename
    if not path.is_file():
        raise RuntimeError(f"QA_DATA_ERROR missing {filename}")
    observed_sha = _sha256(path)
    if observed_sha != spec["sha256"]:
        raise RuntimeError(
            f"QA_DATA_ERROR hash mismatch for {filename}: {observed_sha}"
        )

    rows: list[dict] = []
    ids: set[str] = set()
    n_ans = n_noans = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                if not raw_line.endswith("\n") or not raw_line.strip():
                    raise ValueError(f"invalid JSONL record at line {line_number}")
                row = json.loads(raw_line)
                if type(row) is not dict or set(row) != _ROW_KEYS:
                    raise ValueError(f"invalid schema at line {line_number}")
                if type(row["id"]) is not str or not row["id"]:
                    raise ValueError(f"invalid id at line {line_number}")
                if row["id"] in ids:
                    raise ValueError(f"duplicate id {row['id']!r}")
                if (
                    type(row["question"]) is not str
                    or not row["question"]
                    or row["question"] != row["question"].strip()
                ):
                    raise ValueError(f"invalid question at line {line_number}")
                if type(row["context"]) is not str or not row["context"]:
                    raise ValueError(f"invalid context at line {line_number}")
                if type(row["answers"]) is not list or any(
                    type(answer) is not str or not answer for answer in row["answers"]
                ):
                    raise ValueError(f"invalid answers at line {line_number}")
                if len(row["answers"]) != len(set(row["answers"])):
                    raise ValueError(f"duplicate answers at line {line_number}")
                if type(row["is_impossible"]) is not bool:
                    raise ValueError(f"invalid is_impossible at line {line_number}")
                if row["is_impossible"] != (len(row["answers"]) == 0):
                    raise ValueError(f"answerability mismatch at line {line_number}")

                canonical = json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n"
                if canonical != raw_line:
                    raise ValueError(f"non-canonical record at line {line_number}")

                ids.add(row["id"])
                rows.append(row)
                if row["is_impossible"]:
                    n_noans += 1
                else:
                    n_ans += 1
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"QA_DATA_ERROR {filename}: {exc}") from exc

    observed = (len(rows), n_ans, n_noans)
    expected = (spec["n"], spec["n_ans"], spec["n_noans"])
    if observed != expected:
        raise RuntimeError(
            f"QA_DATA_ERROR cardinality mismatch for {filename}: "
            f"observed={observed} expected={expected}"
        )
    return rows


def _surface_error(message: str) -> ValueError:
    return ValueError(f"QA_SURFACE_ERROR {message}")


def _validate_surface_value(attribute: str, value: Any) -> Any:
    enums = {
        "build_casing": {"preserve", "lowercase"},
        "build_encoding_order": {"question_first", "context_first"},
        "build_question_mode": {"real", "drop"},
        "build_decoder": {"argmax", "constrained"},
        "build_span_aggregation": {"first_feature", "max_score"},
    }
    if attribute in enums:
        if type(value) is not str or value not in enums[attribute]:
            raise _surface_error(
                f"{attribute} must return one of {sorted(enums[attribute])}"
            )
        return value
    if attribute == "build_max_answer_length":
        if type(value) is not int or not 1 <= value <= 200:
            raise _surface_error(f"{attribute} must return an integer in [1, 200]")
        return value
    if attribute == "build_max_seq_len":
        if (
            type(value) is not int
            or not 128 <= value <= 512
            or value % 64 != 0
        ):
            raise _surface_error(
                f"{attribute} must return a multiple of 64 in [128, 512]"
            )
        return value
    if attribute == "build_n_best":
        if type(value) is not int or not 1 <= value <= 50:
            raise _surface_error(f"{attribute} must return an integer in [1, 50]")
        return value
    if attribute == "build_null_threshold":
        if (
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or not -30.0 <= float(value) <= 30.0
        ):
            raise _surface_error(
                f"{attribute} must return a finite number in [-30, 30]"
            )
        return float(value)
    if attribute == "build_doc_stride":
        if type(value) is not int or value not in {0, 64, 128}:
            raise _surface_error(f"{attribute} must return one of [0, 64, 128]")
        return value
    raise _surface_error(f"unknown callable {attribute!r}")


def load_surface_value(solution_path: str, attribute: str) -> Any:
    """Parse one zero-argument literal-return function without executing Python."""
    path = Path(solution_path)
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _surface_error(f"cannot read solution: {exc}") from exc
    if len(source.encode("utf-8")) > 65_536:
        raise _surface_error("solution exceeds 64 KiB")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise _surface_error(f"invalid syntax: {exc.msg}") from exc

    body = list(tree.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and type(body[0].value.value) is str
    ):
        body.pop(0)
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
        raise _surface_error("module must contain exactly one function")
    function = body[0]
    args = function.args
    if (
        function.name != attribute
        or function.decorator_list
        or function.returns is not None
        or function.type_comment is not None
        or args.posonlyargs
        or args.args
        or args.vararg is not None
        or args.kwonlyargs
        or args.kw_defaults
        or args.kwarg is not None
        or args.defaults
        or len(function.body) != 1
        or not isinstance(function.body[0], ast.Return)
        or function.body[0].value is None
    ):
        raise _surface_error(
            f"solution must define only zero-argument {attribute}() with one return"
        )
    try:
        value = ast.literal_eval(function.body[0].value)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise _surface_error("return value must be a Python literal") from exc
    return _validate_surface_value(attribute, value)


def _width_preserving_lower(text: str) -> str:
    lowered: list[str] = []
    for character in text:
        candidate = character.lower()
        lowered.append(candidate if len(candidate) == 1 else character)
    result = "".join(lowered)
    if len(result) != len(text):
        raise RuntimeError("QA_INFERENCE_ERROR casing changed character offsets")
    return result


def _truncate_questions(tokenizer, questions: list[str]) -> list[str]:
    encoded = tokenizer(
        questions,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_QUERY_LENGTH,
        return_offsets_mapping=True,
    )
    result: list[str] = []
    for question, offsets in zip(questions, encoded["offset_mapping"]):
        if not offsets:
            raise RuntimeError("QA_INFERENCE_ERROR empty tokenized question")
        end = int(offsets[-1][1])
        result.append(question[:end])
    if len(result) != len(questions):
        raise RuntimeError("QA_INFERENCE_ERROR question tokenization mismatch")
    return result


def build_features(
    tokenizer,
    rows: list[dict],
    *,
    question_first: bool = True,
    lowercase: bool = False,
    max_seq_len: int = MAX_SEQ_LEN,
    doc_stride: int = DOC_STRIDE,
    drop_question: bool = False,
) -> tuple[list[dict], list[int]]:
    """Build standard overflow features for one bounded example chunk."""
    if not rows:
        raise RuntimeError("QA_INFERENCE_ERROR empty example chunk")
    if type(max_seq_len) is not int or not 128 <= max_seq_len <= 512:
        raise RuntimeError("QA_INFERENCE_ERROR invalid max_seq_len")
    if type(doc_stride) is not int or doc_stride < 0:
        raise RuntimeError("QA_INFERENCE_ERROR invalid doc_stride")

    questions = [row["question"] for row in rows]
    contexts_model = [row["context"] for row in rows]
    if lowercase:
        questions = [_width_preserving_lower(text) for text in questions]
        contexts_model = [_width_preserving_lower(text) for text in contexts_model]
    if drop_question:
        questions = ["what" for _ in rows]
    questions = _truncate_questions(tokenizer, questions)

    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
    min_context_capacity = max_seq_len - MAX_QUERY_LENGTH - special_tokens
    if min_context_capacity <= 1 or doc_stride >= min_context_capacity:
        raise RuntimeError(
            "QA_INFERENCE_ERROR doc_stride must be smaller than context capacity"
        )

    if question_first:
        text_a, text_b = questions, contexts_model
        truncation = "only_second"
        context_sequence_id = 1
    else:
        text_a, text_b = contexts_model, questions
        truncation = "only_first"
        context_sequence_id = 0
    try:
        encoding = tokenizer(
            text_a,
            text_b,
            truncation=truncation,
            max_length=max_seq_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
    except Exception as exc:
        raise RuntimeError(f"QA_INFERENCE_ERROR tokenization failed: {exc}") from exc

    sample_map = list(encoding["overflow_to_sample_mapping"])
    features: list[dict] = []
    for feature_index, sample_index_raw in enumerate(sample_map):
        sample_index = int(sample_index_raw)
        if not 0 <= sample_index < len(rows):
            raise RuntimeError("QA_INFERENCE_ERROR invalid overflow sample mapping")
        input_ids = list(encoding["input_ids"][feature_index])
        attention_mask = list(encoding["attention_mask"][feature_index])
        if len(input_ids) != max_seq_len or len(attention_mask) != max_seq_len:
            raise RuntimeError("QA_INFERENCE_ERROR malformed padded feature")
        sequence_ids = encoding.sequence_ids(feature_index)
        raw_offsets = encoding["offset_mapping"][feature_index]
        offsets = [
            (int(offset[0]), int(offset[1]))
            if sequence_ids[position] == context_sequence_id
            else None
            for position, offset in enumerate(raw_offsets)
        ]
        context = rows[sample_index]["context"]
        context_positions = [position for position, offset in enumerate(offsets) if offset]
        if not context_positions:
            raise RuntimeError("QA_INFERENCE_ERROR feature has no context tokens")
        for position in context_positions:
            start, end = offsets[position]
            if not 0 <= start <= end <= len(context):
                raise RuntimeError("QA_INFERENCE_ERROR invalid context offset")
        try:
            cls_index = input_ids.index(tokenizer.cls_token_id)
        except ValueError as exc:
            raise RuntimeError("QA_INFERENCE_ERROR feature has no CLS token") from exc
        features.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offsets,
                "cls_index": cls_index,
                "_context": context,
            }
        )
    if not features or len(features) != len(sample_map):
        raise RuntimeError("QA_INFERENCE_ERROR no overflow features produced")
    return features, sample_map


def run_logits(model, features: list[dict], device):
    """Run all features and reject incomplete or non-finite model outputs."""
    import numpy as np
    import torch

    if not features:
        raise RuntimeError("QA_INFERENCE_ERROR no features for forward pass")
    all_start: list[Any] = []
    all_end: list[Any] = []
    for offset in range(0, len(features), BATCH_SIZE):
        batch = features[offset : offset + BATCH_SIZE]
        input_ids = torch.tensor(
            [feature["input_ids"] for feature in batch],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.tensor(
            [feature["attention_mask"] for feature in batch],
            dtype=torch.long,
            device=device,
        )
        try:
            with torch.inference_mode():
                output = model(input_ids=input_ids, attention_mask=attention_mask)
        except Exception as exc:
            raise RuntimeError(f"QA_INFERENCE_ERROR model forward failed: {exc}") from exc
        start = output.start_logits.detach().float().cpu().numpy()
        end = output.end_logits.detach().float().cpu().numpy()
        expected_shape = (len(batch), len(batch[0]["input_ids"]))
        if start.shape != expected_shape or end.shape != expected_shape:
            raise RuntimeError("QA_INFERENCE_ERROR logit shape mismatch")
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            raise RuntimeError("QA_INFERENCE_ERROR non-finite logits")
        all_start.append(start)
        all_end.append(end)
    starts = np.concatenate(all_start, axis=0)
    ends = np.concatenate(all_end, axis=0)
    if starts.shape[0] != len(features) or ends.shape != starts.shape:
        raise RuntimeError("QA_INFERENCE_ERROR incomplete logits")
    return starts, ends


def _context_positions(feature: dict) -> list[int]:
    positions = [
        index
        for index, offset in enumerate(feature["offset_mapping"])
        if offset is not None
    ]
    if not positions:
        raise RuntimeError("QA_INFERENCE_ERROR no decodable context positions")
    return positions


def decode_argmax_independent(
    feature: dict,
    start_logit,
    end_logit,
    *,
    max_answer_length: int,
):
    """Decode independent argmax under the shared span-length constraint."""
    if type(max_answer_length) is not int or not 1 <= max_answer_length <= 200:
        raise RuntimeError("QA_INFERENCE_ERROR invalid max_answer_length")
    positions = _context_positions(feature)
    start = max(positions, key=lambda index: float(start_logit[index]))
    end = max(positions, key=lambda index: float(end_logit[index]))
    score = float(start_logit[start] + end_logit[end])
    if not math.isfinite(score):
        raise RuntimeError("QA_INFERENCE_ERROR non-finite argmax score")
    if end < start or end - start + 1 > max_answer_length:
        return "", score
    offsets = feature["offset_mapping"]
    text = feature["_context"][offsets[start][0] : offsets[end][1]].strip()
    return text, score


def decode_constrained(
    feature: dict,
    start_logit,
    end_logit,
    *,
    n_best: int,
    max_answer_length: int,
):
    """Apply the standard top-k valid-span search to one feature."""
    import numpy as np

    if type(n_best) is not int or not 1 <= n_best <= 50:
        raise RuntimeError("QA_INFERENCE_ERROR invalid n_best")
    if type(max_answer_length) is not int or not 1 <= max_answer_length <= 200:
        raise RuntimeError("QA_INFERENCE_ERROR invalid max_answer_length")
    offsets = feature["offset_mapping"]
    start_indices = np.argsort(start_logit)[-n_best:][::-1].tolist()
    end_indices = np.argsort(end_logit)[-n_best:][::-1].tolist()
    best_text = ""
    best_score: float | None = None
    for start in start_indices:
        if offsets[start] is None:
            continue
        for end in end_indices:
            if offsets[end] is None:
                continue
            if end < start or end - start + 1 > max_answer_length:
                continue
            score = float(start_logit[start] + end_logit[end])
            if not math.isfinite(score):
                raise RuntimeError("QA_INFERENCE_ERROR non-finite span score")
            if best_score is None or score > best_score:
                best_score = score
                best_text = feature["_context"][
                    offsets[start][0] : offsets[end][1]
                ].strip()
    return best_text, best_score


def null_score(feature: dict, start_logit, end_logit) -> float:
    cls_index = feature["cls_index"]
    score = float(start_logit[cls_index] + end_logit[cls_index])
    if not math.isfinite(score):
        raise RuntimeError("QA_INFERENCE_ERROR non-finite null score")
    return score


def _groups(sample_map: list[int], n_examples: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for feature_position, example_index in enumerate(sample_map):
        groups[int(example_index)].append(feature_position)
    if set(groups) != set(range(n_examples)):
        raise RuntimeError("QA_INFERENCE_ERROR incomplete feature coverage")
    return groups


def predict(
    model,
    tokenizer,
    device,
    rows: list[dict],
    *,
    method: str = "constrained",
    n_best: int = N_BEST,
    max_answer_length: int = MAX_ANSWER_LENGTH,
    question_first: bool = True,
    lowercase: bool = False,
    max_seq_len: int = MAX_SEQ_LEN,
    doc_stride: int = DOC_STRIDE,
    drop_question: bool = False,
    aggregation: str = "max_score",
    null_threshold: float | None = None,
) -> tuple[dict[str, str], int]:
    """Stream complete examples through tokenization, inference, and decoding."""
    if method not in {"argmax", "constrained"}:
        raise RuntimeError("QA_INFERENCE_ERROR invalid decode method")
    if aggregation not in {"first_feature", "max_score"}:
        raise RuntimeError("QA_INFERENCE_ERROR invalid span aggregation")
    if null_threshold is not None and (
        method != "constrained" or aggregation != "max_score"
    ):
        raise RuntimeError("QA_INFERENCE_ERROR null decoding requires constrained max_score")

    predictions: dict[str, str] = {}
    n_features = 0
    for chunk_start in range(0, len(rows), EXAMPLE_CHUNK_SIZE):
        chunk = rows[chunk_start : chunk_start + EXAMPLE_CHUNK_SIZE]
        features, sample_map = build_features(
            tokenizer,
            chunk,
            question_first=question_first,
            lowercase=lowercase,
            max_seq_len=max_seq_len,
            doc_stride=doc_stride,
            drop_question=drop_question,
        )
        starts, ends = run_logits(model, features, device)
        groups = _groups(sample_map, len(chunk))
        n_features += len(features)

        for example_index, row in enumerate(chunk):
            positions = groups[example_index]
            decode_positions = positions[:1] if aggregation == "first_feature" else positions
            best_text = ""
            best_score: float | None = None
            for position in decode_positions:
                if method == "argmax":
                    text, score = decode_argmax_independent(
                        features[position],
                        starts[position],
                        ends[position],
                        max_answer_length=max_answer_length,
                    )
                else:
                    text, score = decode_constrained(
                        features[position],
                        starts[position],
                        ends[position],
                        n_best=n_best,
                        max_answer_length=max_answer_length,
                    )
                if score is not None and (best_score is None or score > best_score):
                    best_text, best_score = text, score

            if null_threshold is not None:
                if best_score is None:
                    raise RuntimeError(
                        f"QA_INFERENCE_ERROR no valid non-null span for {row['id']}"
                    )
                minimum_null = min(
                    null_score(features[position], starts[position], ends[position])
                    for position in positions
                )
                if minimum_null - best_score > null_threshold:
                    best_text = ""

            if row["id"] in predictions:
                raise RuntimeError(f"QA_INFERENCE_ERROR duplicate prediction {row['id']}")
            if type(best_text) is not str:
                raise RuntimeError("QA_INFERENCE_ERROR prediction is not text")
            predictions[row["id"]] = best_text

    expected_ids = {row["id"] for row in rows}
    if set(predictions) != expected_ids or len(predictions) != len(rows):
        raise RuntimeError("QA_INFERENCE_ERROR incomplete prediction set")
    if n_features < len(rows):
        raise RuntimeError("QA_INFERENCE_ERROR incomplete feature count")
    return predictions, n_features


def _normalize(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(character for character in value if character not in _PUNCTUATION)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def _tokens(text: str) -> list[str]:
    return _normalize(text).split() if text else []


def _f1(gold_text: str, prediction_text: str) -> float:
    gold = _tokens(gold_text)
    prediction = _tokens(prediction_text)
    common = collections.Counter(gold) & collections.Counter(prediction)
    same = sum(common.values())
    if not gold or not prediction:
        return float(gold == prediction)
    if same == 0:
        return 0.0
    precision = same / len(prediction)
    recall = same / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def _exact_match(gold_text: str, prediction_text: str) -> float:
    return float(_normalize(gold_text) == _normalize(prediction_text))


def score_squad(predictions: dict[str, str], rows: list[dict]) -> dict[str, float | int]:
    """Score the exact complete prediction key set with official SQuAD F1/EM."""
    expected_ids = [row["id"] for row in rows]
    if not rows or len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("QA_METRIC_ERROR invalid evaluation rows")
    if set(predictions) != set(expected_ids) or len(predictions) != len(rows):
        raise RuntimeError("QA_METRIC_ERROR prediction IDs do not match dataset IDs")
    if any(type(value) is not str for value in predictions.values()):
        raise RuntimeError("QA_METRIC_ERROR every prediction must be text")

    f1_all: list[float] = []
    em_all: list[float] = []
    f1_answerable: list[float] = []
    em_answerable: list[float] = []
    f1_unanswerable: list[float] = []
    em_unanswerable: list[float] = []
    for row in rows:
        prediction = predictions[row["id"]]
        golds = row["answers"] if row["answers"] else [""]
        f1 = max(_f1(gold, prediction) for gold in golds)
        em = max(_exact_match(gold, prediction) for gold in golds)
        f1_all.append(f1)
        em_all.append(em)
        if row["is_impossible"]:
            f1_unanswerable.append(f1)
            em_unanswerable.append(em)
        else:
            f1_answerable.append(f1)
            em_answerable.append(em)

    def percentage(values: list[float]) -> float:
        return 100.0 * math.fsum(values) / len(values) if values else 0.0

    metrics: dict[str, float | int] = {
        "f1": percentage(f1_all),
        "em": percentage(em_all),
        "f1_ans": percentage(f1_answerable),
        "em_ans": percentage(em_answerable),
        "f1_noans": percentage(f1_unanswerable),
        "em_noans": percentage(em_unanswerable),
        "n": len(rows),
        "n_ans": len(f1_answerable),
        "n_noans": len(f1_unanswerable),
    }
    for name in ("f1", "em", "f1_ans", "em_ans", "f1_noans", "em_noans"):
        value = float(metrics[name])
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise RuntimeError(f"QA_METRIC_ERROR invalid {name}={value}")
    return metrics


def emit_metrics(
    task: str,
    surface: str,
    dataset: str,
    metrics: dict[str, float | int],
    *,
    feature_config_sha256: str,
    n_features: int,
    elapsed: float,
) -> None:
    spec = DATASET_MANIFEST.get(dataset)
    if spec is None:
        raise RuntimeError("QA_METRIC_ERROR unknown dataset at emit")
    counts = (int(metrics["n"]), int(metrics["n_ans"]), int(metrics["n_noans"]))
    expected = (spec["n"], spec["n_ans"], spec["n_noans"])
    if counts != expected or type(n_features) is not int or n_features < counts[0]:
        raise RuntimeError("QA_METRIC_ERROR incomplete evaluation proof")
    if not re.fullmatch(r"[0-9a-f]{64}", feature_config_sha256):
        raise RuntimeError("QA_METRIC_ERROR invalid feature configuration proof")
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("QA_METRIC_ERROR invalid elapsed time")
    print(
        f"QA_METRICS protocol={PROTOCOL} task={task} surface={surface} "
        f"dataset={dataset} dataset_sha256={spec['sha256']} "
        f"model_manifest_sha256={MODEL_MANIFEST_SHA256} "
        f"feature_config_sha256={feature_config_sha256} "
        f"f1={float(metrics['f1']):.6f} "
        f"em={float(metrics['em']):.6f} f1_ans={float(metrics['f1_ans']):.6f} "
        f"em_ans={float(metrics['em_ans']):.6f} "
        f"f1_noans={float(metrics['f1_noans']):.6f} "
        f"em_noans={float(metrics['em_noans']):.6f} n={counts[0]} "
        f"n_ans={counts[1]} n_noans={counts[2]} n_features={n_features} "
        f"elapsed={elapsed:.3f}",
        flush=True,
    )


def emit_completion(
    task: str,
    surface: str,
    dataset: str,
    *,
    feature_config_sha256: str,
    n_examples: int,
    n_features: int,
    n_predictions: int,
) -> None:
    spec = DATASET_MANIFEST.get(dataset)
    if spec is None:
        raise RuntimeError("QA_RUNTIME_ERROR unknown dataset at completion")
    if (
        n_examples != spec["n"]
        or n_predictions != spec["n"]
        or type(n_features) is not int
        or n_features < spec["n"]
        or not re.fullmatch(r"[0-9a-f]{64}", feature_config_sha256)
    ):
        raise RuntimeError("QA_RUNTIME_ERROR incomplete completion proof")
    print(
        f"QA_COMPLETE protocol={PROTOCOL} task={task} surface={surface} "
        f"dataset={dataset} dataset_sha256={spec['sha256']} "
        f"model_manifest_sha256={MODEL_MANIFEST_SHA256} "
        f"feature_config_sha256={feature_config_sha256} n={n_examples} "
        f"n_features={n_features} predictions={n_predictions} status=ok",
        flush=True,
    )


_TASK_SURFACES = {
    "casing": "build_casing",
    "encoding_order": "build_encoding_order",
    "max_answer_length": "build_max_answer_length",
    "max_seq_len": "build_max_seq_len",
    "n_best": "build_n_best",
    "null_threshold": "build_null_threshold",
    "question_inclusion": "build_question_mode",
    "span_decoding": "build_decoder",
    "doc_stride": "build_doc_stride",
    "span_aggregation": "build_span_aggregation",
}


def _prediction_parameters(task: str, value: Any) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "method": "constrained",
        "n_best": N_BEST,
        "max_answer_length": MAX_ANSWER_LENGTH,
        "question_first": True,
        "lowercase": False,
        "max_seq_len": MAX_SEQ_LEN,
        "doc_stride": DOC_STRIDE,
        "drop_question": False,
        "aggregation": "max_score",
        "null_threshold": None,
    }
    if task == "casing":
        parameters["lowercase"] = value == "lowercase"
    elif task == "encoding_order":
        parameters["question_first"] = value == "question_first"
    elif task == "max_answer_length":
        parameters["max_answer_length"] = value
    elif task == "max_seq_len":
        parameters["max_seq_len"] = value
        parameters["doc_stride"] = value // 3
    elif task == "n_best":
        parameters["n_best"] = value
    elif task == "null_threshold":
        parameters["null_threshold"] = value
    elif task == "question_inclusion":
        parameters["drop_question"] = value == "drop"
    elif task == "span_decoding":
        parameters["method"] = value
    elif task == "doc_stride":
        parameters["doc_stride"] = value
    elif task == "span_aggregation":
        parameters["aggregation"] = value
    else:
        raise RuntimeError(f"QA_RUNTIME_ERROR unknown task {task!r}")
    return parameters


def _feature_config_sha256(
    task: str,
    surface: str,
    dataset: str,
    seed: int,
    parameters: dict[str, Any],
) -> str:
    payload = {
        "protocol": PROTOCOL,
        "task": task,
        "surface": surface,
        "dataset": dataset,
        "seed": seed,
        "batch_size": BATCH_SIZE,
        "example_chunk_size": EXAMPLE_CHUNK_SIZE,
        "max_query_length": MAX_QUERY_LENGTH,
        "prediction_parameters": parameters,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _run(task: str, solution: str, dataset: str, seed: int) -> None:
    if task not in _TASK_SURFACES:
        raise RuntimeError(f"QA_RUNTIME_ERROR unknown task {task!r}")
    spec = DATASET_MANIFEST.get(dataset)
    if spec is None:
        raise RuntimeError(f"QA_DATA_ERROR unknown dataset {dataset!r}")
    if task == "null_threshold" and spec["kind"] != "squad2":
        raise RuntimeError("QA_DATA_ERROR null threshold requires SQuAD v2")
    if task != "null_threshold" and spec["kind"] != "mrqa":
        raise RuntimeError("QA_DATA_ERROR answerable task requires MRQA")

    started = time.monotonic()
    surface = _TASK_SURFACES[task]
    value = load_surface_value(solution, surface)
    parameters = _prediction_parameters(task, value)
    feature_config_sha256 = _feature_config_sha256(
        task, surface, dataset, seed, parameters
    )
    rows = load_dataset(dataset)
    device = setup(seed)
    model, tokenizer = load_model_and_tokenizer(device)
    print(
        f"QA_PROTOCOL protocol={PROTOCOL} task={task} surface={surface} "
        f"dataset={dataset} dataset_sha256={spec['sha256']} n={len(rows)} "
        f"n_ans={spec['n_ans']} n_noans={spec['n_noans']} "
        f"model_revision={MODEL_REVISION} model_files={len(MODEL_MANIFEST)} "
        f"model_manifest_sha256={MODEL_MANIFEST_SHA256} "
        f"model_weight_sha256={MODEL_WEIGHT_SHA256} "
        f"model_params={MODEL_PARAMETER_COUNT} seed={seed} device={device.type} "
        f"feature_config_sha256={feature_config_sha256}",
        flush=True,
    )
    predictions, n_features = predict(
        model,
        tokenizer,
        device,
        rows,
        **parameters,
    )
    metrics = score_squad(predictions, rows)
    emit_metrics(
        task,
        surface,
        dataset,
        metrics,
        feature_config_sha256=feature_config_sha256,
        n_features=n_features,
        elapsed=time.monotonic() - started,
    )
    emit_completion(
        task,
        surface,
        dataset,
        feature_config_sha256=feature_config_sha256,
        n_examples=len(rows),
        n_features=n_features,
        n_predictions=len(predictions),
    )


def cli(task: str) -> None:
    """CLI shared by verifier-owned task-specific entry points."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    try:
        _run(task, args.solution, args.dataset, args.seed)
    except Exception as exc:
        message = str(exc)
        if not message.startswith("QA_"):
            message = f"QA_RUNTIME_ERROR {type(exc).__name__}: {message}"
        print(message, file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
