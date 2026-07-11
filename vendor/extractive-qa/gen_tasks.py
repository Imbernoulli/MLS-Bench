#!/usr/bin/env python3
"""Generate the ten active full-scale extractive-QA task packages."""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "extractive-qa"
TASK_ROOT = ROOT / "tasks"

ANSWER_SETTINGS = [
    (
        "squad",
        "mrqa_squad_validation.jsonl",
        10_507,
        10_507,
        0,
        "64ab3a4c69574a258c934044a63605b15d98e1608fa9fb5b244868c5d0af89aa",
    ),
    (
        "newsqa",
        "mrqa_newsqa_validation.jsonl",
        4_212,
        4_212,
        0,
        "87b31cff3db4cb8276ddc58c94b03ca3ca500a72af95b8b9e2c63c9266ded7ad",
    ),
    (
        "hotpotqa",
        "mrqa_hotpotqa_validation.jsonl",
        5_901,
        5_901,
        0,
        "a335e1778d3c2de3a99b00e8eeaa3fc6e9b611386afadcc54532c2f33d3d95ad",
    ),
    (
        "naturalq",
        "mrqa_naturalquestions_validation.jsonl",
        12_836,
        12_836,
        0,
        "705717e225fc972d9a1df01737ab11d59a2c573a6ba9e7018b5ace4c34de6952",
    ),
]

SQUAD2_SETTINGS = [
    (
        "part0",
        "squad2_validation_part0.jsonl",
        3_958,
        1_988,
        1_970,
        "bdb7f256bf8893edef347623c6698a16320608d5ddf31c774de8e8234598f5b9",
    ),
    (
        "part1",
        "squad2_validation_part1.jsonl",
        3_958,
        1_956,
        2_002,
        "4159c7c652415873aa565af317a8c0d460164b5f80b185a35b9cbe6dac40f327",
    ),
    (
        "part2",
        "squad2_validation_part2.jsonl",
        3_957,
        1_984,
        1_973,
        "4b8fff6cb1dd3370416e1cf36cb7d8ba846ef61fd2cb086ccd02ad80a97ce651",
    ),
]
SQUAD2_FEATURE_COUNTS = {"part0": 4_054, "part1": 4_055, "part2": 4_056}

TASKS = {
    "qa-casing": {
        "solution": "casing.py",
        "harness": "harness_casing.py",
        "attribute": "build_casing",
        "native": "preserve",
        "schema": 'the string "preserve" or "lowercase"',
        "baselines": {"preserve": "preserve", "lowercase": "lowercase"},
        "objective": (
            "Measure how case normalization changes a cased extractive-QA "
            "checkpoint while preserving exact character offsets into the source context."
        ),
        "fixed": "Constrained span decoding and all windowing parameters are fixed.",
    },
    "qa-encoding-order": {
        "solution": "encoding_order.py",
        "harness": "harness_encoding_order.py",
        "attribute": "build_encoding_order",
        "native": "question_first",
        "schema": 'the string "question_first" or "context_first"',
        "baselines": {
            "question_first": "question_first",
            "context_first": "context_first",
        },
        "objective": (
            "Study whether the question or context is encoded first in the paired "
            "sequence presented to a frozen extractive-QA checkpoint."
        ),
        "fixed": "The context side is always the overflowing side; decoding is fixed.",
    },
    "qa-max-answer-length": {
        "solution": "max_answer_length.py",
        "harness": "harness_max_answer_length.py",
        "attribute": "build_max_answer_length",
        "native": 30,
        "schema": "an integer from 1 through 200 inclusive",
        "baselines": {"length_30": 30, "length_1": 1, "length_200": 200},
        "objective": (
            "Study the valid-span length constraint used during standard top-k "
            "extractive answer decoding."
        ),
        "fixed": "The n-best depth, model input length, and window aggregation are fixed.",
    },
    "qa-max-seq-len": {
        "solution": "max_seq_len.py",
        "harness": "harness_max_seq_len.py",
        "attribute": "build_max_seq_len",
        "native": 384,
        "schema": "a multiple of 64 from 128 through 512 inclusive",
        "baselines": {"length_384": 384, "length_128": 128, "length_512": 512},
        "objective": (
            "Study the model input window length under complete sliding-window "
            "coverage of long contexts."
        ),
        "fixed": (
            "The overlap is deterministically one third of the selected window; "
            "span decoding and cross-window aggregation are fixed."
        ),
    },
    "qa-n-best": {
        "solution": "n_best.py",
        "harness": "harness_n_best.py",
        "attribute": "build_n_best",
        "native": 20,
        "schema": "an integer from 1 through 50 inclusive",
        "baselines": {"depth_20": 20, "depth_1": 1, "depth_50": 50},
        "objective": (
            "Study how many top start and end positions are searched when forming "
            "a valid extractive answer span."
        ),
        "fixed": "The answer-length cap, windowing, and score aggregation are fixed.",
    },
    "qa-null-threshold": {
        "solution": "null_threshold.py",
        "harness": "harness_null_threshold.py",
        "attribute": "build_null_threshold",
        "native": 0.0,
        "schema": "a finite integer or float from -30 through 30 inclusive",
        "baselines": {"threshold_0": 0.0, "threshold_neg30": -30.0, "threshold_30": 30.0},
        "objective": (
            "Study the no-answer decision boundary applied to the margin between "
            "the CLS null score and the best valid non-null span."
        ),
        "fixed": "Constrained span search and minimum-null cross-window handling are fixed.",
        "squad2": True,
    },
    "qa-question-inclusion": {
        "solution": "question_inclusion.py",
        "harness": "harness_question_inclusion.py",
        "attribute": "build_question_mode",
        "native": "real",
        "schema": 'the string "real" or "drop"',
        "baselines": {"real": "real", "drop": "drop"},
        "objective": (
            "Measure the combined effect of question removal by comparing the real "
            "question with the fixed one-token placeholder `what`."
        ),
        "fixed": (
            "This intervention removes question semantics but also reallocates input "
            "tokens to the context, so it is not presented as a pure conditioning-only effect."
        ),
    },
    "qa-span-decoding": {
        "solution": "span_decoding.py",
        "harness": "harness_span_decoding.py",
        "attribute": "build_decoder",
        "native": "constrained",
        "schema": 'the string "argmax" or "constrained"',
        "baselines": {"constrained": "constrained", "argmax": "argmax"},
        "objective": (
            "Compare independent start/end argmax with top-k search restricted to "
            "ordered spans under a fixed answer-length constraint."
        ),
        "fixed": (
            "Both decoders use the same answer-length cap, windowing, and aggregation; "
            "invalid independent argmax pairs produce an empty span."
        ),
    },
    "qa-doc-stride": {
        "solution": "doc_stride.py",
        "harness": "harness_doc_stride.py",
        "attribute": "build_doc_stride",
        "native": 128,
        "schema": "the integer 0, 64, or 128",
        "baselines": {"stride_128": 128, "stride_0": 0, "stride_64": 64},
        "objective": (
            "Study overlap between consecutive context windows when long documents "
            "are covered without dropping any official examples."
        ),
        "fixed": "Input length, constrained span decoding, and aggregation are fixed.",
    },
    "qa-span-aggregation": {
        "solution": "span_aggregation.py",
        "harness": "harness_span_aggregation.py",
        "attribute": "build_span_aggregation",
        "native": "max_score",
        "schema": 'the string "first_feature" or "max_score"',
        "baselines": {"max_score": "max_score", "first_feature": "first_feature"},
        "objective": (
            "Compare complete max-score aggregation across all overflow windows with "
            "a first-window-only coverage ablation."
        ),
        "fixed": (
            "The `first_feature` arm intentionally ignores later windows and therefore "
            "changes long-context coverage; it is not described as complete aggregation."
        ),
    },
}

MODEL_SHA = "ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7"
MODEL_REVISION = "adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"
MODEL_MANIFEST = {
    "config.json": "64fa58495a722d57609c22f199824bfe98c19be068136a70c268214a08cb8060",
    "model.safetensors": MODEL_SHA,
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "vocab.json": "06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab",
    "tokenizer_config.json": "7a33226d4265e3989cc6341666af179d0cc710136f4059aae0dd8c0797cba556",
    "special_tokens_map.json": "c611b1f7d416eb001ee4f293d903ea8c88e703463f1d403f1866a0352743fd00",
}
MODEL_MANIFEST_SHA = hashlib.sha256(
    json.dumps(MODEL_MANIFEST, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
MODEL_PARAMETER_COUNT = 124_056_578
CALIBRATION_TIMESTAMP = "2026-07-11T01:54:18+00:00"
CALIBRATION_STATUS = {
    "qa-null-threshold": "squad2_fullscale_measured_static_proof_hardened_20260711",
    "mrqa": "official_f1_baseline_free_representative_runtime_static_validated_20260711",
}
CALIBRATION_RESULTS = {
    "threshold_neg30": {
        "f1_part0": 49.823143,
        "f1_part1": 50.581102,
        "f1_part2": 49.886277,
    },
    "threshold_0": {
        "f1_part0": 82.973053,
        "f1_part1": 82.861592,
        "f1_part2": 82.904341,
    },
    "threshold_30": {
        "f1_part0": 46.048585,
        "f1_part1": 45.051846,
        "f1_part2": 45.761413,
    },
}
CALIBRATION_PROVENANCE = {
    "protocol": "qa-official-full-v2",
    "evidence_scope": "qa-null-threshold SQuAD2 only",
    "representative_task": "qa-null-threshold",
    "mangrove_task_id": 96029,
    "run_id": "k1h20-roberta9-v7-4gpu",
    "anchor_root": "/tmp/qa-k1-v7-artifacts",
    "image": (
        "msai-cn-beijing.cr.volces.com/public/bohanlyu2022/"
        "mlsbench-harbor-extractive-qa@"
        "sha256:3bf1e39f4004791522670ee57d4aaa84ce040135bee896ff8dec16124f4a046e"
    ),
    "runner_sha256": "11fc5f1584d089589da4fe00d3ca45bc4de16cd97799e41f6e0237c0a08e354b",
    "verifier_common_sha256": "caa8546c1f589ce58f48157c705f5fa53abb8a0e0b6b2e51e22150b55883b30a",
    "verifier_harness_sha256": "81485497d70010b88cf4894590e119dfcbc38c7664a91415283519212d0b9f8d",
    "zone": "k1h20",
    "allocation_gpus": 4,
    "execution": "nine independent single-GPU cells in 4-wide waves",
    "completion_contract": (
        "All preserved cells have rc=0. QA_COMPLETE and terminal parsing were added "
        "statically; the measured workload, model, data, and metrics are unchanged."
    ),
}

MRQA_REPRESENTATIVE_PROVENANCE = {
    "protocol": "qa-official-full-v2",
    "status": "baseline_free_official_f1_with_representative_runtime",
    "measured_task": None,
    "scoring_calibration": "not required for the fixed raw F1 [0,100] to [0,1] mapping",
    "representative_model_runtime_evidence": {
        "task": "qa-null-threshold",
        "mangrove_task_id": 96029,
        "dataset_family": "SQuAD2",
        "image": CALIBRATION_PROVENANCE["image"],
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": MODEL_SHA,
    },
    "limitation": (
        "The representative SQuAD2 run validates only the shared frozen model and "
        "runtime path. MRQA task surfaces and proof bindings are validated statically; "
        "their per-surface feature counts and runtimes are not measured anchors."
    ),
}


def literal(value) -> str:
    return repr(value)


def solution_source(task: dict, value) -> str:
    stem = task["solution"].removesuffix(".py").replace("_", " ")
    return (
        f'"""Agent-editable literal surface for {stem}.\n\n'
        f"Return exactly one supported literal.\n"
        f'"""\n'
        f"def {task['attribute']}():\n"
        f"    return {literal(value)}\n"
    )


def description(task_name: str, task: dict, settings: list[tuple]) -> str:
    if task.get("squad2"):
        data_text = (
            "The verifier evaluates the complete official SQuAD v2 validation split. "
            "It is partitioned exhaustively by original row index across three "
            "configured commands: the parts are disjoint, their union is the full "
            "split, and every configured command contributes to the task score."
        )
    else:
        data_text = (
            "The verifier evaluates four complete answerable validation domains from "
            "the official MRQA unified validation data. Every source example is loaded "
            "and must produce exactly one prediction; task-specific window-use "
            "interventions are disclosed above. Every configured command contributes "
            "to the task score."
        )
    score_text = (
        "Each per-command official F1 maps directly from 0 to score 0 and from 100 "
        "to score 1; the task score is their geometric mean. This fixed, baseline-free "
        "mapping does not substitute a representative result for the current run."
    )
    return f"""# {task_name}

## Research objective

{task['objective']} {task['fixed']}

## Editable contract

Edit only `extractive-qa/solution/{task['solution']}`. The module must contain
exactly one zero-argument function named `{task['attribute']}` whose body is one
literal `return`. The accepted value is {task['schema']}. Imports, decorators,
annotations, computations, additional statements, and additional functions are
invalid; the verifier parses this surface with a restricted AST and never executes
agent-authored Python.

## Evaluation protocol

The frozen `deepset/roberta-base-squad2` checkpoint and all tokenizer files are
pinned at revision `{MODEL_REVISION}` and verified against a complete six-file
SHA-256 manifest before inference. The full checkpoint contains 124,056,578 model
parameters. Its documented upstream recipe
trained RoBERTa-base on SQuAD 2.0 for two epochs with batch size 96. Questions are capped
at 64 tokens, long contexts use complete overflow-window coverage, and each example
must produce exactly one prediction. {data_text}

The primary metric is official SQuAD token-overlap F1 on a 0-100 scale. Exact
match is reported as a secondary metric. {score_text} Dataset loading, model loading, CUDA
execution, feature construction, inference, prediction completeness, metric
calculation, terminal completion proof emission, or output parsing failure yields no metric and
therefore a score of exactly zero; there is no fallback metric or default score.
"""


def parser_source(task_name: str, task: dict, settings: list[tuple]) -> str:
    expected = {
        label: {
            "dataset": filename,
            "sha256": sha256,
            "n": n,
            "n_ans": n_ans,
            "n_noans": n_noans,
            **(
                {"n_features": SQUAD2_FEATURE_COUNTS[label]}
                if label in SQUAD2_FEATURE_COUNTS
                else {}
            ),
        }
        for label, filename, n, n_ans, n_noans, sha256 in settings
    }
    expected_task = task_name.removeprefix("qa-").replace("-", "_")
    expected_surface = task["attribute"]
    return f'''"""Strict fail-closed parser for full-scale extractive QA."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


_FLOAT = r"[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?"
_PROTOCOL = re.compile(
    r"QA_PROTOCOL protocol=qa-official-full-v2 task=([a-z_]+) "
    r"surface=([a-z_]+) dataset=([a-z0-9_.-]+) "
    r"dataset_sha256=([0-9a-f]{{64}}) n=(\\d+) n_ans=(\\d+) n_noans=(\\d+) "
    r"model_revision=([0-9a-f]{{40}}) model_files=(\\d+) "
    r"model_manifest_sha256=([0-9a-f]{{64}}) "
    r"model_weight_sha256=([0-9a-f]{{64}}) model_params=(\\d+) "
    r"seed=(\\d+) device=cuda feature_config_sha256=([0-9a-f]{{64}})"
)
_METRIC = re.compile(
    rf"QA_METRICS protocol=qa-official-full-v2 task=([a-z_]+) "
    rf"surface=([a-z_]+) dataset=([a-z0-9_.-]+) "
    rf"dataset_sha256=([0-9a-f]{{{{64}}}}) "
    rf"model_manifest_sha256=([0-9a-f]{{{{64}}}}) "
    rf"feature_config_sha256=([0-9a-f]{{{{64}}}}) "
    rf"f1=({{_FLOAT}}) em=({{_FLOAT}}) f1_ans=({{_FLOAT}}) "
    rf"em_ans=({{_FLOAT}}) f1_noans=({{_FLOAT}}) em_noans=({{_FLOAT}}) "
    rf"n=(\\d+) n_ans=(\\d+) n_noans=(\\d+) n_features=(\\d+) "
    rf"elapsed=({{_FLOAT}})"
)
_COMPLETE = re.compile(
    r"QA_COMPLETE protocol=qa-official-full-v2 task=([a-z_]+) "
    r"surface=([a-z_]+) dataset=([a-z0-9_.-]+) "
    r"dataset_sha256=([0-9a-f]{{64}}) "
    r"model_manifest_sha256=([0-9a-f]{{64}}) "
    r"feature_config_sha256=([0-9a-f]{{64}}) n=(\\d+) "
    r"n_features=(\\d+) predictions=(\\d+) status=ok"
)
_EXPECTED_TASK = {expected_task!r}
_EXPECTED_SURFACE = {expected_surface!r}
_MODEL_REVISION = {MODEL_REVISION!r}
_MODEL_SHA256 = {MODEL_SHA!r}
_MODEL_MANIFEST_SHA256 = {MODEL_MANIFEST_SHA!r}
_MODEL_FILE_COUNT = {len(MODEL_MANIFEST)}
_MODEL_PARAMETER_COUNT = {MODEL_PARAMETER_COUNT}
_EXPECTED = {expected!r}
_FAILURE = re.compile(
    r"Traceback \\(most recent call last\\)|QA_[A-Z_]*ERROR|"
    r"CUDA out of memory|COMMAND FAILED|TIMEOUT|OUT_OF_MEMORY|CANCELLED|"
    r"NODE_FAIL|SEGMENTATION FAULT|VERIFICATION FAILED|PROCESS EXITED|"
    r"NON[- ]ZERO EXIT|COMMAND EXITED WITH CODE [1-9]|\\bKILLED\\b",
    re.IGNORECASE,
)


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label not in _EXPECTED or _FAILURE.search(raw_output):
            return ParseResult(feedback=raw_output[-3000:], metrics={{}})
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        protocol_prefixes = sum(line.startswith("QA_PROTOCOL") for line in lines)
        metric_prefixes = sum(line.startswith("QA_METRICS") for line in lines)
        completion_prefixes = sum(line.startswith("QA_COMPLETE") for line in lines)
        protocols = [(index, match) for index, line in enumerate(lines)
                     if (match := _PROTOCOL.fullmatch(line)) is not None]
        metrics = [(index, match) for index, line in enumerate(lines)
                   if (match := _METRIC.fullmatch(line)) is not None]
        completions = [(index, match) for index, line in enumerate(lines)
                       if (match := _COMPLETE.fullmatch(line)) is not None]
        if (
            protocol_prefixes != 1
            or metric_prefixes != 1
            or completion_prefixes != 1
            or len(protocols) != 1
            or len(metrics) != 1
            or len(completions) != 1
            or not (protocols[0][0] < metrics[0][0] < completions[0][0])
            or completions[0][0] != len(lines) - 1
        ):
            return ParseResult(feedback=raw_output[-3000:], metrics={{}})

        protocol = protocols[0][1]
        match = metrics[0][1]
        completion = completions[0][1]
        expected = _EXPECTED[cmd_label]
        values = [float(match.group(index)) for index in range(7, 13)]
        counts = tuple(int(match.group(index)) for index in range(13, 16))
        n_features = int(match.group(16))
        elapsed = float(match.group(17))
        complete_n = int(completion.group(7))
        complete_features = int(completion.group(8))
        complete_predictions = int(completion.group(9))
        f1, em, f1_ans, em_ans, f1_noans, em_noans = values
        weighted_f1 = (
            f1_ans * expected["n_ans"] + f1_noans * expected["n_noans"]
        ) / expected["n"]
        weighted_em = (
            em_ans * expected["n_ans"] + em_noans * expected["n_noans"]
        ) / expected["n"]
        if (
            protocol.group(1) != _EXPECTED_TASK
            or protocol.group(2) != _EXPECTED_SURFACE
            or protocol.group(3) != expected["dataset"]
            or protocol.group(4) != expected["sha256"]
            or tuple(int(protocol.group(index)) for index in range(5, 8))
            != (expected["n"], expected["n_ans"], expected["n_noans"])
            or protocol.group(8) != _MODEL_REVISION
            or int(protocol.group(9)) != _MODEL_FILE_COUNT
            or protocol.group(10) != _MODEL_MANIFEST_SHA256
            or protocol.group(11) != _MODEL_SHA256
            or int(protocol.group(12)) != _MODEL_PARAMETER_COUNT
            or int(protocol.group(13)) != 42
            or match.group(1) != _EXPECTED_TASK
            or match.group(2) != _EXPECTED_SURFACE
            or match.group(3) != expected["dataset"]
            or match.group(4) != expected["sha256"]
            or match.group(5) != _MODEL_MANIFEST_SHA256
            or match.group(6) != protocol.group(14)
            or counts != (expected["n"], expected["n_ans"], expected["n_noans"])
            or n_features < expected["n"]
            or (
                expected.get("n_features") is not None
                and n_features != expected["n_features"]
            )
            or any(not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in values)
            or not math.isfinite(elapsed)
            or elapsed <= 0.0
            or em > f1 + 1e-6
            or em_ans > f1_ans + 1e-6
            or em_noans > f1_noans + 1e-6
            or abs(f1 - weighted_f1) > 2e-6
            or abs(em - weighted_em) > 2e-6
            or completion.group(1) != _EXPECTED_TASK
            or completion.group(2) != _EXPECTED_SURFACE
            or completion.group(3) != expected["dataset"]
            or completion.group(4) != expected["sha256"]
            or completion.group(5) != _MODEL_MANIFEST_SHA256
            or completion.group(6) != protocol.group(14)
            or complete_n != expected["n"]
            or complete_features != n_features
            or complete_predictions != expected["n"]
        ):
            return ParseResult(feedback=raw_output[-3000:], metrics={{}})

        feedback = (
            f"Results ({{cmd_label}}): F1={{f1:.6f}}, EM={{em:.6f}}, "
            f"examples={{counts[0]}}, features={{n_features}}"
        )
        return ParseResult(
            feedback=feedback,
            metrics={{f"f1_{{cmd_label}}": f1, f"em_{{cmd_label}}": em}},
        )
'''


def score_spec_source(task_name: str, settings: list[tuple]) -> str:
    lines = [
        '"""Direct official-F1 scoring; requires the integrated floor-aware scorer."""',
        "from mlsbench.scoring.dsl import *",
        "",
    ]
    for label, *_ in settings:
        term_lines = [
            f'term("f1_{label}",',
            f'    col("f1_{label}").higher().id()',
        ]
        term_lines.extend(
            [
                "    .bounded_power(bound=100.0, floor=const(0.0),",
                "                   ref=const(50.0), ref_score=0.5))",
            ]
        )
        lines.extend((*term_lines, ""))
    for label, *_ in settings:
        lines.append(
            f'setting("{label}", weighted_mean(("f1_{label}", 1.0)))'
        )
    labels = ", ".join(repr(item[0]) for item in settings)
    lines.extend(["", f"task(gmean({labels}))", ""])
    return "\n".join(lines)


def edit_source(task: dict, value, name: str) -> str:
    file_path = f"extractive-qa/solution/{task['solution']}"
    content = f"def {task['attribute']}():\n    return {literal(value)}"
    return f'''"""Reference literal surface {name}."""

_FILE = {file_path!r}
_CONTENT = {content!r}

OPS = [
    {{"op": "replace", "file": _FILE, "start_line": 5, "end_line": 6,
     "content": _CONTENT}},
]
'''


def all_package_files() -> set[str]:
    result: set[str] = set()
    for path in VENDOR.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            result.add(f"extractive-qa/{path.relative_to(VENDOR).as_posix()}")
    return result


def write_vendor_baselines() -> None:
    root = VENDOR / "baselines"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for task_name, task in TASKS.items():
        for baseline_name, value in task["baselines"].items():
            filename = f"{task_name.removeprefix('qa-').replace('-', '_')}__{baseline_name}.py"
            (root / filename).write_text(solution_source(task, value), encoding="utf-8")


def write_task(task_name: str, task: dict, package_files: set[str]) -> None:
    settings = SQUAD2_SETTINGS if task.get("squad2") else ANSWER_SETTINGS
    task_dir = TASK_ROOT / task_name
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "scripts").mkdir(parents=True)
    (task_dir / "edits").mkdir(parents=True)

    test_cmds = []
    for group, (label, filename, *_rest) in enumerate(settings, 1):
        script = task_dir / "scripts" / f"{label}.sh"
        script.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "export QA_DATA=\"${MLSBENCH_VERIFIER_DATA_ROOT:?}/extractive-qa/data\"\n"
            f"python -u {task['harness']} "
            f"--solution solution/{task['solution']} "
            f"--dataset {filename} --seed \"${{SEED:-42}}\"\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        test_cmds.append(
            {
                "cmd": f"scripts/{label}.sh",
                "label": label,
                "group": group,
                "compute": 1.0,
                "time": "2:00:00",
                "mem": 32,
                "package": "extractive-qa",
            }
        )

    baselines = {}
    for name, value in task["baselines"].items():
        edit_path = task_dir / "edits" / f"{name}.edit.py"
        edit_path.write_text(edit_source(task, value, name), encoding="utf-8")
        baselines[name] = {"edit_ops": f"edits/{name}.edit.py"}

    active_solution = f"extractive-qa/solution/{task['solution']}"
    active_harness = f"extractive-qa/{task['harness']}"
    protected = {"extractive-qa/common.py", active_harness}
    visible = {active_solution, "extractive-qa/__init__.py"}
    pruned = sorted(package_files - protected - visible)
    config = {
        "allow_create": False,
        "rigorous_codebase": True,
        "_verifier_serial": True,
        "protocol": "qa-official-full-v2",
        "integration_requires": [
            "d3ad006b",
            "f714d7d5",
            "c6fa7456",
            "830403af",
            "dd0c8df5",
        ],
        "calibration_status": (
            CALIBRATION_STATUS["qa-null-threshold"]
            if task_name == "qa-null-threshold"
            else CALIBRATION_STATUS["mrqa"]
        ),
        "scoring_status": {
            "mode": "baseline_free_direct_official_f1",
            "integration_requires": ["f714d7d5", "c6fa7456"],
            "reason": "explicit zero floor and true geometric mean",
        },
        "agent_data_prune": ["/data/extractive-qa/data"],
        "verifier_data_deps": [
            {
                "name": f"extractive_qa_gold_{label}",
                "host_path": f"{{data_root}}/extractive-qa/data/{filename}",
                "dest": f"data/extractive-qa/data/{filename}",
            }
            for label, filename, *_rest in settings
        ],
        "seeds": [42],
        "test_cmds": test_cmds,
        "baselines": baselines,
        "verifier_only_package_files": ["extractive-qa/common.py", active_harness],
        "agent_pruned_package_files": pruned,
        "files": [
            {
                "filename": active_solution,
                "read": [{"start": -1, "end": -1}],
                "edit": [{"start": 5, "end": 6}],
            }
        ],
        "calibration_provenance": (
            CALIBRATION_PROVENANCE
            if task_name == "qa-null-threshold"
            else MRQA_REPRESENTATIVE_PROVENANCE
        ),
    }
    (task_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (task_dir / "task_description.md").write_text(
        description(task_name, task, settings), encoding="utf-8"
    )
    (task_dir / "parser.py").write_text(
        parser_source(task_name, task, settings), encoding="utf-8"
    )
    (task_dir / "score_spec.py").write_text(
        score_spec_source(task_name, settings), encoding="utf-8"
    )
    labels = [f"f1_{item[0]}" for item in settings]
    leaderboard_lines = [
        ",".join(("timestamp", "model", "is_final", "seed", *labels))
    ]
    if task_name == "qa-null-threshold":
        for baseline_name, metrics in CALIBRATION_RESULTS.items():
            leaderboard_lines.append(
                ",".join(
                    (
                        CALIBRATION_TIMESTAMP,
                        f"baseline:{baseline_name}",
                        "true",
                        "mean",
                        *(str(metrics[label]) for label in labels),
                    )
                )
            )
    (task_dir / "leaderboard.csv").write_text(
        "\n".join(leaderboard_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    for dropped in ("qa-answer-policy", "qa-null-confidence"):
        path = TASK_ROOT / dropped
        if path.exists():
            shutil.rmtree(path)
    write_vendor_baselines()
    package_files = all_package_files()
    for task_name, task in TASKS.items():
        native_path = VENDOR / "solution" / task["solution"]
        native_path.write_text(solution_source(task, task["native"]), encoding="utf-8")
    package_files = all_package_files()
    for task_name, task in TASKS.items():
        write_task(task_name, task, package_files)
    print(f"generated {len(TASKS)} qa task packages")


if __name__ == "__main__":
    main()
