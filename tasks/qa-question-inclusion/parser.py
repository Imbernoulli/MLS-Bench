"""Strict fail-closed parser for full-scale extractive QA."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PROTOCOL = re.compile(
    r"QA_PROTOCOL protocol=qa-official-full-v2 task=([a-z_]+) "
    r"surface=([a-z_]+) dataset=([a-z0-9_.-]+) "
    r"dataset_sha256=([0-9a-f]{64}) n=(\d+) n_ans=(\d+) n_noans=(\d+) "
    r"model_revision=([0-9a-f]{40}) model_files=(\d+) "
    r"model_manifest_sha256=([0-9a-f]{64}) "
    r"model_weight_sha256=([0-9a-f]{64}) model_params=(\d+) "
    r"seed=(\d+) device=cuda feature_config_sha256=([0-9a-f]{64})"
)
_METRIC = re.compile(
    rf"QA_METRICS protocol=qa-official-full-v2 task=([a-z_]+) "
    rf"surface=([a-z_]+) dataset=([a-z0-9_.-]+) "
    rf"dataset_sha256=([0-9a-f]{{64}}) "
    rf"model_manifest_sha256=([0-9a-f]{{64}}) "
    rf"feature_config_sha256=([0-9a-f]{{64}}) "
    rf"f1=({_FLOAT}) em=({_FLOAT}) f1_ans=({_FLOAT}) "
    rf"em_ans=({_FLOAT}) f1_noans=({_FLOAT}) em_noans=({_FLOAT}) "
    rf"n=(\d+) n_ans=(\d+) n_noans=(\d+) n_features=(\d+) "
    rf"elapsed=({_FLOAT})"
)
_COMPLETE = re.compile(
    r"QA_COMPLETE protocol=qa-official-full-v2 task=([a-z_]+) "
    r"surface=([a-z_]+) dataset=([a-z0-9_.-]+) "
    r"dataset_sha256=([0-9a-f]{64}) "
    r"model_manifest_sha256=([0-9a-f]{64}) "
    r"feature_config_sha256=([0-9a-f]{64}) n=(\d+) "
    r"n_features=(\d+) predictions=(\d+) status=ok"
)
_EXPECTED_TASK = 'question_inclusion'
_EXPECTED_SURFACE = 'build_question_mode'
_MODEL_REVISION = 'adc3b06f79f797d1c575d5479d6f5efe54a9e3b4'
_MODEL_SHA256 = 'ac5db66fdcfecb400345d09787b71009d60805ef9883451071669cf951b5e2c7'
_MODEL_MANIFEST_SHA256 = '46889e840eb932d8f002d89d61aebd1807c2ab8f79d6b753f17fee1e39af90a5'
_MODEL_FILE_COUNT = 6
_MODEL_PARAMETER_COUNT = 124056578
_EXPECTED = {'squad': {'dataset': 'mrqa_squad_validation.jsonl', 'sha256': '64ab3a4c69574a258c934044a63605b15d98e1608fa9fb5b244868c5d0af89aa', 'n': 10507, 'n_ans': 10507, 'n_noans': 0}, 'newsqa': {'dataset': 'mrqa_newsqa_validation.jsonl', 'sha256': '87b31cff3db4cb8276ddc58c94b03ca3ca500a72af95b8b9e2c63c9266ded7ad', 'n': 4212, 'n_ans': 4212, 'n_noans': 0}, 'hotpotqa': {'dataset': 'mrqa_hotpotqa_validation.jsonl', 'sha256': 'a335e1778d3c2de3a99b00e8eeaa3fc6e9b611386afadcc54532c2f33d3d95ad', 'n': 5901, 'n_ans': 5901, 'n_noans': 0}, 'naturalq': {'dataset': 'mrqa_naturalquestions_validation.jsonl', 'sha256': '705717e225fc972d9a1df01737ab11d59a2c573a6ba9e7018b5ace4c34de6952', 'n': 12836, 'n_ans': 12836, 'n_noans': 0}}
_FAILURE = re.compile(
    r"Traceback \(most recent call last\)|QA_[A-Z_]*ERROR|"
    r"CUDA out of memory|COMMAND FAILED|TIMEOUT|OUT_OF_MEMORY|CANCELLED|"
    r"NODE_FAIL|SEGMENTATION FAULT|VERIFICATION FAILED|PROCESS EXITED|"
    r"NON[- ]ZERO EXIT|COMMAND EXITED WITH CODE [1-9]|\bKILLED\b",
    re.IGNORECASE,
)


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label not in _EXPECTED or _FAILURE.search(raw_output):
            return ParseResult(feedback=raw_output[-3000:], metrics={})
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
            return ParseResult(feedback=raw_output[-3000:], metrics={})

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
            return ParseResult(feedback=raw_output[-3000:], metrics={})

        feedback = (
            f"Results ({cmd_label}): F1={f1:.6f}, EM={em:.6f}, "
            f"examples={counts[0]}, features={n_features}"
        )
        return ParseResult(
            feedback=feedback,
            metrics={f"f1_{cmd_label}": f1, f"em_{cmd_label}": em},
        )
