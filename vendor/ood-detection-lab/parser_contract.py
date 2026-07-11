"""Verifier-only strict output parser factory for full-image OOD siblings."""
from __future__ import annotations

import math
import re

from mlsbench.agent.parsers import OutputParser, ParseResult


PROTOCOL = "openood_cifar10_resnet18_full_v1"
EXPECTED_DATA_SHA256 = "796799c9a1c073784b02a3f42a00d0fc8b902387b1f3345b5b3d1f2631e0722d"
EXPECTED_MODEL_SHA256 = "8859e0ff484ab029fcd5bd0f85052c5679d82b8e36a7c066e922e2fdff62b7dc"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PROTOCOL_LINE = re.compile(
    rf"OOD_PROTOCOL protocol={PROTOCOL} task=(\S+) "
    rf"model=openood_resnet18_32x32 batch_size=128 seed=42 status=ok"
)
METRIC_LINE = re.compile(
    rf"OOD_METRICS protocol={PROTOCOL} task=(\S+) setting=(\S+) ood=(\S+) "
    rf"auroc=({NUMBER}) fpr95=({NUMBER}) id_acc=({NUMBER}) "
    rf"n_fit=(\d+) n_id=(\d+) n_ood=(\d+) base_ood_batches=(\d+) "
    rf"id_score_batches=(\d+) ood_score_batches=(\d+) "
    rf"inference_seconds=({NUMBER}) status=ok"
)
COMPLETE_LINE = re.compile(
    rf"OOD_COMPLETE protocol={PROTOCOL} task=(\S+) data_sha256=([0-9a-f]{{64}}) "
    rf"checkpoint_sha256=([0-9a-f]{{64}}) n_fit=(\d+) n_id=(\d+) "
    rf"n_svhn=(\d+) n_cifar100=(\d+) n_tin=(\d+) "
    rf"base_forward_images=(\d+) base_forward_batches=(\d+) "
    rf"task_forward_images=(\d+) task_forward_batches=(\d+) status=ok"
)
FAILURE = re.compile(
    r"(?:Traceback \(most recent call last\)|\[COMMAND FAILED|"
    r"\b(?:OOD|VERIFIER)_FAILURE\b|\bTIMEOUT\b|CUDA out of memory|"
    r"\bKilled\b|Segmentation fault|\b(?:verification|evaluation) failed\b|"
    r"\bbudget check failed\b|\bstatus\s*[:=]\s*failed\b|"
    r"\bexit(?:ed)?(?: with)? code\b|\bprocess exited\b|\bcommand failed\b|"
    r"\btimed? out\b|\btime limit\b|\bout of memory\b|\bOOM\b|"
    r"\bcancel(?:led|ed)?\b|\bnode[_ -]?fail(?:ure|ed)?\b|"
    r"\bnon[- ]finite\b|_FALLBACK\b|\b(?:nan|inf|infinity)\b)",
    re.IGNORECASE,
)
AUTHORITATIVE_PREFIX = re.compile(
    r"^OOD_(?:PROTOCOL|METRICS|COMPLETE)(?:\b|_)", re.IGNORECASE,
)


def _setting_name(task: str, ood_name: str) -> str:
    if task == "ood-near-far":
        regime = {"svhn": "far", "cifar100": "near", "tin": "medium"}[ood_name]
        return f"ood_{regime}_{ood_name}_full"
    slug = task.removeprefix("ood-").replace("-", "_")
    return f"ood_{slug}_{ood_name}_full"


def build_parser(task: str):
    if not task.startswith("ood-") or task == "ood-logit-score":
        raise ValueError(f"unsupported shared OOD parser task: {task}")
    slug = task.removeprefix("ood-").replace("-", "_")
    expected_label = f"ood_{slug}_full_protocol"
    expected = {
        _setting_name(task, "svhn"): ("svhn", 26_032, 204, 408),
        _setting_name(task, "cifar100"): ("cifar100", 10_000, 79, 158),
        _setting_name(task, "tin"): ("tin", 10_000, 79, 158),
    }
    is_input_preproc = task == "ood-input-preproc"
    expected_task_counts = (218_096, 1_714) if is_input_preproc else (106_032, 832)

    class StrictOODParser(OutputParser):
        def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
            if cmd_label != expected_label:
                return ParseResult(feedback="full OOD command label is wrong", metrics={})
            if FAILURE.search(raw_output):
                return ParseResult(feedback="full OOD verifier reported a failure", metrics={})

            lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
            authoritative_rows = [
                (index, line) for index, line in enumerate(lines)
                if AUTHORITATIVE_PREFIX.match(line)
            ]
            protocol_rows = [
                (index, line) for index, line in enumerate(lines)
                if line.startswith("OOD_PROTOCOL")
            ]
            metric_rows = [
                (index, line) for index, line in enumerate(lines)
                if line.startswith("OOD_METRICS")
            ]
            completion_rows = [
                (index, line) for index, line in enumerate(lines)
                if line.startswith("OOD_COMPLETE")
            ]
            if (
                len(protocol_rows) != 1
                or len(metric_rows) != 3
                or len(completion_rows) != 1
                or len(authoritative_rows) != 5
                or not lines
                or lines[-1] != completion_rows[0][1]
                or protocol_rows[0][0] >= metric_rows[0][0]
                or any(row[0] >= completion_rows[0][0] for row in metric_rows)
            ):
                return ParseResult(
                    feedback="full OOD proof is missing, duplicated, reordered, or non-terminal",
                    metrics={},
                )

            protocol_match = PROTOCOL_LINE.fullmatch(protocol_rows[0][1])
            completion_match = COMPLETE_LINE.fullmatch(completion_rows[0][1])
            if protocol_match is None or protocol_match.group(1) != task:
                return ParseResult(feedback="full OOD protocol proof is malformed", metrics={})
            if completion_match is None:
                return ParseResult(feedback="full OOD completion proof is malformed", metrics={})
            (
                completion_task,
                data_sha,
                model_sha,
                raw_fit,
                raw_id,
                raw_svhn,
                raw_cifar100,
                raw_tin,
                raw_base_images,
                raw_base_batches,
                raw_task_images,
                raw_task_batches,
            ) = completion_match.groups()
            if completion_task != task:
                return ParseResult(feedback="full OOD completion belongs to another task", metrics={})
            if data_sha != EXPECTED_DATA_SHA256 or model_sha != EXPECTED_MODEL_SHA256:
                return ParseResult(feedback="full OOD completion uses an unpinned artifact", metrics={})
            counts = tuple(map(int, (
                raw_fit,
                raw_id,
                raw_svhn,
                raw_cifar100,
                raw_tin,
                raw_base_images,
                raw_base_batches,
                raw_task_images,
                raw_task_batches,
            )))
            if counts != (
                50_000,
                10_000,
                26_032,
                10_000,
                10_000,
                106_032,
                832,
                *expected_task_counts,
            ):
                return ParseResult(feedback="full OOD completion has the wrong inventory", metrics={})

            metrics: dict[str, float] = {}
            observed: list[str] = []
            accuracies: list[float] = []
            summaries: list[str] = []
            for _index, row in metric_rows:
                match = METRIC_LINE.fullmatch(row)
                if match is None:
                    return ParseResult(feedback="full OOD metric record is malformed", metrics={})
                (
                    metric_task,
                    setting,
                    ood_name,
                    raw_auroc,
                    raw_fpr95,
                    raw_accuracy,
                    raw_metric_fit,
                    raw_metric_id,
                    raw_ood,
                    raw_base_ood_batches,
                    raw_id_score_batches,
                    raw_ood_score_batches,
                    raw_seconds,
                ) = match.groups()
                if metric_task != task or setting not in expected or setting in observed:
                    return ParseResult(feedback="full OOD metric task/settings are invalid", metrics={})
                expected_ood, expected_count, expected_batches, expected_extra_batches = expected[setting]
                metric_counts = tuple(map(int, (
                    raw_metric_fit,
                    raw_metric_id,
                    raw_ood,
                    raw_base_ood_batches,
                    raw_id_score_batches,
                    raw_ood_score_batches,
                )))
                expected_metric_counts = (
                    50_000,
                    10_000,
                    expected_count,
                    expected_batches,
                    158 if is_input_preproc else 0,
                    expected_extra_batches if is_input_preproc else 0,
                )
                if ood_name != expected_ood or metric_counts != expected_metric_counts:
                    return ParseResult(feedback="full OOD metric has the wrong inventory", metrics={})
                try:
                    auroc, fpr95, accuracy, seconds = map(
                        float, (raw_auroc, raw_fpr95, raw_accuracy, raw_seconds),
                    )
                except (ValueError, OverflowError):
                    return ParseResult(feedback="full OOD numeric field is malformed", metrics={})
                if not all(math.isfinite(value) for value in (auroc, fpr95, accuracy, seconds)):
                    return ParseResult(feedback="full OOD metric is non-finite", metrics={})
                if not 0.0 <= auroc <= 1.0 or not 0.0 <= fpr95 <= 1.0:
                    return ParseResult(feedback="full OOD metric is outside [0, 1]", metrics={})
                if not 0.90 <= accuracy <= 1.0 or seconds <= 0.0:
                    return ParseResult(feedback="full OOD classifier/runtime proof is invalid", metrics={})
                observed.append(setting)
                accuracies.append(accuracy)
                metrics[f"auroc_{setting}"] = auroc
                metrics[f"fpr95_{setting}"] = fpr95
                metrics[f"id_acc_{setting}"] = accuracy
                summaries.append(f"{ood_name}: AUROC={auroc:.6f}, FPR95={fpr95:.6f}")
            if observed != list(expected) or len(set(accuracies)) != 1:
                return ParseResult(
                    feedback="full OOD settings are reordered, incomplete, or inconsistent",
                    metrics={},
                )
            return ParseResult(
                feedback=f"complete authenticated full-image OOD evaluation for {task}; "
                + "; ".join(summaries),
                metrics=metrics,
            )

    StrictOODParser.__name__ = "Parser"
    return StrictOODParser
