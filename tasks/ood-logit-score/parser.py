"""Strict parser for the full-image CIFAR-10 OOD protocol."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

PROTOCOL = "openood_cifar10_resnet18_full_v1"
EXPECTED_DATA_SHA256 = "796799c9a1c073784b02a3f42a00d0fc8b902387b1f3345b5b3d1f2631e0722d"
EXPECTED_MODEL_SHA256 = "8859e0ff484ab029fcd5bd0f85052c5679d82b8e36a7c066e922e2fdff62b7dc"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
METRIC = re.compile(
    rf"OOD_METRICS protocol={PROTOCOL} task=ood-logit-score setting=(\S+) ood=(\S+) "
    rf"auroc=({NUMBER}) fpr95=({NUMBER}) id_acc=({NUMBER}) "
    rf"n_fit=(\d+) n_id=(\d+) n_ood=(\d+) forward_batches=(\d+) "
    rf"inference_seconds=({NUMBER}) status=ok"
)
PROTOCOL_RECORD = re.compile(
    rf"OOD_PROTOCOL protocol={PROTOCOL} task=ood-logit-score "
    rf"model=openood_resnet18_32x32 "
    rf"batch_size=128 seed=42 device=.+"
)
COMPLETE = re.compile(
    rf"OOD_COMPLETE protocol={PROTOCOL} task=ood-logit-score "
    rf"data_sha256=([0-9a-f]{{64}}) "
    rf"checkpoint_sha256=([0-9a-f]{{64}}) n_fit=(\d+) n_id=(\d+) "
    rf"n_svhn=(\d+) n_cifar100=(\d+) n_tin=(\d+) "
    rf"total_forward_images=(\d+) total_forward_batches=(\d+) status=ok"
)
EXPECTED = {
    "ood_logit_svhn_full": ("svhn", 26_032, 204),
    "ood_logit_cifar100_full": ("cifar100", 10_000, 79),
    "ood_logit_tin_full": ("tin", 10_000, 79),
}
FAILURE = re.compile(
    r"(?:Traceback \(most recent call last\)|\[COMMAND FAILED|"
    r"\b(?:OOD|VERIFIER)_FAILURE\b|\b(?:verification|evaluation) failed\b|"
    r"\bbudget check failed\b|\bstatus\s*[:=]\s*failed\b|"
    r"\bexit(?:ed)?(?: with)? code\b|\bprocess exited\b|\bcommand failed\b|"
    r"\bTIMEOUT\b|\btimed? out\b|\btime limit\b|CUDA out of memory|"
    r"\bout of memory\b|\bOOM\b|\bKilled\b|Segmentation fault|"
    r"\bcancel(?:led|ed)?\b|\bnode[_ -]?fail(?:ure|ed)?\b|"
    r"\bnon[- ]finite\b|_FALLBACK\b|\b(?:nan|inf|infinity)\b)",
    re.IGNORECASE,
)
AUTHORITATIVE_PREFIX = re.compile(
    r"^OOD_(?:PROTOCOL|METRICS|COMPLETE)(?:\b|_)", re.IGNORECASE,
)


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label != "ood_logit_full_protocol":
            return ParseResult(feedback="full OOD command label is wrong", metrics={})
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        authoritative = [line for line in lines if AUTHORITATIVE_PREFIX.match(line)]
        protocols = [line for line in lines if line.startswith("OOD_PROTOCOL")]
        records = [line for line in lines if line.startswith("OOD_METRICS")]
        completions = [line for line in lines if line.startswith("OOD_COMPLETE")]
        if FAILURE.search(raw_output):
            return ParseResult(feedback="full OOD verification reported a failure", metrics={})
        if (
            len(protocols) != 1
            or PROTOCOL_RECORD.fullmatch(protocols[0]) is None
            or len(records) != 3
            or len(completions) != 1
            or len(authoritative) != 5
            or not lines
            or lines[-1] != completions[0]
            or lines.index(protocols[0]) >= lines.index(records[0])
        ):
            return ParseResult(feedback="full OOD verification did not complete exactly once", metrics={})

        completion = COMPLETE.fullmatch(completions[0])
        if completion is None:
            return ParseResult(feedback="full OOD completion proof is malformed", metrics={})
        data_sha, model_sha, *raw_counts = completion.groups()
        if data_sha != EXPECTED_DATA_SHA256 or model_sha != EXPECTED_MODEL_SHA256:
            return ParseResult(feedback="full OOD completion proof has an unpinned artifact", metrics={})
        if tuple(map(int, raw_counts)) != (50_000, 10_000, 26_032, 10_000, 10_000, 106_032, 832):
            return ParseResult(feedback="full OOD completion proof has the wrong inventory", metrics={})

        metrics: dict[str, float] = {}
        observed_settings: list[str] = []
        accuracies: list[float] = []
        summaries: list[str] = []
        for record in records:
            match = METRIC.fullmatch(record)
            if match is None:
                return ParseResult(feedback="full OOD metric record is malformed", metrics={})
            setting, ood_name, raw_auroc, raw_fpr95, raw_acc, raw_fit, raw_id, raw_ood, raw_batches, raw_seconds = match.groups()
            if setting not in EXPECTED or setting in observed_settings:
                return ParseResult(feedback="full OOD settings are missing or duplicated", metrics={})
            expected_ood, expected_count, expected_batches = EXPECTED[setting]
            counts = tuple(map(int, (raw_fit, raw_id, raw_ood, raw_batches)))
            if ood_name != expected_ood or counts != (50_000, 10_000, expected_count, expected_batches):
                return ParseResult(feedback="full OOD metric has the wrong setting inventory", metrics={})
            auroc, fpr95, accuracy, seconds = map(
                float, (raw_auroc, raw_fpr95, raw_acc, raw_seconds),
            )
            if not all(map(math.isfinite, (auroc, fpr95, accuracy, seconds))):
                return ParseResult(feedback="full OOD metric is non-finite", metrics={})
            if not 0.0 <= auroc <= 1.0 or not 0.0 <= fpr95 <= 1.0:
                return ParseResult(feedback="full OOD metric is outside [0, 1]", metrics={})
            if not 0.90 <= accuracy <= 1.0 or seconds <= 0.0:
                return ParseResult(feedback="full OOD classifier proof is invalid", metrics={})
            observed_settings.append(setting)
            accuracies.append(accuracy)
            metrics[f"auroc_{setting}"] = auroc
            metrics[f"fpr95_{setting}"] = fpr95
            metrics[f"id_acc_{setting}"] = accuracy
            summaries.append(
                f"{ood_name}: AUROC={auroc:.6f}, FPR95={fpr95:.6f}, n_ood={expected_count}"
            )
        if observed_settings != list(EXPECTED) or len(set(accuracies)) != 1:
            return ParseResult(
                feedback="full OOD settings are reordered, incomplete, or inconsistent",
                metrics={},
            )
        return ParseResult(
            feedback="complete full-image OOD evaluation; " + "; ".join(summaries),
            metrics=metrics,
        )
