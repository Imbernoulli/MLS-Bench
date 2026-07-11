"""Strict parser for the full-scale Market-1501 re-ID protocol."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult


_TASK_ID = "reid-embedding-head"
_PROTOCOL_ID = "market1501-resnet50-60e-v2"
_EXPECTED_TOTAL_STEPS = 11_003
_EXPECTED_TRAIN_SAMPLES = 704_192
_EXPECTED_PROTOCOL = (
    f"REID_PROTOCOL schema=2 task={_TASK_ID} protocol={_PROTOCOL_ID} "
    "seed=42 model=resnet50 epochs=60 batch=64 instances=4 "
    "train_images=12936 query_images=3368 gallery_images=19732 "
    "train_ids=751 query_ids=750 "
    "train_sha=4f1a5416bad595a67a45652568919252e56a54e99c49fd74f1fd29492123f3d3 "
    "query_sha=d34ff6d094521111a10a16f7879f01bb210abdeab24efba4b950fe1f3b9e90f7 "
    "gallery_sha=7900c8355955f1ca7e2ad5d6844f4be03dddfc3ded1f7a21cf43e55441075c4e "
    "weights_sha=0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
)
_EXPECTED_QUERIES = {"easy": 1122, "medium": 1123, "hard": 1123}
_EPOCH_RE = re.compile(
    r"^REID_EPOCH epoch=(\d+) steps=(\d+) total_steps=(\d+) "
    r"loss=(\S+) lr=(\S+)$"
)
_TRAIN_DONE_RE = re.compile(
    r"^REID_TRAIN_COMPLETE epochs=(\d+) total_steps=(\d+) train_samples=(\d+)$"
)
_METRIC_RE = re.compile(
    r"^REID_METRICS setting=(easy|medium|hard) map=(\S+) rank1=(\S+) "
    r"rank5=(\S+) num_query=(\d+) num_gallery=(\d+) elapsed=(\S+)$"
)
_EXPECTED_EVAL_DONE = (
    f"REID_EVAL_COMPLETE schema=2 task={_TASK_ID} protocol={_PROTOCOL_ID} "
    "settings=easy,medium,hard query_total=3368 gallery=19732 "
    "total_steps=11003 train_samples=704192 status=ok"
)
_FAILURE_MARKERS = (
    "traceback",
    "verification failed",
    "evaluation failed",
    "command failed",
    "budget check failed",
    "status: failed",
    "exit code",
    "process exited",
    "timed out",
    "timeout",
    "out of memory",
    "cuda out of memory",
    "cancelled",
    "canceled",
    "node_fail",
    "killed",
    "fatal:",
    "reid_surface_fallback",
    "reid_nonfinite",
    "reid_protocol_error",
    "reid_weights_fallback",
)


def _failure(reason: str) -> ParseResult:
    return ParseResult(feedback=f"Full-scale protocol failed: {reason}", metrics={})


class Parser(OutputParser):
    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        if cmd_label != "market":
            return _failure(f"unexpected command label {cmd_label!r}")
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        folded_output = raw_output.casefold()
        if any(marker in folded_output for marker in _FAILURE_MARKERS):
            return _failure("runtime reported a failure marker")

        protocol_at = [i for i, line in enumerate(lines) if line.startswith("REID_PROTOCOL")]
        if len(protocol_at) != 1 or lines[protocol_at[0]] != _EXPECTED_PROTOCOL:
            return _failure("missing, duplicate, or mismatched inventory proof")

        epoch_rows = []
        for index, line in enumerate(lines):
            if not line.startswith("REID_EPOCH"):
                continue
            match = _EPOCH_RE.fullmatch(line)
            if match is None:
                return _failure("malformed epoch proof")
            try:
                loss = float(match.group(4))
                lr = float(match.group(5))
            except ValueError:
                return _failure("invalid epoch numeric field")
            epoch_rows.append(
                (
                    index,
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    loss,
                    lr,
                )
            )
        if len(epoch_rows) != 60 or [row[1] for row in epoch_rows] != list(range(60)):
            return _failure("training did not report exactly epochs 0 through 59")
        previous_total = 0
        for _index, _epoch, steps, total_steps, loss, lr in epoch_rows:
            if steps < 150 or total_steps != previous_total + steps:
                return _failure("epoch step accounting is incomplete")
            if not math.isfinite(loss) or loss < 0 or not math.isfinite(lr) or lr <= 0:
                return _failure("epoch loss or learning rate is invalid")
            previous_total = total_steps
        if previous_total != _EXPECTED_TOTAL_STEPS:
            return _failure("optimizer budget does not match the full protocol")

        train_done = []
        for index, line in enumerate(lines):
            if not line.startswith("REID_TRAIN_COMPLETE"):
                continue
            match = _TRAIN_DONE_RE.fullmatch(line)
            if match is None:
                return _failure("malformed training completion proof")
            train_done.append(
                (index, int(match.group(1)), int(match.group(2)), int(match.group(3)))
            )
        if len(train_done) != 1:
            return _failure("missing or duplicate training completion proof")
        train_at, epochs, total_steps, train_samples = train_done[0]
        if (
            epochs != 60
            or total_steps != _EXPECTED_TOTAL_STEPS
            or train_samples != _EXPECTED_TRAIN_SAMPLES
            or train_samples != total_steps * 64
        ):
            return _failure("training completion counters do not match the epoch trace")

        metric_rows = {}
        metric_indices = []
        for index, line in enumerate(lines):
            if not line.startswith("REID_METRICS"):
                continue
            match = _METRIC_RE.fullmatch(line)
            if match is None:
                return _failure("malformed retrieval metric proof")
            setting = match.group(1)
            if setting in metric_rows:
                return _failure(f"duplicate metric for {setting}")
            try:
                values = tuple(float(match.group(i)) for i in (2, 3, 4, 7))
            except ValueError:
                return _failure(f"invalid metric numeric field for {setting}")
            map_value, rank1, rank5, elapsed = values
            num_query = int(match.group(5))
            num_gallery = int(match.group(6))
            if not all(math.isfinite(value) for value in values):
                return _failure(f"non-finite metric for {setting}")
            if not all(0.0 <= value <= 1.0 for value in (map_value, rank1, rank5)):
                return _failure(f"out-of-range retrieval metric for {setting}")
            if elapsed <= 0 or num_query != _EXPECTED_QUERIES[setting] or num_gallery != 19_732:
                return _failure(f"incomplete evaluation inventory for {setting}")
            if rank5 < rank1:
                return _failure(f"inconsistent CMC metrics for {setting}")
            metric_rows[setting] = (map_value, rank1, rank5, elapsed)
            metric_indices.append(index)
        if set(metric_rows) != set(_EXPECTED_QUERIES):
            return _failure("one or more required difficulty settings are missing")
        if [lines[index].split(" setting=", 1)[1].split(" ", 1)[0] for index in metric_indices] != [
            "easy",
            "medium",
            "hard",
        ]:
            return _failure("retrieval settings are out of order")
        elapsed_values = {values[3] for values in metric_rows.values()}
        if len(elapsed_values) != 1:
            return _failure("retrieval settings disagree on elapsed time")

        eval_done_at = [i for i, line in enumerate(lines) if line.startswith("REID_EVAL_COMPLETE")]
        if len(eval_done_at) != 1 or lines[eval_done_at[0]] != _EXPECTED_EVAL_DONE:
            return _failure("missing, duplicate, or mismatched evaluation completion proof")
        if eval_done_at[0] != len(lines) - 1:
            return _failure("evaluation completion must be the final non-empty record")
        if not (
            protocol_at[0] < epoch_rows[0][0]
            and epoch_rows[-1][0] < train_at
            and train_at < min(metric_indices)
            and max(metric_indices) < eval_done_at[0]
        ):
            return _failure("protocol records are out of order")

        metrics = {}
        feedback_rows = []
        for setting in ("easy", "medium", "hard"):
            map_value, rank1, rank5, _elapsed = metric_rows[setting]
            metrics[f"map_{setting}"] = map_value
            metrics[f"rank1_{setting}"] = rank1
            metrics[f"rank5_{setting}"] = rank5
            feedback_rows.append(
                f"{setting}: mAP={map_value:.6f} rank1={rank1:.6f} rank5={rank5:.6f}"
            )
        return ParseResult(
            feedback="Full-scale Market-1501 completed.\n" + "\n".join(feedback_rows),
            metrics=metrics,
        )
