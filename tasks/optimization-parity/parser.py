"""Host-side output parser / scorer for the optimization-parity task.

This runs in the harness process on the HOST — never inside the agent's
container. The agent's program no longer computes test accuracy (it does not
have the hidden parity secret or the test labels). Instead, for each training
run it emits its thresholded, bit-packed predictions on a deterministically
regenerated test set:

    PARITY_PRED config=<tag> seed=<s> secret=<si> order=<oi> n=<n> preds=<b64>

The hidden secret, the parity labeling function, the test-set construction and
the metric live in ``holdout/optimization-parity/dgp.py`` — a module that is NOT
bind-mounted into the agent's container. This parser regenerates the held-out
test labels ``y_test`` here, host-side, and scores the predictions against them
using the same 0.5 threshold and the same mean aggregation as before, so honest
results are identical to the pre-fix pipeline while the agent's process can never
reach the answer.
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Host-only scoring module (never inside the agent container).
# Locate the host-only DGP module. Native layout: <repo>/holdout/optimization-parity/dgp.py.
# Harbor layout: score_task.py copies tests/meta/ to a private dir and loads
# parser.py from there, with the held-out dgp.py staged next to it. Try both.
_HERE = Path(__file__).resolve().parent
for _cand in (PROJECT_ROOT / "holdout" / "optimization-parity", _HERE, _HERE / "holdout" / "optimization-parity"):
    if (_cand / "dgp.py").exists():
        sys.path.insert(0, str(_cand))
        break

from mlsbench.agent.parsers import OutputParser, ParseResult

import dgp  # noqa: E402  (from holdout/optimization-parity/)


_PRED_RE = re.compile(
    r"PARITY_PRED\s+config=(\S+)\s+seed=(\d+)\s+secret=(\d+)\s+order=(\d+)\s+"
    r"n=(\d+)\s+preds=(\S+)"
)

# n<N>_k<K> -> (n_features, secret_size)
_TAG_RE = re.compile(r"n(\d+)_k(\d+)")

_RUN_RE = re.compile(
    r"RUN_METRICS\s+secret=(\d+)\s+order=(\d+)\s+steps=(\d+)"
)


def _parse_tag(tag):
    m = _TAG_RE.fullmatch(tag)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


class Parser(OutputParser):
    """Parse PARITY_PRED predictions, score against held-out truth."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts: list[str] = []

        train_lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip().startswith("TRAIN_METRICS")
        ]
        if train_lines:
            feedback_parts.append("Training progress:\n" + "\n".join(train_lines[-5:]))

        # Cache regenerated test labels per (config, seed, secret_index).
        truth_cache: dict[tuple, np.ndarray] = {}
        accuracies: list[float] = []
        run_lines: list[str] = []

        for m in _PRED_RE.finditer(raw_output):
            tag, seed_s, si_s, order_s, n_s, payload = m.groups()
            n_features, secret_size = _parse_tag(tag)
            if n_features is None:
                continue
            seed = int(seed_s)
            secret_index = int(si_s)
            order_index = int(order_s)
            n = int(n_s)
            try:
                packed = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
                preds = np.unpackbits(packed)[:n].astype(np.float32)
            except Exception:  # malformed payload -> skip run
                continue
            if preds.shape[0] != n:
                continue

            key = (n_features, secret_size, seed, secret_index, n)
            if key not in truth_cache:
                config = dgp.make_config(
                    n_features=n_features, secret_size=secret_size
                )
                # Regenerate the held-out test labels at exactly the number of
                # predictions the runner emitted (prefix-consistent across the
                # default / smoke test_set_size).
                truth_cache[key] = dgp.truth(
                    config, seed, secret_index, test_set_size=n
                )
            y_test = truth_cache[key]
            if preds.shape[0] != y_test.shape[0]:
                continue

            acc = dgp.accuracy(preds, y_test)
            accuracies.append(acc)
            run_lines.append(
                f"  secret={secret_index} order={order_index} test_accuracy={acc:.6f}"
            )

        # Recover steps for mean_steps (informational, matches old metric).
        steps = [int(m.group(3)) for m in _RUN_RE.finditer(raw_output)]

        if not accuracies:
            feedback_parts.append(raw_output[-3000:])
            return ParseResult(feedback="\n\n".join(feedback_parts), metrics={})

        acc_arr = np.asarray(accuracies, dtype=np.float64)
        test_accuracy = float(acc_arr.mean())
        test_accuracy_std = float(acc_arr.std())  # population std (ddof=0)
        mean_steps = float(np.mean(steps)) if steps else 0.0
        num_runs = int(acc_arr.shape[0])

        base_metrics = {
            "test_accuracy": test_accuracy,
            "score": test_accuracy,
            "test_accuracy_std": test_accuracy_std,
            "mean_steps": mean_steps,
            "num_runs": num_runs,
        }
        suffix = "" if cmd_label == "eval" else f"_{cmd_label}"
        metrics = {f"{key}{suffix}": value for key, value in base_metrics.items()}

        if run_lines:
            feedback_parts.append(
                f"Recent runs ({cmd_label}, last {min(5, len(run_lines))} of "
                f"{len(run_lines)}):\n" + "\n".join(run_lines[-5:])
            )
        feedback_parts.append(
            f"Final metrics ({cmd_label}):\n"
            f"  test_accuracy: {test_accuracy:.6f}\n"
            f"  score: {test_accuracy:.6f}\n"
            f"  test_accuracy_std: {test_accuracy_std:.6f}\n"
            f"  mean_steps: {mean_steps:.1f}\n"
            f"  num_runs: {num_runs}"
        )

        return ParseResult(feedback="\n\n".join(feedback_parts), metrics=metrics)
