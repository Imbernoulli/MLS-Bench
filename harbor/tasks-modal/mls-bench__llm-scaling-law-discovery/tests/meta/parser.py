"""Parser for llm-scaling-law-discovery.

The program reports its held-out predictions on stdout:
    SLD_PRED benchmark=<name> y_pred=<json array>
and the test metrics are computed HERE against the held-out targets — which
never enter the agent's process (the baked test JSONLs carry features only).
Natively the targets live in holdout/llm-scaling-law-discovery/; in Harbor
the same dgp.py + truth JSONLs are staged next to this parser under
tests/meta/. Reporting fabricated predictions is not an exploit: emitting
predictions IS the interface, and they are always scored against the
held-out targets.

Metric keys and values are unchanged: r2/mae/rmse/nmae per benchmark label,
with the same formulas the benchmark previously evaluated in-process, rounded
through the same %.6f formatting the old TEST_METRICS line carried.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

EPS = 1e-8

_DGP_CACHE = None


def _load_dgp():
    """Import the held-out-target provider from either layout: next to this
    parser (Harbor: staged under tests/meta/) or PROJECT_ROOT/holdout (native)."""
    global _DGP_CACHE
    if _DGP_CACHE is not None:
        return _DGP_CACHE
    for cand in (
        _HERE / "dgp.py",
        PROJECT_ROOT / "holdout" / "llm-scaling-law-discovery" / "dgp.py",
    ):
        if cand.exists():
            spec = importlib.util.spec_from_file_location("sld_dgp", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _DGP_CACHE = mod
            return mod
    raise FileNotFoundError(
        "llm-scaling-law-discovery dgp.py not found (looked next to parser and in holdout/)"
    )


_PRED_RE = re.compile(r"SLD_PRED\s+benchmark=(\S+)\s+y_pred=(\[[^\n]*\])")


def _score(y_true, y_pred) -> dict:
    """Same formulas the benchmark previously computed in-process."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 0.0 if denom < EPS else float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nmae = float(mae / (np.std(y_true) + EPS))
    # Round through the same %.6f formatting the old TEST_METRICS line used,
    # so leaderboard values are bit-identical to the previous pipeline.
    return {k: float(f"{v:.6f}") for k, v in
            (("r2", r2), ("mae", mae), ("rmse", rmse), ("nmae", nmae))}


class Parser(OutputParser):
    """Parse scaling-law benchmark output and score predictions externally."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics = {}

        train_lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip().startswith("TRAIN_METRICS")
        ]
        if train_lines:
            feedback_parts.append(
                f"Training progress ({cmd_label}):\n" + "\n".join(train_lines[-5:])
            )

        # Last report per benchmark wins.
        pred = None
        for m in _PRED_RE.finditer(raw_output):
            if m.group(1) == cmd_label:
                pred = m.group(2)
        if pred is not None:
            try:
                y_pred = json.loads(pred)
                y_true = _load_dgp().truth(cmd_label)
            except Exception as exc:  # malformed report or unknown benchmark
                feedback_parts.append(f"Could not score predictions ({cmd_label}): {exc}")
            else:
                if len(y_pred) != len(y_true):
                    feedback_parts.append(
                        f"Prediction length mismatch ({cmd_label}): "
                        f"got {len(y_pred)}, expected {len(y_true)}; no metrics recorded."
                    )
                else:
                    scored = _score(y_true, y_pred)
                    for key, val in scored.items():
                        metrics[f"{key}_{cmd_label.replace('-', '_')}"] = val
                    pretty = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
                    feedback_parts.append(f"Final metrics ({cmd_label}): {pretty}")

        if not feedback_parts:
            feedback_parts.append(raw_output[-3000:])

        return ParseResult(feedback="\n".join(feedback_parts), metrics=metrics)
