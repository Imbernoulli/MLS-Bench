"""Task-specific output parser for optimization-nas.

The search program reports the architecture it selected:
    FINAL_ARCH arch=<arch_str> queries=<n>
and the held-out TEST accuracy is looked up HERE — host-side natively,
verifier-side in Harbor — from the NAS-Bench-201 table provider in
``holdout/optimization-nas/dgp.py``. The test split never exists in the
agent's process. Reporting a fabricated FINAL_ARCH is not an exploit: naming
an architecture is exactly the ``get_best_architecture()`` contract, and the
accuracy is always taken from the held-out table, never from the program's
own output.

Metrics are keyed by dataset label, e.g. test_accuracy_CIFAR-10.
"""

import importlib.util
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlsbench.agent.parsers import OutputParser, ParseResult

_DGP_CACHE = None


def _load_dgp():
    """Import the table provider from either layout: next to this parser
    (Harbor: staged under tests/meta/) or under PROJECT_ROOT/holdout (native)."""
    global _DGP_CACHE
    if _DGP_CACHE is not None:
        return _DGP_CACHE
    for cand in (
        _HERE / "dgp.py",
        PROJECT_ROOT / "holdout" / "optimization-nas" / "dgp.py",
    ):
        if cand.exists():
            spec = importlib.util.spec_from_file_location("optimization_nas_dgp", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _DGP_CACHE = mod
            return mod
    raise FileNotFoundError(
        "optimization-nas dgp.py not found (looked next to parser and in holdout/)"
    )


# Eval-command label -> env name used by the table provider.
_LABEL_TO_ENV = {
    "CIFAR-10": "cifar10",
    "CIFAR-100": "cifar100",
    "ImageNet16-120": "imagenet16",
}

_FINAL_RE = re.compile(r"FINAL_ARCH\s+arch=(\S+)(?:\s+queries=(\d+))?")


class Parser(OutputParser):
    """Parser for the optimization-nas task."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        feedback_parts = []
        metrics: dict = {}

        train_feedback = self._parse_train_metrics(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        # Last report wins — same semantics as get_best_architecture().
        match = None
        for m in _FINAL_RE.finditer(raw_output):
            match = m
        if match:
            arch_str = match.group(1)
            env = _LABEL_TO_ENV.get(cmd_label)
            try:
                test_acc = float(_load_dgp().test_accuracy(env, arch_str))
            except KeyError:
                feedback_parts.append(
                    f"Final architecture {arch_str!r} is not a valid "
                    f"NAS-Bench-201 architecture for {cmd_label}; "
                    f"no test accuracy recorded."
                )
            else:
                metrics[f"test_accuracy_{cmd_label}"] = test_acc
                feedback_parts.append(
                    f"Test results ({cmd_label}):\n"
                    f"  Test accuracy: {test_acc:.4f}"
                )

        feedback = "\n".join(feedback_parts) if feedback_parts else raw_output
        return ParseResult(feedback=feedback, metrics=metrics)

    def _parse_train_metrics(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_METRICS ")]
        if not lines:
            return ""
        return "Search progress (last epochs):\n" + "\n".join(lines[-5:])
