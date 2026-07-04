"""Host-side output parser / scorer for optimization-multi-objective.

Runs in the harness process on the HOST — never inside the agent container. The
agent's program now only emits its final non-dominated population's objective
values:

    MOEA_PRED env=<alias> seed=<seed> shape=<n>,<m> objs=<base64 float64>

The benchmark problem identity (ZDT/DTLZ), the analytic true Pareto front, and
the IGD / hypervolume / spread metrics live in
``holdout/optimization-multi-objective/dgp.py`` — not bind-mounted into the
agent container. This parser regenerates the front here and scores HV/IGD/Spread
with the same per-problem keys (hv_<label>, igd_<label>, spread_<label>).
Honest results are identical to the pre-fix pipeline.
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Locate the host-only DGP module. Native layout:
# <repo>/holdout/optimization-multi-objective/dgp.py. Harbor layout: score_task.py loads this
# parser from a private copy of tests/meta/ with the held-out dgp.py staged
# next to it -- so also try the parser's own directory. Try both.
_HERE = Path(__file__).resolve().parent
for _cand in (PROJECT_ROOT / "holdout" / "optimization-multi-objective", _HERE, _HERE / "holdout" / "optimization-multi-objective"):
    if (_cand / "dgp.py").exists():
        sys.path.insert(0, str(_cand))
        break

from mlsbench.agent.parsers import OutputParser, ParseResult

import dgp  # noqa: E402  (host-only)

_PRED_RE = re.compile(
    r"MOEA_PRED\s+env=(\S+)\s+seed=(\d+)\s+shape=(\d+),(\d+)\s+objs=(\S+)"
)


class Parser(OutputParser):
    """Parser for optimization-multi-objective (optimize-then-score)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback_parts = []

        # The container only ever sees the opaque alias; map the host-side
        # problem label (e.g. "zdt1") to its alias to match the emitted line.
        alias = dgp.ALIASES.get(cmd_label)

        train_feedback = self._parse_train_progress(raw_output)
        if train_feedback:
            feedback_parts.append(train_feedback)

        for m in _PRED_RE.finditer(raw_output):
            env, seed_s, n_s, mdim_s, payload = m.groups()
            if alias is not None and env != alias:
                continue
            try:
                flat = np.frombuffer(base64.b64decode(payload), dtype=np.float64)
            except Exception:
                continue
            n, mdim = int(n_s), int(mdim_s)
            if flat.shape[0] != n * mdim:
                continue
            front_values = flat.reshape(n, mdim)

            res = dgp.score_front(front_values, cmd_label, seed=int(seed_s))
            metrics[f"hv_{cmd_label}"] = res["hv"]
            metrics[f"igd_{cmd_label}"] = res["igd"]
            metrics[f"spread_{cmd_label}"] = res["spread"]
            feedback_parts.append(
                f"Test results ({cmd_label}):\n"
                f"  hv: {res['hv']:.6f}\n"
                f"  igd: {res['igd']:.6f}\n"
                f"  spread: {res['spread']:.6f}"
            )

        if not feedback_parts:
            return ParseResult(feedback=raw_output[-3000:], metrics={})
        return ParseResult(feedback="\n".join(feedback_parts), metrics=metrics)

    def _parse_train_progress(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_PROGRESS")]
        if not lines:
            return ""
        return "Training progress (last generations):\n" + "\n".join(lines[-5:])
