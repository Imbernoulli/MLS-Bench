"""Host-side output parser / scorer for optimization-multi-objective.

Runs in the harness process on the HOST — never inside the agent container. The
agent's program now only emits its final non-dominated population's objective
values and decision vectors:

    MOEA_PRED env=<alias> seed=<seed> shape=<n>,<m> objs=<base64 float64>
              nvar=<d> xs=<base64 float64>

The emitted objective values are NOT trusted: the fitness values are assigned
in the same process as the agent-editable strategy, which can overwrite them.
This parser re-evaluates every emitted decision vector with the true objective
function from ``dgp.PROBLEMS``, rejects the run (no metrics for that problem)
if a vector is non-finite or out of bounds or if an emitted objective value
disagrees with its re-evaluation, and scores the non-dominated subset of the
re-evaluated values. For an honest run the re-evaluated values equal the
emitted ones and the emitted set is already non-dominated, so the metrics are
unchanged.

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
    r"\s+nvar=(\d+)\s+xs=(\S+)"
)
_PRED_LINE_RE = re.compile(r"MOEA_PRED\s+env=(\S+)")

# Relative / absolute tolerance for an emitted objective value versus its
# re-evaluation. The in-container evaluator inlines the same arithmetic as
# deap.benchmarks, so honest values agree to the last bit; the tolerance only
# absorbs harmless floating-point reordering.
_OBJ_RTOL = 1e-9
_OBJ_ATOL = 1e-9


def _nondominated_mask(F: np.ndarray) -> np.ndarray:
    """Rows of F (minimized) not dominated by any other row.

    Same dominance rule as DEAP (no worse in every objective, strictly better in
    at least one); duplicates do not dominate each other and are all kept.
    """
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        le = np.all(F <= F[i], axis=1)
        lt = np.any(F < F[i], axis=1)
        if np.any(le & lt):
            keep[i] = False
    return keep


def verify_front(cmd_label, objs, xs):
    """Re-evaluate the emitted decision vectors with the true objectives.

    Returns (true_front_values, None) on success, or (None, reason) when the
    emitted population is invalid (wrong dimensions, non-finite or
    out-of-bounds decision vectors, or objective values that were not produced
    by the true objective function).
    """
    cfg = dgp.PROBLEMS[cmd_label]
    n_var, n_obj = int(cfg["n_var"]), int(cfg["n_obj"])
    lo, hi = float(cfg["bounds"][0]), float(cfg["bounds"][1])
    if objs.shape[0] == 0:
        return None, "empty front"
    if objs.shape[1] != n_obj:
        return None, f"objective dimension {objs.shape[1]} != {n_obj}"
    if xs.shape != (objs.shape[0], n_var):
        return None, f"decision-vector shape {xs.shape} != ({objs.shape[0]}, {n_var})"
    if not np.all(np.isfinite(xs)):
        return None, "non-finite decision variable"
    if np.any(xs < lo) or np.any(xs > hi):
        return None, f"decision variable outside bounds [{lo}, {hi}]"
    func = cfg["func"]
    true_vals = np.array(
        [tuple(func([float(v) for v in row])) for row in xs], dtype=np.float64
    )
    if not np.allclose(objs, true_vals, rtol=_OBJ_RTOL, atol=_OBJ_ATOL):
        bad = int(np.sum(~np.all(np.isclose(objs, true_vals, rtol=_OBJ_RTOL,
                                            atol=_OBJ_ATOL), axis=1)))
        return None, (f"{bad} of {objs.shape[0]} emitted objective vectors differ "
                      "from the true objective values of their decision vectors")
    return true_vals[_nondominated_mask(true_vals)], None


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

        n_pred_lines = sum(
            1 for m in _PRED_LINE_RE.finditer(raw_output)
            if alias is None or m.group(1) == alias
        )
        # The fixed harness prints exactly one MOEA_PRED line per run; any other
        # count means something else printed prediction lines -> reject.
        if n_pred_lines > 1:
            feedback_parts.append(
                f"REJECTED ({cmd_label}): {n_pred_lines} MOEA_PRED lines found, "
                "expected exactly one; no metrics are reported for this problem."
            )
            return ParseResult(feedback="\n".join(feedback_parts), metrics={})
        n_valid = 0
        for m in _PRED_RE.finditer(raw_output):
            env, seed_s, n_s, mdim_s, payload, nvar_s, x_payload = m.groups()
            if alias is not None and env != alias:
                continue
            try:
                flat = np.frombuffer(base64.b64decode(payload), dtype=np.float64)
                x_flat = np.frombuffer(base64.b64decode(x_payload), dtype=np.float64)
            except Exception:
                continue
            n, mdim, nvar = int(n_s), int(mdim_s), int(nvar_s)
            if flat.shape[0] != n * mdim or x_flat.shape[0] != n * nvar:
                continue
            n_valid += 1
            front_values, reason = verify_front(
                cmd_label, flat.reshape(n, mdim), x_flat.reshape(n, nvar)
            )
            if front_values is None:
                feedback_parts.append(
                    f"REJECTED ({cmd_label}): {reason}; no metrics are reported "
                    "for this problem."
                )
                continue

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

        if n_pred_lines > n_valid:
            feedback_parts.append(
                f"REJECTED ({cmd_label}): {n_pred_lines - n_valid} MOEA_PRED line(s) "
                "are malformed or lack the decision vectors (nvar=/xs=)."
            )

        if not feedback_parts:
            return ParseResult(feedback=raw_output[-3000:], metrics={})
        return ParseResult(feedback="\n".join(feedback_parts), metrics=metrics)

    def _parse_train_progress(self, output: str) -> str:
        lines = [l.strip() for l in output.splitlines() if l.strip().startswith("TRAIN_PROGRESS")]
        if not lines:
            return ""
        return "Training progress (last generations):\n" + "\n".join(lines[-5:])
