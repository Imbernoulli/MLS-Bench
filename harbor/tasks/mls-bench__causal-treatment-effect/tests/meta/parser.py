"""Host-side output parser / scorer for causal-treatment-effect.

This runs in the harness process on the HOST — never inside the agent's
container. The agent's program now only emits its cross-fitted predictions:

    CATE_PRED dataset=<d> seed=<s> rep=<r> data_seed=<ds> n=<n> tau_hat=<base64 float64>

The held-out ground truth (the true treatment effect ``tau``) lives in
``holdout/causal-treatment-effect/dgp.py`` — a module that is NOT bind-mounted
into the agent's container. This parser regenerates the truth here, host-side,
and scores the predictions against it, using the same PEHE / ATE_error formulas
and the same per-rep mean aggregation as before. Honest results are therefore
identical to the pre-fix pipeline, while the agent's process can never reach the
answer.
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Host-only scoring module (never inside the agent container).
# Locate the host-only DGP module. Native layout:
# <repo>/holdout/causal-treatment-effect/dgp.py. Harbor layout: score_task.py loads this
# parser from a private copy of tests/meta/ with the held-out dgp.py staged
# next to it -- so also try the parser's own directory. Try both.
_HERE = Path(__file__).resolve().parent
for _cand in (PROJECT_ROOT / "holdout" / "causal-treatment-effect", _HERE, _HERE / "holdout" / "causal-treatment-effect"):
    if (_cand / "dgp.py").exists():
        sys.path.insert(0, str(_cand))
        break

from mlsbench.agent.parsers import OutputParser, ParseResult

import dgp  # noqa: E402  (from holdout/causal-treatment-effect/)


_PRED_RE = re.compile(
    r"CATE_PRED\s+dataset=(\S+)\s+seed=(\S+)\s+rep=(\d+)\s+data_seed=(\d+)\s+"
    r"n=(\d+)\s+tau_hat=(\S+)"
)


class Parser(OutputParser):
    """Parser for the causal-treatment-effect task (predict-then-score)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        per_rep = []  # (rep, pehe, ate_err)
        for m in _PRED_RE.finditer(raw_output):
            dataset, _seed, rep_s, data_seed_s, n_s, payload = m.groups()
            if dataset != cmd_label:
                continue
            rep = int(rep_s)
            data_seed = int(data_seed_s)
            n = int(n_s)
            try:
                tau_hat = np.frombuffer(
                    base64.b64decode(payload), dtype=np.float64
                )
            except Exception:  # malformed payload -> skip rep
                continue
            if tau_hat.shape[0] != n:
                continue
            tau, ate = dgp.truth(dataset, data_seed)
            if tau_hat.shape[0] != tau.shape[0]:
                continue
            pehe = float(dgp.compute_pehe(tau, tau_hat))
            ate_err = float(dgp.compute_ate_error(ate, tau_hat))
            per_rep.append((rep, pehe, ate_err))

        if not per_rep:
            # No scorable predictions — surface a tail of the raw output to help
            # the agent debug (no ground truth is ever included here).
            return ParseResult(feedback=raw_output[-3000:], metrics={})

        per_rep.sort()
        pehe_vals = np.array([p for _, p, _ in per_rep])
        ate_vals = np.array([a for _, _, a in per_rep])
        mean_pehe = float(np.mean(pehe_vals))
        mean_ate = float(np.mean(ate_vals))

        metrics = {
            f"PEHE_{cmd_label}": mean_pehe,
            f"ATE_error_{cmd_label}": mean_ate,
        }

        last = per_rep[-5:]
        rep_lines = "\n".join(
            f"  rep={r} PEHE={p:.6f} ATE_error={a:.6f}" for r, p, a in last
        )
        feedback = (
            f"Per-repetition metrics ({cmd_label}, last {len(last)} of "
            f"{len(per_rep)}):\n{rep_lines}\n"
            f"Final metrics ({cmd_label}):\n"
            f"  PEHE_{cmd_label}: {mean_pehe:.6f}\n"
            f"  ATE_error_{cmd_label}: {mean_ate:.6f}"
        )
        return ParseResult(feedback=feedback, metrics=metrics)
