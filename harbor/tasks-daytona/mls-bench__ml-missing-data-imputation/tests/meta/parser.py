"""Host-side output parser / scorer for ml-missing-data-imputation.

Runs in the harness process on the HOST — never inside the agent container. The
agent's program now only emits its imputed matrix:

    IMPUTE_PRED env=<env> seed=<seed> rows=<r> cols=<c> X_imputed=<base64 float64>

The dataset identity, the true matrix, the missingness mask, and the labels live
in ``holdout/ml-missing-data-imputation/dgp.py`` — a module NOT bind-mounted into
the agent container (so the agent neither knows which datasets are used for
evaluation nor can read the held-out true values). This parser regenerates the
truth here and scores the imputed matrix with the same RMSE + downstream metrics
and the same per-dataset keys (rmse_<env>, downstream_score_<env>). Honest
results are identical to the pre-fix pipeline.
"""

import base64
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Locate the host-only DGP module. Native layout:
# <repo>/holdout/ml-missing-data-imputation/dgp.py. Harbor layout: score_task.py loads this
# parser from a private copy of tests/meta/ with the held-out dgp.py staged
# next to it -- so also try the parser's own directory. Try both.
_HERE = Path(__file__).resolve().parent
for _cand in (PROJECT_ROOT / "holdout" / "ml-missing-data-imputation", _HERE, _HERE / "holdout" / "ml-missing-data-imputation"):
    if (_cand / "dgp.py").exists():
        sys.path.insert(0, str(_cand))
        break

from mlsbench.agent.parsers import OutputParser, ParseResult

import dgp  # noqa: E402  (host-only)

_PRED_RE = re.compile(
    r"IMPUTE_PRED\s+env=(\S+)\s+seed=(\d+)\s+rows=(\d+)\s+cols=(\d+)\s+X_imputed=(\S+)"
)


class Parser(OutputParser):
    """Parser for ml-missing-data-imputation (predict-then-score)."""

    def parse(self, cmd_label: str, raw_output: str) -> ParseResult:
        metrics: dict = {}
        feedback_lines = []

        for m in _PRED_RE.finditer(raw_output):
            env, seed_s, rows_s, cols_s, payload = m.groups()
            # The agent only ever sees an opaque token for the dataset; map
            # the real test-command label to that token to match the line.
            if env != dgp.opaque_label(cmd_label):
                continue
            env = cmd_label  # use the real dataset for host-side scoring
            seed = int(seed_s)
            rows, cols = int(rows_s), int(cols_s)
            try:
                X_imputed = np.frombuffer(
                    base64.b64decode(payload), dtype=np.float64
                ).reshape(rows, cols)
            except Exception:
                continue
            X_scaled, y, mask, task_type = dgp.truth(env, seed)
            if X_imputed.shape != X_scaled.shape:
                continue
            rmse = float(dgp.compute_imputation_rmse(X_scaled, X_imputed, mask))
            downstream = float(
                dgp.compute_downstream_score(X_imputed, y, task_type, seed=seed)
            )
            metrics[f"rmse_{cmd_label}"] = rmse
            metrics[f"downstream_score_{cmd_label}"] = downstream
            feedback_lines.append(f"  rmse: {rmse:.6f}")
            feedback_lines.append(f"  downstream_score: {downstream:.6f}")

        if not metrics:
            return ParseResult(feedback=raw_output[-3000:], metrics={})

        feedback = f"Test results ({cmd_label}):\n" + "\n".join(feedback_lines)
        return ParseResult(feedback=feedback, metrics=metrics)
