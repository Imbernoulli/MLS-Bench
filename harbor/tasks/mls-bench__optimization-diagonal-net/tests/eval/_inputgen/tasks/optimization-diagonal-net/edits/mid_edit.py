"""Mid-edit: create the opt-diagonal-net scaffold inside the RAIN workspace.

Creates the agent-editable ``custom_optimizer.py`` and the FIXED
``fixed_benchmark.py`` driver, AND pre-generates the observable sparse-recovery
problems (X_train, y_train, X_test, y_test) the driver loads at run time — one
per (script CLI-arg combo) x search-seed x (smoke / non-smoke grid).

The data-generating process that also builds the true sparse vector w_star lives
ONLY in ``holdout/optimization-diagonal-net/dgp.py`` (host-side, never
bind-mounted). Here we import it host-side and write ONLY the observable problem
into the workspace as base64 text — the agent never sees the generator or
w_star. Inputs are byte-identical to the originals, so honest results are
unchanged.
"""

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent
_TASK_DIR = _DIR.parent
_PROJECT_ROOT = _TASK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "holdout" / "optimization-diagonal-net"))
import dgp  # host-only

_CUSTOM_PY = (_DIR / "custom_template.py").read_text()
_FIXED_PY = (_DIR / "fixed_benchmark.py").read_text()


# --- search-protocol constants (mirror fixed_benchmark.SearchConfig / run_cli) ---
_FULL_GRID = (50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600)
_FULL_NUM_SEEDS = 5
_SMOKE_GRID = (100, 200, 400, 800)
_SMOKE_NUM_SEEDS = 2


def _parse_script_args(text):
    """Extract --flag value pairs from a script body into a dict (last wins)."""
    toks = text.replace("\\\n", " ").split()
    args = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t[2:]
            has_val = i + 1 < len(toks) and not toks[i + 1].startswith("--")
            args[key] = toks[i + 1] if has_val else None
            i += 2 if has_val else 1
        else:
            i += 1
    return args


def _resolved_grid(base_grid, grid_max):
    """Mirror run_cli's --grid-max handling."""
    if grid_max is None:
        return tuple(base_grid)
    base = list(base_grid)
    extended = [v for v in base if v <= grid_max]
    if not extended or extended[-1] < grid_max:
        extended.append(grid_max)
    return tuple(extended)


def _input_key(dim, sparsity, n_max_train, n_test, seed):
    # Must match fixed_benchmark.py _input_key() exactly.
    return (
        f"d{int(dim)}_k{int(sparsity)}"
        f"_nmax{int(n_max_train)}_nt{int(n_test)}_seed{int(seed)}"
    )


def _encode_problem(dim, sparsity, n_max_train, n_test, seed):
    X_train, y_train, X_test, y_test = dgp.gen_input(
        dim=dim, sparsity=sparsity, n_max_train=n_max_train,
        n_test=n_test, seed=seed,
    )
    buf = io.BytesIO()
    np.savez(
        buf,
        X_train=np.ascontiguousarray(X_train, dtype=np.float64),
        y_train=np.ascontiguousarray(y_train, dtype=np.float64),
        X_test=np.ascontiguousarray(X_test, dtype=np.float64),
        y_test=np.ascontiguousarray(y_test, dtype=np.float64),
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


try:
    _cfg = json.loads((_TASK_DIR / "config.json").read_text())
    _MASTER_SEEDS = _cfg.get("seeds") or [42]
except Exception:
    _MASTER_SEEDS = [42]


OPS = [
    {
        "op": "create",
        "file": "RAIN/opt_diagonal_net/custom_optimizer.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "RAIN/opt_diagonal_net/fixed_benchmark.py",
        "content": _FIXED_PY,
    },
]

_seen = set()
for _script in sorted((_TASK_DIR / "scripts").glob("*.sh")):
    _a = _parse_script_args(_script.read_text())
    if _a.get("dim") is None or _a.get("sparsity") is None:
        continue
    _dim = int(_a["dim"])
    _sparsity = int(_a["sparsity"])
    _n_test = int(_a["n-test"]) if _a.get("n-test") else 4096
    _grid_max = int(_a["grid-max"]) if _a.get("grid-max") else None

    for _master in _MASTER_SEEDS:
        for _base_grid, _num_seeds in (
            (_FULL_GRID, _FULL_NUM_SEEDS),
            (_SMOKE_GRID, _SMOKE_NUM_SEEDS),
        ):
            _grid = _resolved_grid(_base_grid, _grid_max)
            _n_max_train = max(_grid)
            for _seed in range(int(_master), int(_master) + _num_seeds):
                _key = _input_key(_dim, _sparsity, _n_max_train, _n_test, _seed)
                if _key in _seen:
                    continue
                _seen.add(_key)
                OPS.append({
                    "op": "create",
                    "file": f"RAIN/opt_diagonal_net/_inputs/{_key}.npz.b64",
                    "content": _encode_problem(
                        _dim, _sparsity, _n_max_train, _n_test, _seed,
                    ),
                })
