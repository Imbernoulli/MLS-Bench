"""Mid-edit operations for causal-observational-nonlinear.

Creates the agent-editable ``bench/custom_algorithm.py`` and the FIXED
``bench/run_eval.py`` driver, AND pre-generates the observational matrices X
that the driver loads at run time — one per (script CLI-arg combo) x seed.

The nonlinear ANM data-generating process (which also produces the true
adjacency matrix) and the metrics live ONLY in
``holdout/causal-observational-nonlinear/dgp.py`` (host-side, never bind-mounted).
Here we import it host-side and write ONLY X into the workspace as base64 text —
the agent never sees the simulator, the true adjacency, or the metric. Inputs are
byte-identical to the originals, so honest results are unchanged.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
_TASK_DIR = _HERE.parent
_PROJECT_ROOT = _TASK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "holdout" / "causal-observational-nonlinear"))
import dgp  # host-only


def _parse_script_args(text):
    """Extract --flag value pairs from a script body into a dict (last wins)."""
    toks = text.replace("\\\n", " ").split()
    args = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t[2:]
            val = toks[i + 1] if i + 1 < len(toks) else None
            args[key] = val
            i += 2
        else:
            i += 1
    return args


# run_eval.py argparse defaults — mirror these so missing flags resolve identically.
# Note: er_prob default is 0.3 for this task (nonlinear ANM).
_DEFAULTS = {
    "graph_type": None,
    "n_nodes": None,
    "n_samples": None,
    "noise_type": None,
    "fn_type": "mixed",
    "er_prob": "0.3",
    "sf_m": "2",
    "seed": "42",
}


def _resolve(args):
    out = dict(_DEFAULTS)
    out.update({k: v for k, v in args.items() if v is not None})
    return out


def _seeds():
    try:
        cfg = json.loads((_TASK_DIR / "config.json").read_text())
        return cfg.get("seeds") or [42]
    except Exception:
        return [42]


def _input_key(graph_type, n_nodes, n_samples, noise_type, fn_type, er_prob, sf_m, seed):
    # Must match run_eval.py _input_key() exactly (argparse-stringified values).
    return (
        f"{graph_type}_n{int(n_nodes)}_s{int(n_samples)}"
        f"_{noise_type}_{fn_type}_p{float(er_prob)}_m{int(sf_m)}_seed{int(seed)}"
    )


def _encode_X(graph_type, n_nodes, n_samples, noise_type, fn_type, er_prob, sf_m, seed):
    X = dgp.gen_input(
        graph_type=graph_type, n_nodes=n_nodes, n_samples=n_samples,
        noise_type=noise_type, fn_type=fn_type, er_prob=er_prob, sf_m=sf_m, seed=seed,
    )
    X = np.ascontiguousarray(X, dtype=np.float64)
    return base64.b64encode(X.tobytes()).decode("ascii")


OPS = [
    {
        "op": "create",
        "file": "causal-learn/bench/run_eval.py",
        "content": (_HERE / "run_eval_template.py").read_text(),
    },
    {
        "op": "create",
        "file": "causal-learn/bench/custom_algorithm.py",
        "content": (_HERE / "custom_template.py").read_text(),
    },
]

_seen = set()
for _script in sorted((_TASK_DIR / "scripts").glob("*.sh")):
    _a = _resolve(_parse_script_args(_script.read_text()))
    if not _a.get("graph_type") or not _a.get("noise_type"):
        continue
    for _seed in _seeds():
        _key = _input_key(
            _a["graph_type"], _a["n_nodes"], _a["n_samples"], _a["noise_type"],
            _a["fn_type"], _a["er_prob"], _a["sf_m"], _seed,
        )
        if _key in _seen:
            continue
        _seen.add(_key)
        OPS.append({
            "op": "create",
            "file": f"causal-learn/bench/_inputs/{_key}.npy.b64",
            "content": _encode_X(
                _a["graph_type"], int(_a["n_nodes"]), int(_a["n_samples"]), _a["noise_type"],
                _a["fn_type"], float(_a["er_prob"]), int(_a["sf_m"]), int(_seed),
            ),
        })
