"""Official RHG / ITD baseline for optimization-bilevel.

Reference:
- RHG/data_hyper_clean_rhg.py
"""

_FILE = "penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py"

_CONTENT = '''\
TOY_HPARAMS = {
    "gams": (10.0,),
    "alpha0": 0.1,
}


HYPERCLEAN_HPARAMS = {
    "linear": {
        "lr": 0.001,
        "lr_inner": 0.1,
        "outer_itr": 100,
        "T": 500,
        "K": 500,
        "reg": 0.0,
        "eval_interval": 1,
    },
    "mlp": {
        "lr": 0.001,
        "lr_inner": 0.4,
        "outer_itr": 100,
        "T": 500,
        "K": 500,
        "reg": 0.0,
        "eval_interval": 1,
    },
}


def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
    return run_rhg_family(state, hparams, grad_fns)
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 227,
        "end_line": 262,
        "content": _CONTENT,
    },
]
