"""Official G-PBGD baseline for optimization-bilevel.

Reference:
- G-PBGD/data_hyper_clean_gpbgd.py
"""

_FILE = "penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py"

_CONTENT = '''\
TOY_HPARAMS = {
    "gams": (10.0,),
    "alpha0": 0.1,
}


HYPERCLEAN_HPARAMS = {
    "linear": {
        "lrx": 0.3,
        "lry": 0.5,
        "gamma_init": 0.0,
        "gamma_max": 37.0,
        "gamma_argmax_step": 5_000,
        "outer_itr": 40_000,
        "reg": 0.0,
        "eval_interval": 10,
    },
    "mlp": {
        "lrx": 0.5,
        "lry": 0.5,
        "gamma_init": 0.0,
        "gamma_max": 37.0,
        "gamma_argmax_step": 30_000,
        "outer_itr": 50_000,
        "reg": 0.0,
        "eval_interval": 10,
    },
}


def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
    return run_g_pbgd(state, hparams, grad_fns)
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
