"""Official V-PBGD baseline for optimization-bilevel.

Reference:
- V-PBGD/toy/toy.py
- V-PBGD/data-hyper-cleaning/data_hyper_clean.py
"""

_FILE = "penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py"

_CONTENT = '''\
TOY_HPARAMS = {
    "gams": (10.0,),
    "alpha0": 0.1,
}


HYPERCLEAN_HPARAMS = {
    "linear": {
        "lrx": 0.1,
        "lry": 0.1,
        "lr_inner": 0.01,
        "gamma_init": 0.0,
        "gamma_max": 0.2,
        "gamma_argmax_step": 30_000,
        "outer_itr": 40_000,
        "inner_itr": 1,
        "reg": 0.0,
        "eval_interval": 10,
    },
    "mlp": {
        "lrx": 0.1,
        "lry": 0.01,
        "lr_inner": 0.01,
        "gamma_init": 0.0,
        "gamma_max": 0.1,
        "gamma_argmax_step": 10_000,
        "outer_itr": 80_000,
        "inner_itr": 1,
        "reg": 0.0,
        "eval_interval": 10,
    },
}


def algorithm(state: dict, hparams: dict, grad_fns: dict) -> dict:
    return run_v_pbgd(state, hparams, grad_fns)
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
