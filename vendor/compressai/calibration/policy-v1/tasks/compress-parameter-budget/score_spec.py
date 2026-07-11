"""Measured replay calibration from three accepted 192-case family proofs."""
from mlsbench.scoring.dsl import *

_CALIBRATION = {'full': {'factorized': 23.913780824, 'hyperprior_scale': 25.37669412, 'meanscale': 25.782157098, 'midpoint': 25.37669412, 'scale': 1.055268878694864}, 'low': {'factorized': 28.54132752, 'hyperprior_scale': 30.002501356, 'meanscale': 30.554348434, 'midpoint': 30.002501356, 'scale': 1.054014123536952}, 'mid': {'factorized': 25.330345496, 'hyperprior_scale': 26.770872725, 'meanscale': 27.204651958, 'midpoint': 26.770872725, 'scale': 1.0391207447719106}, 'high': {'factorized': 17.869669456, 'hyperprior_scale': 19.356708278, 'meanscale': 19.587470901, 'midpoint': 19.356708278, 'scale': 1.0726717670543822}}

for _setting in ('full', 'low', 'mid', 'high'):
    _metric = f"rd12_{_setting}"
    _values = _CALIBRATION[_setting]
    term(
        _metric,
        col(_metric).higher().id().sigmoid(
            ref=const(_values['midpoint']),
            scale=_values['scale'],
        ),
    )
    _constraint = f"mean_params_{_setting}"
    _constraint_values = {'target': 7603523.0, 'sharpness': 1.477507393556676e-07}
    term(
        _constraint,
        penalty_upper(
            col(_constraint).lower().id(),
            target=_constraint_values['target'],
            sharpness=_constraint_values['sharpness'],
        ),
    )
    setting(
        _setting,
        weighted_mean((_metric, 1.0)),
        constraints=[_constraint],
    )

task(gmean("full", "low", "mid", "high"))
