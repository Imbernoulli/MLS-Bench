"""Measured CompressAI-1.2.8 Kodak24 q1..8 calibration.

The official scale-hyperprior is the 0.5 midpoint for each required
setting. Logistic width is the larger measured distance to the official
factorized or mean-scale anchor divided by ln(4), so the farther extreme
maps to 0.2 or 0.8 without a positive floor or fallback.
"""
from math import log

from mlsbench.scoring.dsl import *

_CALIBRATION = {'full': {'weak_factorized': 23.913780824, 'native_hyperprior_scale': 25.37669412, 'strong_meanscale': 25.782157098, 'ref': 25.37669412, 'scale': 1.055268878694864}, 'low': {'weak_factorized': 28.54132752, 'native_hyperprior_scale': 30.002501356, 'strong_meanscale': 30.554348434, 'ref': 30.002501356, 'scale': 1.054014123536952}, 'mid': {'weak_factorized': 25.330345496, 'native_hyperprior_scale': 26.770872725, 'strong_meanscale': 27.204651958, 'ref': 26.770872725, 'scale': 1.0391207447719106}, 'high': {'weak_factorized': 17.869669456, 'native_hyperprior_scale': 19.356708278, 'strong_meanscale': 19.587470901, 'ref': 19.356708278, 'scale': 1.0726717670543822}}

for _setting, _values in _CALIBRATION.items():
    _name = f"mean_rd_utility_{_setting}"
    term(
        _name,
        col(_name).higher().id().sigmoid(
            ref=const(_values['native_hyperprior_scale']),
            scale=_values['scale'],
        ),
    )
    setting(_setting, weighted_mean((_name, 1.0)))

task(gmean('full', 'low', 'mid', 'high'))
