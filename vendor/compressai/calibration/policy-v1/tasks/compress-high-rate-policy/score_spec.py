"""Measured replay calibration from three accepted 192-case family proofs."""
from mlsbench.scoring.dsl import *

_CALIBRATION = {'full': {'factorized': 19.403569749, 'hyperprior_scale': 22.053442793, 'meanscale': 22.615093522, 'midpoint': 22.053442793, 'scale': 1.9114793497820708}, 'low': {'factorized': 26.329692243, 'hyperprior_scale': 28.795191031, 'meanscale': 29.695531341, 'midpoint': 28.795191031, 'scale': 1.778481437382676}, 'mid': {'factorized': 21.638495192, 'hyperprior_scale': 24.136817901, 'meanscale': 24.804371941, 'midpoint': 24.136817901, 'scale': 1.8021588914072904}, 'high': {'factorized': 10.24252181, 'hyperprior_scale': 13.228319448, 'meanscale': 13.345377284, 'midpoint': 13.228319448, 'scale': 2.153797722720291}}

for _setting in ('full', 'low', 'mid', 'high'):
    _metric = f"highq_rd12_{_setting}"
    _values = _CALIBRATION[_setting]
    term(
        _metric,
        col(_metric).higher().id().sigmoid(
            ref=const(_values['midpoint']),
            scale=_values['scale'],
        ),
    )
    setting(_setting, weighted_mean((_metric, 1.0)))

task(gmean("full", "low", "mid", "high"))
