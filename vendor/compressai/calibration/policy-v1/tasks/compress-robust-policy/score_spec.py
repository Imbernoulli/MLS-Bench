"""Measured replay calibration from three accepted 192-case family proofs."""
from mlsbench.scoring.dsl import *

_CALIBRATION = {'full': {'factorized': 19.534655865, 'hyperprior_scale': 21.127698656, 'meanscale': 21.642780775, 'midpoint': 21.127698656, 'scale': 1.1491374672498056}, 'low': {'factorized': 25.416632695, 'hyperprior_scale': 27.266615736, 'meanscale': 28.037273849, 'midpoint': 27.266615736, 'scale': 1.3344806794896906}, 'mid': {'factorized': 21.385139808, 'hyperprior_scale': 23.018126143, 'meanscale': 23.595971232, 'midpoint': 23.018126143, 'scale': 1.1779506436719704}, 'high': {'factorized': 11.802195091, 'hyperprior_scale': 13.098354091, 'meanscale': 13.295097244, 'midpoint': 13.098354091, 'scale': 0.9349810807517985}}

for _setting in ('full', 'low', 'mid', 'high'):
    _metric = f"rd18_{_setting}"
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
