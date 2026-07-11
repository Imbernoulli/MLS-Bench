"""Measured replay calibration from three accepted 192-case family proofs."""
from mlsbench.scoring.dsl import *

_CALIBRATION = {'full': {'factorized': 31.451158754, 'hyperprior_scale': 33.273111356, 'meanscale': 33.511429558, 'midpoint': 33.273111356, 'scale': 1.3142609918200714}, 'low': {'factorized': 32.715106817, 'hyperprior_scale': 32.621043836, 'meanscale': 32.297646774, 'midpoint': 32.621043836, 'scale': 0.23328166879272885}, 'mid': {'factorized': 32.218718299, 'hyperprior_scale': 33.422160567, 'meanscale': 33.314374858, 'midpoint': 33.422160567, 'scale': 0.8681000960198808}, 'high': {'factorized': 26.194669457, 'hyperprior_scale': 27.681708278, 'meanscale': 27.912470901, 'midpoint': 27.681708278, 'scale': 1.0726717663330345}}

for _setting in ('full', 'low', 'mid', 'high'):
    _metric = f"target_utility_{_setting}"
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
