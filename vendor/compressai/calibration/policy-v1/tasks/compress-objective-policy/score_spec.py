"""Measured replay calibration from three accepted 192-case family proofs."""
from mlsbench.scoring.dsl import *

_CALIBRATION = {'full': {'factorized': 28.292905783, 'hyperprior_scale': 29.625689583, 'meanscale': 29.92153342, 'midpoint': 29.625689583, 'scale': 0.961400289418575}, 'low': {'factorized': 31.666022344, 'hyperprior_scale': 32.738386976, 'meanscale': 33.071423019, 'midpoint': 32.738386976, 'scale': 0.7735475683055586}, 'mid': {'factorized': 29.275551185, 'hyperprior_scale': 30.523619307, 'meanscale': 30.813332683, 'midpoint': 30.523619307, 'scale': 0.9002908451505005}, 'high': {'factorized': 23.937143822, 'hyperprior_scale': 25.615062465, 'meanscale': 25.879844558, 'midpoint': 25.615062465, 'scale': 1.2103624526356207}}

for _setting in ('full', 'low', 'mid', 'high'):
    _metric = f"rd6_{_setting}"
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
