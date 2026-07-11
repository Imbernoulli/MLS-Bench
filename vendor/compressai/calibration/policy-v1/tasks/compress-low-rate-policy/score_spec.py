"""Measured replay calibration from three accepted 192-case family proofs."""
from mlsbench.scoring.dsl import *

_CALIBRATION = {'full': {'factorized': 25.851848517, 'hyperprior_scale': 26.609710063, 'meanscale': 26.879284033, 'midpoint': 26.609710063, 'scale': 0.5466815470473222}, 'low': {'factorized': 28.880235848, 'hyperprior_scale': 29.650254003, 'meanscale': 30.002216071, 'midpoint': 29.650254003, 'scale': 0.5554506868064838}, 'mid': {'factorized': 26.699528031, 'hyperprior_scale': 27.472940999, 'meanscale': 27.729048389, 'midpoint': 27.472940999, 'scale': 0.5578995267464064}, 'high': {'factorized': 21.975781672, 'hyperprior_scale': 22.705935187, 'meanscale': 22.90658764, 'midpoint': 22.705935187, 'scale': 0.5266944275890737}}

for _setting in ('full', 'low', 'mid', 'high'):
    _metric = f"lowq_rd12_{_setting}"
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
