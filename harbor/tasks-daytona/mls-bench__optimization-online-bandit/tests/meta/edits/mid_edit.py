"""Mid-edit operations for the opt-online-bandit task.

Applied to the SMPyBandits workspace after pre_edit, before the agent starts.
Creates two files:

* ``mlsb_bandit_harness.py`` -- the fixed driver (read-only).  It owns the
  environments, draws each run's instance from fresh OS entropy, and plays it
  against the policy running in a separate interpreter.
* ``custom_bandit.py`` -- the agent's algorithm file; only the
  ``BanditPolicy`` region is editable.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()
_HARNESS_PY = (Path(__file__).parent / "harness.py").read_text()

# -- Mid-edit operations --------------------------------------------------

OPS = [
    {
        "op": "create",
        "file": "SMPyBandits/mlsb_bandit_harness.py",
        "content": _HARNESS_PY,
    },
    {
        "op": "create",
        "file": "SMPyBandits/custom_bandit.py",
        "content": _CUSTOM_PY,
    },
]
