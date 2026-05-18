"""Default baseline — unmodified template.

The template already implements classifier guidance (CumRewClassifier with
CG weights from config: w_cg=0.3/0.007/0.0001 per env, w_cfg=0.0). Running it
as-is is the paper-level Diffuser configuration.
"""

_FILE = "CleanDiffuser/pipelines/custom_guidance.py"

OPS = []
