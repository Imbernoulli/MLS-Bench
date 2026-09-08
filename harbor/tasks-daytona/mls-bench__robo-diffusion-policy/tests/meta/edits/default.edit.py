"""Default baseline — unmodified template.

The template is a line-for-line port of cleandiffuser's dql_d4rl_mujoco.py
(diffusion actor + twin Q critic, BC + Q-learning loss). Running it as-is is
the paper-level Diffusion Q-Learning configuration.
"""

_FILE = "CleanDiffuser/pipelines/custom_policy.py"

OPS = []
