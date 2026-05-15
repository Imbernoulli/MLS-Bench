"""Mid-edit operations for the llm-rl-advantage task.

Applied to the verl workspace after pre_edit, before the agent starts.
1. Creates custom_advantage.py — the agent's editable advantage estimator.
2. Injects an import into main_ppo.py so the @register_adv_est("custom")
   decorator fires at startup.
3. Injects additional kwargs (old_log_probs, ref_log_probs) into the
   compute_advantage() else branch in ray_trainer.py so the custom
   estimator can access policy log-probability information.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# Import line to inject into main_ppo.py (after line 32, the last existing import).
# This ensures the @register_adv_est("custom") decorator runs when main_ppo is executed.
_IMPORT_LINE = "import verl.trainer.ppo.custom_advantage  # noqa: F401  register custom adv estimator"

# Additional kwargs to inject into ray_trainer.py compute_advantage() else branch.
# Inserted after line 213 (after the OPTIMAL_TOKEN_BASELINE block, before the
# "# calculate advantage estimator" comment at line 214).
# Indentation: 8 spaces (inside the else block of compute_advantage).
_EXTRA_KWARGS_SNIPPET = (
    '        if "old_log_probs" in data.batch:\n'
    '            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]\n'
    '        if "ref_log_probs" in data.batch:\n'
    '            adv_kwargs["ref_log_probs"] = data.batch["ref_log_probs"]'
)

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    # 1. Create the editable advantage estimator file
    {
        "op": "create",
        "file": "verl/verl/trainer/ppo/custom_advantage.py",
        "content": _CUSTOM_PY,
    },
    # 2. Inject import into main_ppo.py so the custom estimator is registered
    {
        "op": "insert",
        "file": "verl/verl/trainer/main_ppo.py",
        "after_line": 32,
        "content": _IMPORT_LINE,
    },
    # 3. Inject old_log_probs/ref_log_probs kwargs into ray_trainer.py
    #    else branch (after line 213, before "# calculate advantage estimator")
    {
        "op": "insert",
        "file": "verl/verl/trainer/ppo/ray_trainer.py",
        "after_line": 213,
        "content": _EXTRA_KWARGS_SNIPPET,
    },
]
