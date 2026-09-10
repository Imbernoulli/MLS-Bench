"""Mid-edit operations for the llm-rl-reward-normalization task.

Applied to the verl workspace after pre_edit, before the agent starts.

1. Creates ``custom_reward_normalization.py`` — the agent-editable module
   implementing the reward normalization function.
2. Injects an import into ``main_ppo.py`` so the module loads at startup
   (ensures ``verl.trainer.ppo.custom_reward_normalization`` is importable
   in the driver process).
3. Patches ``ray_trainer.py`` to call ``normalize_rewards`` immediately
   after ``token_level_scores`` is set from the reward manager, and
   BEFORE any KL-in-reward penalty / advantage computation.  This places
   our edit UPSTREAM of ``compute_advantage`` (where llm-rl-advantage
   operates), so the two tasks are cleanly disjoint.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# Import so the module is loaded in the driver (and transitively by the
# line we inject into ray_trainer.py, which is what actually calls it).
_IMPORT_LINE = (
    "import verl.trainer.ppo.custom_reward_normalization  # noqa: F401  custom reward normalization hook"
)

# Snippet inserted into ray_trainer.py right after
#   batch.batch["token_level_scores"] = reward_tensor
# (currently line 1443).  Indentation = 24 spaces to match the surrounding
# ``with marked_timer("adv", ...):`` block inside the train step.
_NORMALIZE_HOOK = (
    '                        from verl.trainer.ppo.custom_reward_normalization import normalize_rewards as _custom_normalize_rewards\n'
    '                        _rn_index = batch.non_tensor_batch.get("uid", None)\n'
    '                        _rn_mask = batch.batch.get("response_mask", None)\n'
    '                        if _rn_mask is None:\n'
    '                            from verl.trainer.ppo.ray_trainer import compute_response_mask as _rn_compute_mask\n'
    '                            _rn_mask = _rn_compute_mask(batch)\n'
    '                            batch.batch["response_mask"] = _rn_mask\n'
    '                        batch.batch["token_level_scores"] = _custom_normalize_rewards(\n'
    '                            token_level_scores=batch.batch["token_level_scores"],\n'
    '                            response_mask=_rn_mask,\n'
    '                            index=_rn_index,\n'
    '                            config=self.config.algorithm,\n'
    '                        )'
)

OPS = [
    # 1. Create the agent-editable reward-normalization file.
    {
        "op": "create",
        "file": "verl/verl/trainer/ppo/custom_reward_normalization.py",
        "content": _CUSTOM_PY,
    },
    # 2. Register the module by importing it in main_ppo.py (after the
    #    last stdlib/third-party import block).
    {
        "op": "insert",
        "file": "verl/verl/trainer/main_ppo.py",
        "after_line": 32,
        "content": _IMPORT_LINE,
    },
    # 3. Patch ray_trainer.py to call normalize_rewards() right after
    #    ``token_level_scores`` is populated from the reward manager and
    #    BEFORE the KL-in-reward branch or compute_advantage().
    {
        "op": "insert",
        "file": "verl/verl/trainer/ppo/ray_trainer.py",
        "after_line": 1443,
        "content": _NORMALIZE_HOOK,
    },
]
