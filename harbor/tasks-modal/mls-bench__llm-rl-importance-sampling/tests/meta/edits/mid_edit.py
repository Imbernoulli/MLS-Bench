"""Mid-edit operations for the llm-rl-importance-sampling task.

Applied to the verl workspace after pre_edit, before the agent starts.
1. Creates custom_policy_loss.py — the agent's editable policy-loss file.
2. Injects the registration import into main_ppo.py (driver process).
3. Injects the same import into dp_actor.py and megatron_actor.py so Ray
   FSDP/Megatron worker processes also trigger the @register_policy_loss
   decorator. Without this, get_policy_loss_fn("custom") raises inside the
   actor worker because the custom module is never imported there — the
   driver-side main_ppo import doesn't propagate to Ray worker processes.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

_IMPORT_LINE = "import verl.trainer.ppo.custom_policy_loss  # noqa: F401  register custom policy loss"

OPS = [
    # 1. Create the editable policy-loss file
    {
        "op": "create",
        "file": "verl/verl/trainer/ppo/custom_policy_loss.py",
        "content": _CUSTOM_PY,
    },
    # 2. Inject import into main_ppo.py (driver)
    {
        "op": "insert",
        "file": "verl/verl/trainer/main_ppo.py",
        "after_line": 32,
        "content": _IMPORT_LINE,
    },
    # 3. Inject import into dp_actor.py (FSDP Ray worker — where
    #    get_policy_loss_fn is actually called; the driver-side import in
    #    main_ppo does not propagate into Ray worker processes).
    {
        "op": "insert",
        "file": "verl/verl/workers/actor/dp_actor.py",
        "after_line": 30,
        "content": _IMPORT_LINE,
    },
    # 4. Same injection for Megatron actor (belt-and-suspenders; not used by
    #    our current train.sh but makes the task safe to repurpose).
    {
        "op": "insert",
        "file": "verl/verl/workers/actor/megatron_actor.py",
        "after_line": 40,
        "content": _IMPORT_LINE,
    },
]
