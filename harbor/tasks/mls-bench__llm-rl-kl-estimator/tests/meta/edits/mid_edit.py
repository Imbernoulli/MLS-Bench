"""Mid-edit operations for the llm-rl-kl-estimator task.

Applied to the verl workspace after pre_edit, before the agent starts.
1. Creates custom_kl_penalty.py — the agent's editable KL estimator.
2. Injects an import of the custom module into main_ppo.py (driver) so
   the module is imported on the Ray driver process.
3. Injects the same import into dp_actor.py so the Ray FSDP worker
   process (where ``kl_penalty(...)`` is actually invoked) runs the
   monkey-patch of ``core_algos.kl_penalty_forward`` that adds a
   ``"custom"`` dispatch branch.  Without this the worker would hit
   the fall-through ``raise NotImplementedError`` when it sees
   ``kl_loss_type=custom``.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

# Import line to inject into main_ppo.py (after the last existing import
# at line 32) and into dp_actor.py (after the core_algos import at line 30).
_IMPORT_LINE = "import verl.trainer.ppo.custom_kl_penalty  # noqa: F401  register custom KL estimator"

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    # 1. Create the editable KL-penalty file.
    {
        "op": "create",
        "file": "verl/verl/trainer/ppo/custom_kl_penalty.py",
        "content": _CUSTOM_PY,
    },
    # 2. Inject import into main_ppo.py so the module loads on the driver
    #    process (belt-and-suspenders; the real work happens in the Ray
    #    worker below, but this makes the task self-consistent if a user
    #    probes the driver).
    {
        "op": "insert",
        "file": "verl/verl/trainer/main_ppo.py",
        "after_line": 32,
        "content": _IMPORT_LINE,
    },
    # 3. Inject import into dp_actor.py (FSDP Ray worker — where
    #    ``kl_penalty(..., kl_penalty='custom')`` is invoked).  Must
    #    follow the existing ``from core_algos import ... kl_penalty``
    #    line (currently line 30) so the monkey-patch runs at import
    #    time and takes effect before the first training step.
    {
        "op": "insert",
        "file": "verl/verl/workers/actor/dp_actor.py",
        "after_line": 30,
        "content": _IMPORT_LINE,
    },
]
