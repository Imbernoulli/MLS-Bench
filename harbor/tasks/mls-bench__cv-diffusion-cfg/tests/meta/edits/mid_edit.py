"""Mid-edit operations for cv-diffusion-cfg task.

Applied to the CFGpp-main workspace after pre_edit, before the agent starts.
Replaces BaseDDIMCFGpp in both latent_diffusion.py and latent_sdxl.py with
custom templates, removing the baseline implementations so the agent must
implement them.
Also creates batch_eval.py for efficient single-process evaluation.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_TEMPLATE = _TEMPLATE_PATH.read_text()

_SDXL_TEMPLATE_PATH = Path(__file__).parent / "custom_template_sdxl.py"
_SDXL_CUSTOM_TEMPLATE = _SDXL_TEMPLATE_PATH.read_text()

_BATCH_EVAL_PATH = Path(__file__).parent / "batch_eval.py"
_BATCH_EVAL = _BATCH_EVAL_PATH.read_text()

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "replace",
        "file": "CFGpp-main/latent_diffusion.py",
        "start_line": 621,
        "end_line": 679,
        "content": _CUSTOM_TEMPLATE,
    },
    {
        "op": "replace",
        "file": "CFGpp-main/latent_sdxl.py",
        "start_line": 713,
        "end_line": 755,
        "content": _SDXL_CUSTOM_TEMPLATE,
    },
    {
        "op": "create",
        "file": "CFGpp-main/batch_eval.py",
        "content": _BATCH_EVAL,
    },
]
