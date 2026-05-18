"""Mid-edit operations for the quant-graph-stock task.

Applied to the qlib workspace after pre_edit, before the agent starts.
Creates custom_model.py (the agent's editable model file) from custom_template.py,
and workflow_config.yaml (the qlib workflow configuration) from workflow_config.yaml.
"""

from pathlib import Path

_TASK_DIR = Path(__file__).parent

_MODEL_TEMPLATE = (_TASK_DIR / "custom_template.py").read_text()
_WORKFLOW_YAML = (_TASK_DIR / "workflow_config.yaml").read_text()

# -- Mid-edit operations --------------------------------------------------

OPS = [
    {
        "op": "create",
        "file": "qlib/custom_model.py",
        "content": _MODEL_TEMPLATE,
    },
    {
        "op": "create",
        "file": "qlib/workflow_config.yaml",
        "content": _WORKFLOW_YAML,
    },
]
