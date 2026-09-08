"""
Mid-edit operations for the cv-dbm-scheduler task.
Creates the 'blank' function from the template.
"""
from pathlib import Path
try:
    from .custom_template import _TEMPLATE
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from custom_template import _TEMPLATE

_FILE = "dbim-codebase/ddbm/karras_diffusion.py"

# NOTE: line numbers are POST-pre_edit. The pkg_config pre_edit op at
# karras_diffusion.py:275-279 expands 5 → 14 lines (+9 shift), so what
# was originally lines 301-311 (`get_sigmas_karras` + `get_sigmas_uniform`)
# now lives at lines 310-320 in the workspace copy. The shift is the same
# under docker — pre_edit is shared between runtimes.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 310,
        "end_line": 320,
        "content": _TEMPLATE,
    },
]
