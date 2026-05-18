from pathlib import Path
try:
    from .custom_template import _TEMPLATE
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from custom_template import _TEMPLATE

_FILE = "dbim-codebase/ddbm/karras_diffusion.py"

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 448,
        "end_line": 518,
        "content": _TEMPLATE,
    },
]
