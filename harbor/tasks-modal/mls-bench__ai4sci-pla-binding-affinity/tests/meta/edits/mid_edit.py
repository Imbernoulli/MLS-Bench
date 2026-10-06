"""Mid-edit operations for pla-binding-affinity.
Creates EHIGN_PLA/custom_pla.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

OPS = [
    {
        "op": "create",
        "file": "EHIGN_PLA/custom_pla.py",
        "content": _CUSTOM_PY,
    },
    # The upstream repo ships the PDBbind test sets' pdbid + affinity labels as
    # CSVs; nothing here reads them, and the test labels are withheld from the
    # workspace (they reach the eval from tests/ only), so drop them.
    {"op": "remove_path", "file": "EHIGN_PLA/data/test2013.csv"},
    {"op": "remove_path", "file": "EHIGN_PLA/data/test2016.csv"},
    {"op": "remove_path", "file": "EHIGN_PLA/data/test2019.csv"},
]
