"""Mid-edit operations for pde-design-solver.
Creates models/Custom.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()
_SEED_ARG = "parser.add_argument('--seed', type=int, default=int(os.environ.get(\"SEED\", \"42\")), help='random seed')\n"
_SEED_INIT = """\
seed = args.seed
os.environ["PYTHONHASHSEED"] = str(seed)
import random
import torch
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
"""

OPS = [
    {
        "op": "create",
        "file": "Neural-Solver-Library/models/Custom.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "insert",
        "file": "Neural-Solver-Library/run.py",
        "after_line": 24,
        "content": _SEED_ARG,
    },
    {
        "op": "insert",
        "file": "Neural-Solver-Library/run.py",
        "after_line": 81,
        "content": _SEED_INIT,
    },
]
