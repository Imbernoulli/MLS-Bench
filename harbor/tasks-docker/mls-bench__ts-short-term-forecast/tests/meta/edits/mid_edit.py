"""Mid-edit operations for ts-short-term-forecast.
Creates models/Custom.py from template.
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()
_RUN_SEED_PATCH = """\
    fix_seed = int(os.environ.get("SEED", "42"))
    if "--seed" in os.sys.argv:
        try:
            fix_seed = int(os.sys.argv[os.sys.argv.index("--seed") + 1])
        except (IndexError, ValueError) as exc:
            raise ValueError("--seed must be followed by an integer") from exc
    os.environ["PYTHONHASHSEED"] = str(fix_seed)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fix_seed)
    np.random.seed(fix_seed)
"""

OPS = [
    {
        "op": "create",
        "file": "Time-Series-Library/models/Custom.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "replace",
        "file": "Time-Series-Library/run.py",
        "start_line": 10,
        "end_line": 13,
        "content": _RUN_SEED_PATCH,
    },
]
