"""Mid-edit operations for ts-imputation.
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

_RUN_WITHHELD_PATCH = """\
    # MLS-Bench: the scored test rows of each dataset are withheld from this
    # workspace and mounted only for the final evaluation. When the data
    # directory carries the marker, the run reports the validation split in
    # the test split's place (training and early stopping are unchanged).
    if os.path.exists(os.path.join(args.root_path, '.mlsb_test_withheld')):
        print('NOTE: the test rows of this dataset are withheld from the workspace; '
              'the test metrics printed below are computed on the validation split.', flush=True)
        _mlsb_get_data = Exp._get_data

        def _mlsb_get_data_val(self, flag):
            return _mlsb_get_data(self, 'val' if flag == 'test' else flag)

        Exp._get_data = _mlsb_get_data_val

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
    {
        # After the replace above (+8 lines): line 209 is the blank line that
        # ends the Exp selection, right before `if args.is_training:`.
        "op": "insert",
        "file": "Time-Series-Library/run.py",
        "after_line": 209,
        "content": _RUN_WITHHELD_PATCH,
    },
]
