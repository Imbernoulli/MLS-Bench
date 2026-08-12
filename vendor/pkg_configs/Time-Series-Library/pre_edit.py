"""Pre-edit operations for the Time-Series-Library package.
Injects TRAIN_METRICS print statements into all experiment files so that
training metrics appear on stdout for parser consumption.
Also injects per-pattern SMAPE computation for short-term forecasting.
Also patches run.py so MLS-Bench's injected SEED is respected.
"""

_RUNPY_RESEED = (
    "    args = parser.parse_args()\n"
    "    fix_seed = args.seed\n"
    "    random.seed(fix_seed)\n"
    "    torch.manual_seed(fix_seed)\n"
    "    np.random.seed(fix_seed)"
)

_RUNPY_SEED_ARG = (
    "    parser.add_argument('--seed', type=int, default=fix_seed, help=\"Randomization seed\")"
)

_RUNPY_SEED_BOOTSTRAP = (
    "    fix_seed = int(os.environ.get('SEED', '2021'))\n"
    "    random.seed(fix_seed)\n"
    "    torch.manual_seed(fix_seed)\n"
    "    np.random.seed(fix_seed)"
)

# Long-term forecasting: inject after line 155 (after the Epoch/Steps/Loss print)
_LTF_TRAIN_METRICS = (
    '            print(f"TRAIN_METRICS epoch={epoch+1} '
    'train_loss={train_loss:.7f} vali_loss={vali_loss:.7f} '
    'test_loss={test_loss:.7f}", flush=True)'
)

# Short-term forecasting: inject after line 116 (after the Epoch/Steps/Loss print)
_STF_TRAIN_METRICS = (
    '            print(f"TRAIN_METRICS epoch={epoch+1} '
    'train_loss={train_loss:.7f} vali_loss={vali_loss:.7f}", flush=True)'
)

# Short-term forecasting: inject per-pattern SMAPE after line 206 (after 'test shape' print)
# Note: line 205 in original file shifts to 206 after the first STF insertion at line 116
_STF_SMAPE = (
    '        # Per-pattern metric computation (injected by pre_edit)\n'
    '        import numpy as _np\n'
    '        _forecast = preds[:, :, 0]\n'
    '        _actual = _np.array([t[:_forecast.shape[1]] for t in trues])\n'
    '        _smape = _np.mean(200.0 * _np.abs(_forecast - _actual) / '
    '(_np.abs(_forecast) + _np.abs(_actual) + 1e-8))\n'
    '        _mape = _np.mean(_np.abs((_actual - _forecast) / '
    '(_np.abs(_actual) + 1e-8)))\n'
    '        print(f"smape:{_smape:.6f}")\n'
    '        print(f"mape:{_mape:.6f}")'
)

# Short-term forecasting: race-safe creation of the shared ./m4_results/<model>/
# leaf (pristine lines 208-210). All three m4_* labels (plus every baseline /
# oracle run, and every seed) execute concurrently in one workspace, and the
# pristine `if not os.path.exists(folder_path): os.makedirs(folder_path)` is a
# check-then-create TOCTOU: the loser of the race dies with FileExistsError
# AFTER training completes, losing its metric.
_STF_MAKEDIRS_FIX = (
    "        os.makedirs(folder_path, exist_ok=True)"
)

# Imputation: inject after line 144 (after the Epoch/Steps/Loss print)
_IMP_TRAIN_METRICS = (
    '            print(f"TRAIN_METRICS epoch={epoch+1} '
    'train_loss={train_loss:.7f} vali_loss={vali_loss:.7f} '
    'test_loss={test_loss:.7f}", flush=True)'
)

# Anomaly detection: inject after line 116 (after the Epoch/Steps/Loss print)
_AD_TRAIN_METRICS = (
    '            print(f"TRAIN_METRICS epoch={epoch+1} '
    'train_loss={train_loss:.7f} vali_loss={vali_loss:.7f} '
    'test_loss={test_loss:.7f}", flush=True)'
)

# Classification: inject after line 134 (after the Epoch/Steps/Loss/Acc print)
_CLS_TRAIN_METRICS = (
    '            print(f"TRAIN_METRICS epoch={epoch+1} '
    'train_loss={train_loss:.3f} vali_loss={vali_loss:.3f} '
    'vali_acc={val_accuracy:.3f} test_acc={test_accuracy:.3f}", flush=True)'
)

# Fix numpy compatibility: M4 data has variable-length time series,
# newer numpy rejects ragged sequences in np.array().
# Use a plain list instead — timeseries is accessed by index anyway.
_M4_NUMPY_FIX = (
    '        training_values = [v[~np.isnan(v)] for v in\n'
    '             dataset.values[dataset.groups == self.seasonal_patterns]]  # split different frequencies\n'
)

OPS = [
    # Respect MLS-Bench-injected SEED in the package entrypoint.
    # Order is bottom-to-top within run.py so line shifts do not affect later ops.
    {
        "op": "replace",
        "file": "Time-Series-Library/run.py",
        "start_line": 155,
        "end_line": 155,
        "content": _RUNPY_RESEED,
    },
    {
        "op": "replace",
        "file": "Time-Series-Library/run.py",
        "start_line": 114,
        "end_line": 114,
        "content": _RUNPY_SEED_ARG,
    },
    {
        "op": "replace",
        "file": "Time-Series-Library/run.py",
        "start_line": 10,
        "end_line": 13,
        "content": _RUNPY_SEED_BOOTSTRAP,
    },
    # Fix M4 data loader numpy compatibility (must come before any line-shifting inserts)
    {
        "op": "replace",
        "file": "Time-Series-Library/data_provider/data_loader.py",
        "start_line": 364,
        "end_line": 366,
        "content": _M4_NUMPY_FIX,
    },
    # Long-term forecasting
    {
        "op": "insert",
        "file": "Time-Series-Library/exp/exp_long_term_forecasting.py",
        "after_line": 155,
        "content": _LTF_TRAIN_METRICS,
    },
    # Short-term forecasting - race-safe m4_results dir creation.
    # Placed BEFORE the two STF inserts below so its pristine coordinates
    # (209-210) are valid; it only shifts lines >210, so the inserts at
    # after_line=116 and after_line=206 (pristine 205, shifted +1 by the
    # first insert) keep their existing coordinates.
    {
        "op": "replace",
        "file": "Time-Series-Library/exp/exp_short_term_forecasting.py",
        "start_line": 209,
        "end_line": 210,
        "content": _STF_MAKEDIRS_FIX,
    },
    # Short-term forecasting - training metrics
    {
        "op": "insert",
        "file": "Time-Series-Library/exp/exp_short_term_forecasting.py",
        "after_line": 116,
        "content": _STF_TRAIN_METRICS,
    },
    # Short-term forecasting - per-pattern SMAPE
    {
        "op": "insert",
        "file": "Time-Series-Library/exp/exp_short_term_forecasting.py",
        "after_line": 206,
        "content": _STF_SMAPE,
    },
    # Imputation
    {
        "op": "insert",
        "file": "Time-Series-Library/exp/exp_imputation.py",
        "after_line": 144,
        "content": _IMP_TRAIN_METRICS,
    },
    # Anomaly detection
    {
        "op": "insert",
        "file": "Time-Series-Library/exp/exp_anomaly_detection.py",
        "after_line": 116,
        "content": _AD_TRAIN_METRICS,
    },
    # Classification
    {
        "op": "insert",
        "file": "Time-Series-Library/exp/exp_classification.py",
        "after_line": 134,
        "content": _CLS_TRAIN_METRICS,
    },
]
