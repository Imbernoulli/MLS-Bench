"""Held-out-label guard for the qlib stock-prediction workflow (host-controlled).

This module is imported by ``run_workflow.py`` (the FIXED, non-editable entry
point) and is NOT part of the agent-editable surface. It closes the oracle-leak
vector in which the editable ``CustomModel.predict()`` reads the held-out
scoring label directly out of the ``DatasetH`` object it is handed.

Threat model (in scope)
-----------------------
``SignalRecord.generate()`` calls ``self.model.predict(self.dataset)`` and then,
*from the same dataset object*, ``generate_label(self.dataset)`` to fetch the
raw test-segment label (``segments='test', col_set='label',
data_key=DataHandlerLP.DK_R``) that IC / Rank IC are scored against. A cheating
``predict()`` can fetch that exact label and return it as the prediction
(IC == 1.0).

What this guard does
--------------------
It monkey-patches ``SignalRecord.generate`` so that ``model.predict(...)`` is
handed a *label-free view* of the dataset, while the scorer's
``generate_label`` and the downstream backtest keep operating on the untouched
original dataset. The label-free view:

  * shares the SAME feature columns / values (honest results are byte-identical),
  * has every ``label`` column NaN-ed out in the handler's cached frames
    (``_data`` / DK_R, ``_infer`` / DK_I, ``_learn`` / DK_L), so reading the
    label via ``prepare(...)`` OR directly via ``handler._data`` / ``_infer`` /
    ``_learn`` / ``fetch`` yields NaN,
  * has its ``data_loader`` neutered, so the label cannot be re-derived by
    re-loading the raw frames from disk through the handler.

KNOWN RESIDUAL VECTOR (out of this guard's reach -- see the task report):
The label ``Ref($close,-2)/Ref($close,-1)-1`` is a deterministic function of the
raw ``$close`` series, which is the same on-disk data the features are built
from. An adversarial ``predict()`` can therefore bypass the dataset object
entirely and reconstruct the label from the global ``qlib.data.D`` provider
against the bind-mounted ``cn_data``. Closing THAT requires a data-level change
(restricting the on-disk calendar visible during inference), not a per-task
dataset guard. This module closes only the "read the answer key out of the
object you were handed" vector, which is the in-scope threat.
"""

import copy

import numpy as np
import pandas as pd

# When False, the fit-guard prepare() patch passes labels through untouched
# (used only while the host-side scorer reads the held-out test label).
_GUARD_ACTIVE = True


def _null_label_columns(df):
    """Return a copy of ``df`` with any ``label`` columns set to NaN.

    Feature columns are left untouched (same values). Handles both the
    multi-level column frames produced by ``DataHandlerLP`` (outer level names
    the field group: ``feature`` / ``label``) and single-level label-only
    frames.
    """
    if df is None:
        return None
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        lab_mask = cols.get_level_values(0) == "label"
        if lab_mask.any():
            df = df.copy()
            df.loc[:, lab_mask] = np.nan
        return df
    # Single-level frame: if it carries a 'label' column, null it; otherwise
    # it is a feature-only frame and must be returned untouched.
    if "label" in [str(c) for c in cols]:
        df = df.copy()
        df.loc[:, :] = np.nan
    return df


class _BlockedLoader:
    """Stand-in for ``handler.data_loader`` inside ``predict()``.

    Refuses to re-load raw frames (which would re-materialize the held-out
    label from disk). Honest models never touch the loader directly -- they use
    ``dataset.prepare(col_set='feature', ...)`` -- so this does not affect
    legitimate predictions.
    """

    def load(self, *args, **kwargs):
        raise PermissionError(
            "Re-loading raw data is disabled inside predict(); fetch features "
            "via dataset.prepare(segment, col_set='feature', "
            "data_key=DataHandlerLP.DK_I)."
        )

    def __getattr__(self, name):
        raise PermissionError("data_loader is disabled inside predict().")


def _build_label_free_dataset(real_ds):
    """Shallow-copy ``real_ds`` + its handler, swapping the cached frames for
    label-nulled copies and neutering the loader.

    The original ``real_ds`` / handler / cached frames are left completely
    intact, so ``SignalRecord.generate_label`` and ``PortAnaRecord``'s backtest
    -- which run on the original dataset -- are unaffected.
    """
    handler = getattr(real_ds, "handler", None)
    if handler is None:
        return real_ds  # not a DatasetH-with-handler; nothing to guard

    new_handler = copy.copy(handler)  # shallow: keeps refs we then overwrite
    for attr in ("_data", "_infer", "_learn"):
        if hasattr(handler, attr):
            setattr(new_handler, attr, _null_label_columns(getattr(handler, attr)))
    if hasattr(new_handler, "data_loader"):
        try:
            new_handler.data_loader = _BlockedLoader()
        except Exception:
            pass

    new_ds = copy.copy(real_ds)
    new_ds.handler = new_handler
    return new_ds


def install():
    """Monkey-patch ``SignalRecord.generate`` to hand ``predict()`` a
    label-free dataset view. Idempotent."""
    from qlib.workflow.record_temp import SignalRecord
    from qlib.data.dataset import DatasetH
    from qlib.log import get_module_logger

    if getattr(SignalRecord, "_mlsbench_label_guarded", False):
        return
    logger = get_module_logger("mlsbench_label_guard")

    def generate(self, **kwargs):
        real_ds = self.dataset
        if isinstance(real_ds, DatasetH):
            safe_ds = _build_label_free_dataset(real_ds)
        else:
            safe_ds = real_ds
        # prediction on the label-free view
        pred = self.model.predict(safe_ds)
        if isinstance(pred, pd.Series):
            pred = pred.to_frame("score")
        self.save(**{"pred.pkl": pred})
        logger.info(
            "Signal record 'pred.pkl' has been saved (label-free predict view)."
        )
        # scoring label from the ORIGINAL, untouched dataset. Turn the
        # fit-guard label masking OFF while the host-side scorer reads the
        # held-out test label (the editable model never runs here).
        if isinstance(real_ds, DatasetH):
            global _GUARD_ACTIVE
            _prev = _GUARD_ACTIVE
            _GUARD_ACTIVE = False
            try:
                raw_label = self.generate_label(real_ds)
            finally:
                _GUARD_ACTIVE = _prev
            self.save(**{"label.pkl": raw_label})

    SignalRecord.generate = generate
    SignalRecord._mlsbench_label_guarded = True

    _install_fit_label_guard()


def _install_fit_label_guard():
    """Mask the held-out TEST-segment label in ``DatasetH.prepare`` so the
    editable ``CustomModel.fit()`` (which runs on the un-guarded dataset, before
    the predict view exists) cannot read the test label out of the dataset object
    -- e.g. ``dataset.prepare('test', col_set=['feature','label'])`` -- and stash
    it for replay in ``predict()`` (IC == 1.0).

    Only labels for rows inside the *test* segment date range are NaN-ed; train /
    valid labels pass through untouched. Honest fits are unaffected (verified:
    lgbm prepares only train/valid; lstm / transformer unpack df_test but never
    use it). The host-side scorer reads the real test label by toggling
    ``_GUARD_ACTIVE`` off (see ``generate`` above). Idempotent.
    """
    from qlib.data.dataset import DatasetH

    if getattr(DatasetH, "_mlsbench_fit_guarded", False):
        return
    _orig_prepare = DatasetH.prepare

    def _nan_test_labels(obj, test_seg):
        if isinstance(obj, (list, tuple)):
            return type(obj)(_nan_test_labels(x, test_seg) for x in obj)
        df = obj
        if df is None or not hasattr(df, "columns") or not hasattr(df, "index"):
            return df
        cols = df.columns
        if isinstance(cols, pd.MultiIndex):
            lab_mask = cols.get_level_values(0) == "label"
            if not lab_mask.any():
                return df
            lab_cols = cols[lab_mask]
        else:
            if "label" not in [str(c) for c in cols]:
                return df
            lab_cols = None
        idx = df.index
        if isinstance(idx, pd.MultiIndex):
            names = list(idx.names or [])
            dt = idx.get_level_values("datetime") if "datetime" in names \
                else idx.get_level_values(0)
        else:
            dt = idx
        try:
            start, end = pd.Timestamp(test_seg[0]), pd.Timestamp(test_seg[1])
        except Exception:
            return df
        in_test = np.asarray((dt >= start) & (dt <= end))
        if not bool(in_test.any()):
            return df
        df = df.copy()
        if lab_cols is not None:
            df.loc[in_test, lab_cols] = np.nan
        else:
            df.loc[in_test, :] = np.nan
        return df

    def prepare(self, *args, **kwargs):
        df = _orig_prepare(self, *args, **kwargs)
        if not _GUARD_ACTIVE:
            return df
        segs = getattr(self, "segments", None)
        test_seg = segs.get("test") if isinstance(segs, dict) else None
        if test_seg is None:
            return df
        return _nan_test_labels(df, test_seg)

    DatasetH.prepare = prepare
    DatasetH._mlsbench_fit_guarded = True
