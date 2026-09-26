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

Causality check (look-ahead)
----------------------------
The label ``Ref($close,-2)/Ref($close,-1)-1`` is a deterministic function of
future ``$close``, and ``predict()`` receives the whole test panel at once, so a
model could read the features of dates after t (e.g. a per-instrument
``shift(-1)``) for its date-t prediction. After the normal prediction,
``generate`` therefore re-builds the dataset from the workflow config with
qlib's data provider cut off at a date t drawn at random from the test
segment (calendar and raw features end at t, so the label of the last two
dates is unknown as well), predicts with a copy of the model taken before the
normal prediction, and requires the predictions of the last dates up to t to
match the normal ones. A prediction that used any information from after t
(in ``predict()``, a processor, the handler's features or ``qlib.data.D``
during the check) changes and the run is rejected. The only predictions not
compared are those whose qlib MTSDatasetH window is itself padded with another
instrument's later rows (a qlib quirk for an instrument's first seq_len-1
samples, also reaching TRA's memory).

KNOWN RESIDUAL (in-process): the check cannot stop code that deliberately
carries the full-panel answers into the check -- e.g. test-segment features
stashed during ``fit()`` (fit receives the dataset), a cache kept in a module
global or on disk, or reading the provider's ``.bin`` files directly.
"""

import bisect
import copy
import os
import random
import time

import numpy as np
import pandas as pd

# When False, the fit-guard prepare() patch passes labels through untouched
# (used only while the host-side scorer reads the held-out test label).
_GUARD_ACTIVE = True

# Workflow dataset config (set by install()); the causality check re-builds
# the dataset from it.
_DATASET_CONFIG = None
# Compared dates ending at the cutoff; the cutoff is never one of the last
# _CHECK_TAIL test dates (the full panel has no future there either).
_CHECK_WINDOW = 5
_CHECK_TAIL = 10
# Per compared date, the check predictions must equal the normal ones up to
# float noise of GPU kernels: Pearson correlation >= _CHECK_MIN_CORR (noise of
# 1e-3 x std gives 1 - 5e-7; a prediction that used the next dates moves by
# O(std)).
_CHECK_MIN_CORR = 0.999


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


def _shallow_copy(obj):
    """Shallow copy that keeps EVERY attribute.

    ``copy.copy`` goes through qlib ``Serializable.__getstate__``, which drops
    all ``_``-prefixed attributes (e.g. ``MTSDatasetH._index`` / ``_data`` /
    ``_batch_slices``), so the copied dataset could not ``prepare()`` anymore
    and every MTSDatasetH model (the TRA baseline) crashed in ``predict()``.
    """
    new = object.__new__(type(obj))
    new.__dict__.update(obj.__dict__)
    return new


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

    new_handler = _shallow_copy(handler)  # shallow: keeps refs we then overwrite
    for attr in ("_data", "_infer", "_learn"):
        if hasattr(handler, attr):
            setattr(new_handler, attr, _null_label_columns(getattr(handler, attr)))
    if hasattr(new_handler, "data_loader"):
        try:
            new_handler.data_loader = _BlockedLoader()
        except Exception:
            pass

    new_ds = _shallow_copy(real_ds)
    new_ds.handler = new_handler
    return new_ds


class _DataCutoff:
    """qlib's data provider ends at ``cutoff`` inside this context: the
    calendar is truncated and raw feature reads stop at the cutoff index, so
    expressions (features, labels, ``qlib.data.D``) cannot see later dates."""

    def __init__(self, cutoff):
        self.cutoff = pd.Timestamp(cutoff)

    def __enter__(self):
        from qlib.data import data as qd
        from qlib.data.cache import H

        cutoff, memo = self.cutoff, {}
        orig_cal = self._orig_cal = qd.CalendarProvider._get_calendar
        orig_feat = self._orig_feat = qd.LocalFeatureProvider.feature

        def _get_calendar(prov, freq, future):
            key = f"{freq}_future_{future}"
            if key not in memo:
                cal, _ = orig_cal(prov, freq, future)
                cal = cal[: bisect.bisect_right(cal, cutoff)]
                memo[key] = (cal, {x: i for i, x in enumerate(cal)})
            return memo[key]

        def feature(prov, instrument, field, start_index, end_index, freq):
            last = len(qd.Cal._get_calendar(freq, False)[0]) - 1
            if end_index is None or end_index > last:
                end_index = last
            if start_index is not None and start_index > end_index:
                return pd.Series(dtype=np.float32)
            return orig_feat(prov, instrument, field, start_index, end_index, freq)

        H["f"].clear()  # expression cache filled with unrestricted data
        qd.CalendarProvider._get_calendar = _get_calendar
        qd.LocalFeatureProvider.feature = feature
        return self

    def __exit__(self, *exc):
        from qlib.data import data as qd
        from qlib.data.cache import H

        qd.CalendarProvider._get_calendar = self._orig_cal
        qd.LocalFeatureProvider.feature = self._orig_feat
        H["f"].clear()
        return False


def _rng_state():
    import torch

    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), cuda


def _set_rng_state(state):
    import torch

    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def _snapshot(model, dataset):
    """Model copy, RNG state and dataset memory taken BEFORE the normal
    prediction, so nothing the normal prediction stores can reach the check.
    The copy goes through ``__dict__`` (qlib's ``Serializable`` drops
    ``_``-prefixed attributes in ``copy``/``pickle``)."""
    try:
        snap = object.__new__(type(model))
        snap.__dict__.update(copy.deepcopy(model.__dict__))
    except Exception as e:  # not deep-copyable: fall back to the fitted object
        print(f"[causality check] model not deep-copyable ({type(e).__name__}); "
              "the check re-uses the fitted model object")
        snap = model
    memory = getattr(dataset, "_memory", None)
    memory = memory.copy() if isinstance(memory, np.ndarray) else None
    return snap, _rng_state(), memory


def _warm_memory(real_ds, memory, new_ds):
    """MTSDatasetH keeps a memory (TRA's loss history) that fit() fills for the
    dates before the test segment; give the re-built dataset those rows."""
    dst = getattr(new_ds, "_memory", None)
    if memory is None or not isinstance(dst, np.ndarray) or memory.shape[1:] != dst.shape[1:]:
        return
    before = pd.Timestamp(new_ds.segments["test"][0])
    if getattr(new_ds, "memory_mode", None) == "sample":
        src_idx, dst_idx = real_ds._index, new_ds._index
        keep = np.flatnonzero(dst_idx.get_level_values(1) < before)
    elif getattr(new_ds, "memory_mode", None) == "daily":
        src_idx = pd.Index(real_ds._daily_index.values)
        dst_idx = pd.Index(new_ds._daily_index.values)
        keep = np.flatnonzero(dst_idx < before)
    else:
        return
    pos = src_idx.get_indexer(dst_idx[keep])
    ok = pos >= 0
    dst[keep[ok]] = memory[pos[ok]]


def _mts_padding_rows(real_ds, ds, rows):
    """qlib's MTSDatasetH fills the first seq_len-1 windows of an instrument with
    the rows of the instrument sorted before it, whose latest rows lie after the
    cutoff in the full panel, and TRA-style memory carries those windows' losses
    into the later states of the instrument. Returns the ``rows`` whose window
    contains a test-period padded window whose data differs between the full
    and the cut dataset (none for any other dataset class)."""
    from qlib.contrib.data.dataset import MTSDatasetH

    if type(real_ds) is not MTSDatasetH or type(ds) is not MTSDatasetH:
        return rows[:0]
    codes = np.asarray(ds._index.get_level_values(0))
    dts = ds._index.get_level_values(1)
    first = pd.Series(np.arange(len(codes))).groupby(codes).transform("min").values
    starts = np.array([s.start for s in ds._batch_slices])
    cal = dts.unique().sort_values()
    lo = cal[max(0, cal.searchsorted(rows.get_level_values(0).min()) - ds.seq_len - 1)]
    lo = max(lo, pd.Timestamp(ds.segments["test"][0]))  # earlier memory is copied
    pos_r = pd.Series(np.arange(len(real_ds._index)), index=real_ds._index)
    tainted = np.zeros(len(codes) + 1, dtype=np.int64)
    for i in np.flatnonzero((starts < first) & (dts >= lo)):
        j = pos_r.get(ds._index[i])
        if j is None or not np.array_equal(ds._data[ds._batch_slices[i]],
                                           real_ds._data[real_ds._batch_slices[j]]):
            tainted[i + 1] = 1
    tainted = np.cumsum(tainted)
    pos_c = pd.Series(np.arange(len(codes)), index=ds._index)
    out = []
    for dt, inst in rows:
        i = pos_c.get((inst, dt))
        if i is not None:
            s = ds._batch_slices[i]
            if tainted[s.stop] > tainted[s.start]:
                out.append((dt, inst))
    return pd.MultiIndex.from_tuples(out, names=rows.names) if out else rows[:0]


def _score_series(pred):
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    if not isinstance(pred, pd.Series) or "datetime" not in list(pred.index.names or []):
        raise RuntimeError(
            "causality check: predict() must return a pd.Series (or DataFrame) "
            "indexed by (datetime, instrument)")
    names = list(pred.index.names)
    if names != ["datetime", "instrument"] and set(names) == {"datetime", "instrument"}:
        pred = pred.reorder_levels(["datetime", "instrument"])
    return pred.astype("float64")


def _causality_check(snapshot, pred, real_ds):
    """Re-predict with the data provider cut off at a random test date t and
    require the predictions of the last dates <= t to be unchanged."""
    from qlib.data.dataset import Dataset
    from qlib.utils import init_instance_by_config

    start = time.time()
    model, rng_before, memory = snapshot
    full = _score_series(pred)
    dates = pd.DatetimeIndex(full.index.get_level_values("datetime").unique()).sort_values()
    n_cut = len(dates) - _CHECK_TAIL - _CHECK_WINDOW
    if n_cut <= 0:
        raise RuntimeError("causality check: the test segment is too short")
    cutoff = dates[_CHECK_WINDOW + int.from_bytes(os.urandom(8), "little") % n_cut]
    window = dates[dates <= cutoff][-_CHECK_WINDOW:]

    rng_after = _rng_state()
    with _DataCutoff(cutoff):
        dataset = init_instance_by_config(copy.deepcopy(_DATASET_CONFIG), accept_types=Dataset)
        _warm_memory(real_ds, memory, dataset)
        _set_rng_state(rng_before)
        check = model.predict(_build_label_free_dataset(dataset))
    _set_rng_state(rng_after)

    ref = full[full.index.get_level_values("datetime").isin(window)]
    padded = _mts_padding_rows(real_ds, dataset, ref.index)
    del dataset
    if len(padded) > 0.2 * len(ref):
        raise RuntimeError(
            f"CAUSALITY CHECK FAILED: {len(padded)} of {len(ref)} predictions for "
            f"{window[0].date()}..{window[-1].date()} get MTSDatasetH windows padded "
            "with other instruments' later rows; use a shorter seq_len.")
    ref = ref[~ref.index.isin(padded)]
    got = _score_series(check)
    got = got[~got.index.duplicated()].reindex(ref.index)
    min_corr = 1.0
    for date in window:
        sel = np.asarray(ref.index.get_level_values("datetime") == date)
        a, b = ref.values[sel], got.values[sel]
        fin = np.isfinite(a)
        ok = bool((fin == np.isfinite(b)).all())
        a, b = a[fin], b[fin]
        if ok and not np.allclose(a, b, rtol=1e-6, atol=1e-12):
            corr = np.corrcoef(a, b)[0, 1] if len(a) > 2 and a.std() > 0 and b.std() > 0 else np.nan
            ok = bool(corr >= _CHECK_MIN_CORR)
            min_corr = min(min_corr, float(corr)) if ok else min_corr
        if not ok:
            av, bv = ref.values[sel], got.values[sel]
            changed = int((~((np.isnan(av) & np.isnan(bv))
                             | np.isclose(av, bv, rtol=1e-6, atol=1e-12))).sum())
            i = int(np.argmax(np.abs(np.nan_to_num(a - b, nan=np.inf)))) if len(a) else 0
            ex = f" (e.g. {a[i]!r} -> {b[i]!r})" if len(a) else ""
            raise RuntimeError(
                f"CAUSALITY CHECK FAILED: with the data ending at {cutoff.date()}, "
                f"{changed} of {int(sel.sum())} predictions for {date.date()} changed{ex}. "
                "The prediction for date t may only use data up to date t.")
    print(f"CAUSALITY_CHECK passed: data cut at {cutoff.date()}, {len(ref)} predictions "
          f"on {len(window)} dates unchanged (min corr {min_corr:.9f}"
          + (f"; {len(padded)} MTSDatasetH-padded windows skipped" if len(padded) else "")
          + f"), {time.time() - start:.0f}s")


def install(dataset_config):
    """Monkey-patch ``SignalRecord.generate`` to hand ``predict()`` a
    label-free dataset view and to run the causality check. Idempotent.

    ``dataset_config`` is the workflow's ``task.dataset`` config (after the
    CLI overrides); the causality check re-builds the dataset from it."""
    global _DATASET_CONFIG
    _DATASET_CONFIG = copy.deepcopy(dataset_config)
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
        snapshot = _snapshot(self.model, real_ds)
        pred = self.model.predict(safe_ds)
        # look-ahead check: fails the run if a prediction used later data
        _causality_check(snapshot, pred, real_ds)
        del snapshot
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
