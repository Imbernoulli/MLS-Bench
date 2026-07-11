"""Prepare fixed UCI-regression splits for the gpytorch-gp (gp-*) MLS-Bench tasks.

Produces, under {data_root}/gpytorch-gp/:
  concrete.npz   Concrete Compressive Strength (1030 x 8)  -> gp-kernel-design
  elevators.npz  Elevators (16599 x 18)                    -> gp-sparse-inducing
  kin8nm.npz     Kin8nm (8192 x 8)                         -> gp-deep-kernel

Each .npz holds train_x, train_y, test_x, test_y (float arrays, raw scale). A fixed
90/10 split at seed 42 is used for every dataset so every run is scored on identical
held-out points. Standardization happens later in the harness (train-stats only).

Datasets are pulled from OpenML via scikit-learn (no auth). Requires network on the
HOST (or a proxy: set http_proxy/https_proxy). OpenML data ids:
  concrete=4353 (target: compressive strength), kin8nm=189 (target: y),
  elevators=216 (target: Goal). All numeric-feature regression sets.
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np

SEED = 42
# name -> (openml data_id, target column name)
DATASETS = {
    "concrete": (4353, None),   # last column is the target
    "kin8nm": (189, "y"),
    "elevators": (216, "Goal"),
}
FORMAT_VERSION = 2
SPLIT_PROTOCOL = "openml-full-random90-10-seed42-v1"
DATASET_CONTRACTS = {
    "concrete": {
        "rows": 1030,
        "dim": 8,
        "source_sha256": "092c87480aff2080e039c1c1ef9d87e6f5748352549682875a5834eb52a8bfde",
        "split_sha256": "c6cb20776e4eebaee665fea5d9a5a688db19c6699219dfcac2a258e7977f1773",
    },
    "kin8nm": {
        "rows": 8192,
        "dim": 8,
        "source_sha256": "970265a06441ea9ed0524a75c28a60880c3e76fbac1a5a374d090b848b6156bd",
        "split_sha256": "7b0680527b8b8835c300c183a81fa92c6b7f64ce047db6b574d9f63aa02d9fd0",
    },
    "elevators": {
        "rows": 16599,
        "dim": 18,
        "source_sha256": "9e68b4c7838d3c2c998d5ae96f5aa033b722a7fad2a581b7a9511a73f05d0a39",
        "split_sha256": "2d852b51f9424cf235786e4402aa7a80cc0676cee68b0eafcfaf8e9c71e46c2c",
    },
}


def _source_sha256(x: np.ndarray, y: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (x, y):
        canonical = np.ascontiguousarray(value, dtype="<f8")
        digest.update(str(canonical.shape).encode())
        digest.update(b"\0")
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _split_sha256(arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    for label, value in zip(("train_x", "train_y", "test_x", "test_y"), arrays):
        canonical = np.ascontiguousarray(value, dtype="<f4")
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(str(canonical.shape).encode())
        digest.update(b"\0")
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _fixed_split(x: np.ndarray, y: np.ndarray, frac_test: float = 0.1):
    rng = np.random.RandomState(SEED)
    n = x.shape[0]
    perm = rng.permutation(n)
    n_test = int(round(n * frac_test))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def _from_openml(data_id: int, target: str | None):
    from sklearn.datasets import fetch_openml  # noqa: WPS433
    fr = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = fr.frame
    if target is None:
        target = df.columns[-1]
    elif target not in df.columns:
        raise RuntimeError(f"OpenML dataset {data_id} is missing target column {target!r}")
    y = df[target].to_numpy(dtype=np.float64)
    features = df.drop(columns=[target])
    numeric = features.select_dtypes(include="number")
    if list(numeric.columns) != list(features.columns):
        raise RuntimeError(f"OpenML dataset {data_id} contains non-numeric feature columns")
    x = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError(f"OpenML dataset {data_id} contains missing or non-finite values")
    return x, y


def build(name: str):
    data_id, target = DATASETS[name]
    x, y = _from_openml(data_id, target)
    contract = DATASET_CONTRACTS[name]
    if x.shape != (contract["rows"], contract["dim"]) or y.shape != (contract["rows"],):
        raise RuntimeError(
            f"{name}: source shape mismatch: x={x.shape}, y={y.shape}, "
            f"expected ({contract['rows']}, {contract['dim']}) and ({contract['rows']},)"
        )
    source_sha256 = _source_sha256(x, y)
    if source_sha256 != contract["source_sha256"]:
        raise RuntimeError(
            f"{name}: OpenML source checksum mismatch: {source_sha256} != "
            f"{contract['source_sha256']}"
        )
    return _fixed_split(x, y), source_sha256


def _cache_is_current(path: Path, name: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    contract = DATASET_CONTRACTS[name]
    expected_test = int(round(contract["rows"] * 0.1))
    expected_train = contract["rows"] - expected_test
    try:
        with np.load(path, allow_pickle=False) as cache:
            return (
                int(cache["format_version"].item()) == FORMAT_VERSION
                and str(cache["split_protocol"].item()) == SPLIT_PROTOCOL
                and str(cache["dataset_name"].item()) == name
                and int(cache["openml_data_id"].item()) == DATASETS[name][0]
                and str(cache["source_sha256"].item()) == contract["source_sha256"]
                and str(cache["split_sha256"].item()) == contract["split_sha256"]
                and cache["train_x"].shape == (expected_train, contract["dim"])
                and cache["train_y"].shape == (expected_train,)
                and cache["test_x"].shape == (expected_test, contract["dim"])
                and cache["test_y"].shape == (expected_test,)
            )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--only", default=None, help="comma-separated subset of datasets")
    args = ap.parse_args()

    out_dir = Path(args.data_root) / "gpytorch-gp"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(DATASETS)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]

    for name in names:
        out = out_dir / f"{name}.npz"
        if _cache_is_current(out, name):
            print(f"{name} verified and current -> {out}", flush=True)
            continue
        if out.exists():
            print(f"{name} cache is stale; rebuilding from checksum-verified OpenML data", flush=True)
        print(f"Building {name} ...", flush=True)
        (tr_x, tr_y, te_x, te_y), source_sha256 = build(name)
        split_sha256 = _split_sha256((tr_x, tr_y, te_x, te_y))
        if split_sha256 != DATASET_CONTRACTS[name]["split_sha256"]:
            raise RuntimeError(
                f"{name}: deterministic split checksum mismatch: {split_sha256} != "
                f"{DATASET_CONTRACTS[name]['split_sha256']}"
            )
        with tempfile.NamedTemporaryFile(
            prefix=f".{name}.", suffix=".npz", dir=out_dir, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            np.savez(
                temporary,
                train_x=tr_x.astype(np.float32),
                train_y=tr_y.astype(np.float32),
                test_x=te_x.astype(np.float32),
                test_y=te_y.astype(np.float32),
                format_version=np.int64(FORMAT_VERSION),
                split_protocol=np.asarray(SPLIT_PROTOCOL),
                dataset_name=np.asarray(name),
                openml_data_id=np.int64(DATASETS[name][0]),
                source_sha256=np.asarray(source_sha256),
                split_sha256=np.asarray(split_sha256),
                split_seed=np.int64(SEED),
                total_rows=np.int64(DATASET_CONTRACTS[name]["rows"]),
                feature_dim=np.int64(DATASET_CONTRACTS[name]["dim"]),
            )
            temporary.replace(out)
        finally:
            temporary.unlink(missing_ok=True)
        print(
            f"  {name}: train {tr_x.shape} test {te_x.shape} d={tr_x.shape[1]} -> {out}",
            flush=True,
        )


if __name__ == "__main__":
    main()
