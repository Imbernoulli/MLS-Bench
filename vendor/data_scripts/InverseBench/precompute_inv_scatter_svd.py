"""Precompute the inv-scatter forward-operator SVD artifact ONCE at build time.

WHY THIS EXISTS
---------------
``inverse_problems/inverse_scatter.py::InverseScatter.compute_svd`` builds a
dense ``(2*numTrans*numRec, Nx*Ny)`` matrix and runs ``torch.svd`` +
``torch.linalg.pinv`` on it in float64.  For the inv-scatter problem config
(numTrans=20, numRec=360, Nx=Ny=128) that matrix is ``(14400, 16384)`` and the
construction materialises a ``16384 x 16384`` complex128 ``torch.diag`` (~4 GiB)
per transmitter.  The float64 SVD/pinv is the dominant cost.

The operator already supports loading this artifact from
``cache/inv-scatter_numT_<numTrans>_numR_<numRec>/{U,S,Vt,matrix,matrix_inv}.pt``
(relative to the runtime cwd ``/workspace/InverseBench``, which is the mounted
writable cache dir).  But nothing ever populated that cache, so EVERY run
(every baseline, every agent, every seed) recomputed the SVD from scratch.  On
GPUs with crippled FP64 throughput (e.g. H20) this float64 SVD scales
pathologically and can consume the entire timeout without emitting a metric.

The artifact depends ONLY on the deterministic forward-operator config
(problem=inv-scatter), NOT on the agent's algorithm or the seed, so it is safe
to compute once and share across all runs/baselines/seeds.

This script reproduces ``compute_svd`` EXACTLY (same math, same dtypes, same
file names) so the operator's existing cache-load branch picks it up verbatim.
It is meant to be executed *inside the InverseBench container* (which has the
exact numpy<2 / scipy / torch versions and a CUDA GPU), invoked by
``prepare_data.py``.  Idempotent: if the artifact already exists it exits 0.
"""

import os
import sys
import time

import torch

# The package is on PYTHONPATH (/workspace/InverseBench) inside the container.
from inverse_problems.inverse_scatter import construct_parameters


# Forward-operator parameters for problem=inv-scatter. These MUST match
# configs/problem/inv-scatter.yaml. (svd default True; numTrans/numRec drive
# the cache path used by InverseScatter.compute_svd.)
PARAMS = dict(
    Lx=0.18,
    Ly=0.18,
    Nx=128,
    Ny=128,
    wave=6,
    numRec=360,
    numTrans=20,
    sensorRadius=1.6,
)


def main() -> int:
    # Cache root mirrors the runtime layout: the container mounts the host
    # vendor/data/inversebench/cache at /workspace/InverseBench/cache and runs
    # with cwd=/workspace/InverseBench, so the operator resolves the relative
    # path 'cache/inv-scatter_numT_<T>_numR_<R>'. We accept an explicit root so
    # the script works regardless of cwd.
    cache_root = os.environ.get("INV_SCATTER_CACHE_ROOT", "cache")
    numT = PARAMS["numTrans"]
    numR = PARAMS["numRec"]
    path = os.path.join(cache_root, f"inv-scatter_numT_{numT}_numR_{numR}")

    required = ["U.pt", "S.pt", "Vt.pt", "matrix.pt", "matrix_inv.pt"]
    if all(os.path.exists(os.path.join(path, f)) for f in required):
        print(f"[inv-scatter-svd] cache already complete at {path}; skipping", flush=True)
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[inv-scatter-svd] precomputing SVD on device={device} -> {path}", flush=True)
    t0 = time.time()

    Lx, Ly, Nx, Ny = PARAMS["Lx"], PARAMS["Ly"], PARAMS["Nx"], PARAMS["Ny"]
    dx, dy = Lx / Nx, Ly / Ny

    # Mirror InverseScatter.__init__ tensor setup exactly.
    _, sensor_greens, uinc, _ = construct_parameters(
        Lx, Ly, Nx, Ny, PARAMS["wave"], numR, numT, PARAMS["sensorRadius"], device
    )
    sensor_greens = sensor_greens.to(torch.complex128)  # (Ny, Nx, numRec)
    uinc = uinc.to(torch.complex128)                     # (Ny, Nx, numTrans)

    # Mirror compute_svd() matrix construction exactly.
    T = uinc[..., 0].flatten(0, 1)                       # (Nx*Ny, numTrans)
    R = sensor_greens[..., 0].reshape(-1, numR)          # (Nx*Ny, numRec)
    A = torch.cat(
        [R.T @ torch.conj(torch.diag(T[:, i])) for i in range(T.shape[-1])],
        dim=0,
    ) * dx * dy
    A = torch.view_as_real(A).permute(0, 2, 1).flatten(0, 1)
    print(
        f"[inv-scatter-svd] matrix A built shape={tuple(A.shape)} "
        f"dtype={A.dtype} dev={A.device} ({time.time()-t0:.1f}s)",
        flush=True,
    )

    U, Sigma, V = torch.svd(A)
    V_t = V.T
    A_inv = torch.linalg.pinv(A)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"[inv-scatter-svd] SVD + pinv done ({time.time()-t0:.1f}s)", flush=True)

    os.makedirs(path, exist_ok=True)
    # Save atomically-ish: write each tensor, matrix.pt LAST so the operator's
    # existence check (which keys on matrix.pt) only sees a complete cache.
    torch.save(U, os.path.join(path, "U.pt"))
    torch.save(Sigma, os.path.join(path, "S.pt"))
    torch.save(V_t, os.path.join(path, "Vt.pt"))
    torch.save(A_inv, os.path.join(path, "matrix_inv.pt"))
    torch.save(A, os.path.join(path, "matrix.pt"))
    print(
        f"[inv-scatter-svd] saved cache to {path} (total {time.time()-t0:.1f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
