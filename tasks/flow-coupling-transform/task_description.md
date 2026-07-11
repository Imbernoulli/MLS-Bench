# Flow Coupling Transform

Select the coupling-transform recipe evaluated by one frozen training pipeline.

Edit `normflows-density/solution/coupling.py` and implement
`select_coupling_transform() -> str`. Valid values are `affine`, `spline4`, and
`spline8`. Each recipe contains eight coupling transforms and eight
between-coupling permutation modules, for 16 total `nf.flows` objects. The
verifier fixes the base distribution, width, optimizer, data, seed, and budget.

All three configured settings, `checkerboard`, `moons`, and `8gaussians`, are
scored. Each uses 30,000 training samples, 30,000 verifier-only test samples,
seed 42, and 20,000 Adam optimizer steps on one CUDA GPU. The verifier reports
exact NLL; lower is preferred. A metric is accepted only after the complete
ordered protocol and terminal success proof validate.
