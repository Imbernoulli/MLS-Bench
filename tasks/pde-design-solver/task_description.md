# Industrial CFD Design: Custom Neural Operator Design

## Objective
Design and implement a custom neural operator for industrial aerodynamic design prediction on 3D unstructured point clouds. Your code goes in the `Model` class in `models/Custom.py`. Reference implementations (PointNet, GraphSAGE, Graph_UNet, Transolver) from Neural-Solver-Library are provided as read-only context.

## Background
The task evaluates point-cloud / mesh-based neural operators on three steady aerodynamic design benchmarks. Key reference architectures:

- **PointNet** (Qi, Su, Mo, Guibas, "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation", CVPR 2017; arXiv:1612.00593). Per-point MLP with global max-pooling for permutation invariance over point sets. Code: https://github.com/charlesq34/pointnet.
- **GraphSAGE** (Hamilton, Ying, Leskovec, "Inductive Representation Learning on Large Graphs", NeurIPS 2017; arXiv:1706.02216). Inductive node embeddings via sample-and-aggregate over local neighborhoods; used here for message passing on the mesh graph.
- **Graph U-Net** (Gao, Ji, "Graph U-Nets", ICML 2019; arXiv:1905.05178). Encoder-decoder with learnable graph pooling (gPool) and unpooling (gUnpool) operations.
- **Transolver** (Wu, Luo, Wang, Wang, Long, "Transolver: A Fast Transformer Solver for PDEs on General Geometries", ICML 2024; arXiv:2402.02366). Physics-Attention that adaptively splits the discretized domain into learnable slices and computes attention among physical states rather than mesh points. Code: https://github.com/thuml/Transolver.

Underlying CFD benchmarks: **AirfRANS** (Bonnet et al., NeurIPS 2022 Datasets & Benchmarks; arXiv:2212.07564) and the **ShapeNet-Car** subset of ShapeNet popularized for surrogate aerodynamics by Umetani & Bickel (SIGGRAPH 2018).

## Model Interface
Your model receives `args` at initialization and must implement:
```python
forward(self, x, fx, T=None, geo=None) -> output
```
- `x`: 3D spatial coordinates, shape `(1, N, 3)` where N varies per mesh (~5000–10000 points).
- `fx`: input features (boundary conditions + geometry), shape `(1, N, 7)`.
- `T`: unused (always `None`).
- `geo`: **edge_index** tensor for graph connectivity between mesh points (required for graph-based models, can be `None` for non-graph approaches).
- output: predicted flow field, shape `(1, N, 4)` for Car/AirfRANS or `(1, N, 6)` for AirCraft (velocity + pressure components).

**Note**: Batch size is always 1 (one mesh per forward pass). Graph models (PointNet, GraphSAGE, Graph_UNet) squeeze the batch dimension and use `geo` for message passing. Non-graph models like Transolver ignore `geo`.

Key `args` attributes: `n_hidden`, `n_layers`, `n_heads`, `space_dim` (2 for AirfRANS, 3 for Car/AirCraft), `fun_dim=7`, `out_dim` (4 for Car/AirfRANS, 6 for AirCraft), `act`, `mlp_ratio`, `dropout`, `geotype` (`unstructured`), `radius` (for graph construction), `slice_num` (for Transolver-style physics attention).

## Hyperparameter Override (`CONFIG_OVERRIDES`)
The shell scripts (`scripts/car.sh`, `scripts/airfrans.sh`, `scripts/aircraft.sh`) default to `--n_hidden 128 --slice_num 32`. Different model families need different widths to be competitive — for example Transolver uses 256 in the original paper, while PointNet and Graph_UNet typically use much smaller widths. To set per-method values, edit the `CONFIG_OVERRIDES` dict at the bottom of `models/Custom.py`:

```python
# Allowed keys: n_hidden (int), slice_num (int).
CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}
```

Allowed keys are restricted to `n_hidden` and `slice_num`. The shell scripts read these from your file at runtime and pass them through as `--n_hidden` and `--slice_num`.

## Fixed Pipeline
Dataset loaders, OneCycleLR training schedule (200 epochs), loss function, metric computation, and the parameter budget check are all fixed. Only the `Model` class and the two `CONFIG_OVERRIDES` knobs are editable.

## Evaluation
Trained for 200 epochs (OneCycleLR) on three benchmarks. Metrics (multiple, all reported):
- **rho_d**: Spearman rank correlation of drag coefficient (higher is better).
- **c_d**: Relative error of drag coefficient (lower is better).
- **relative L2 error press/velo**: Relative L2 errors for pressure and velocity fields (lower is better).

Datasets:
- **Car** (ShapeNet-Car): public design benchmark used in Neural-Solver-Library / Transolver paper.
- **AirfRANS**: public 2D RANS airfoil benchmark used in Neural-Solver-Library / Transolver paper.
- **AirCraft**: a custom 3D aircraft design benchmark assembled for this task. There is no published paper baseline for AirCraft; it is included as a generalization probe and the reported numbers should be treated as task-internal references rather than literature reproductions.

## Parameter Budget
A budget check (`budget_check.py`) runs before training and rejects models whose parameter count exceeds 1.05× the largest paper-faithful baseline (Transolver at `n_hidden=256, slice_num=32`).
