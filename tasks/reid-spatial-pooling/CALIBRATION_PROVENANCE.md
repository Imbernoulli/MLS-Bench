# ReID Spatial-Pooling Evidence

The task score is baseline-free. Historical max/average/GeM rows were removed
because the source tree did not contain immutable worker IDs, raw-log hashes, or
artifact hashes for those H200 measurements.

The following independent terminal representative remains valid as workload and
runtime evidence; it is not used as a scoring anchor:

- Dataset version: `18766`
- Mangrove task/container request: `96623` / `4950705`
- Source commit: `2b06af3b2bb14d4c76c204703560337c282e28bf`
- Image: `sha256:fbaaa5d4dcd03ea4e2bf1084b1b8cc78c5ae09723033b5a05d4ec96bd2b8264f`
- Resource: one H20 GPU
- Native surface: global average pooling
- Training: 60 epochs, 11,003 optimizer steps, 704,192 sampled images
- Evaluation: 3,368 queries partitioned 1,122 / 1,123 / 1,123; each group uses the complete 19,732-image gallery
- Command / verifier / setup / platform: 1,078.771s / 1,085.342s / 164.083s / 1,253.196s
- Recorded reward under the superseded calibration mapping: `0.1932609536455854`

The schema-2 task binding, exact-budget rejection, and terminal completion proof
do not change the material spatial-pooling workload. A new training run is only
required if that workload, data inventory, checkpoint, or model path changes.
