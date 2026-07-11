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

## Verifier identity audit

The terminal render and the current source have different proof-layer file
hashes, so runtime reuse was checked by source diff rather than assumed:

| file | terminal render at `2b06af3b2bb14d4c76c204703560337c282e28bf` | current source |
|---|---|---|
| `common.py` | `bea9b261a0c68982f5ca6f05dd432da34f912ade519f7f0abfa53ef75409bdd3` | `91dec560e2424c359405bf500fdeb054c5fddbd9cf07b2cd9b38473a0de8bddc` |
| `harness_pool.py` | `7cfba2ca18832373cd17afd8c5b378e5aec1dd23a2acc8f20f9c3594c88418be` | `5e5a54360ac2198e6613173a0c34fa76c21264f8b2a16aa4eec6d2b0a1a2e771` |

The spatial-pooling model, data loading, deterministic 60-epoch training loop,
11,003-update / 704,192-sample workload, and complete retrieval evaluation are
unchanged. The material differences bind the literal task/schema proof, use a
monotonic runtime clock, and fail before evaluation when the exact budget is
not met. The shared training helper changed for the other nine siblings, but
`harness_pool.py` does not call that helper.

The schema-2 task binding, exact-budget rejection, and terminal completion proof
do not change the material spatial-pooling workload. A new training run is only
required if that workload, data inventory, checkpoint, or model path changes.
