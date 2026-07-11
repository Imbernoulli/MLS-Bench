# Image-Inpainting Question Design

The full-resolution protocol exposes ten distinct research surfaces. Candidate quality and ordering remain unmeasured until complete worker-side anchors exist.

| Task ID | Surface | Owned decision point |
| --- | --- | --- |
| `cv-inpaint-architecture` | `arch` | Replaces the complete inpainting network through `build_net`; it does not use the component-hook model. |
| `cv-inpaint-activation` | `activation` | Chooses the shape-preserving nonlinearity after fixed convolutions and normalization layers. |
| `cv-inpaint-norm` | `norm` | Builds channel-aware normalization layers before activation; it does not change the nonlinearity. |
| `cv-inpaint-attention` | `attention` | Transforms the 32x32 encoder bottleneck immediately before the dilation stack. |
| `cv-inpaint-dilation` | `dilation` | Chooses the receptive-field stack immediately after the attention position. |
| `cv-inpaint-fusion` | `fusion` | Combines each upsampled decoder tensor with its matching encoder skip tensor. |
| `cv-inpaint-gate` | `gate` | Filters encoder features using learned gate values and the resized validity mask. |
| `cv-inpaint-loss-design` | `loss` | Defines the non-negative scalar training objective; architecture and masks remain fixed. |
| `cv-inpaint-train-masking` | `masking` | Generates the optimization-time corruption masks; evaluation masks remain fixed. |
| `cv-inpaint-upsample` | `upsample` | Chooses each decoder's channel-preserving 2x spatial upsampling operator. |

The former `cv-inpaint-refine` sibling was removed. Its callable was a stateless post-processing function, not a trainable second-stage refinement network, and it overlapped the complete-architecture question. Keeping the old name would have overstated what the code evaluated.

Each task trains one checkpoint for exactly 100,000 optimizer steps and then evaluates that same checkpoint sequentially on `small`, `large`, and `strokes`. Training three independent checkpoints would triple cost, add optimization variance to a mask-only comparison, and would not isolate the question being asked. The parser accepts all three evaluation proofs atomically, so no subset can receive a score.

## P1 Context-Hook Review

`attention` and `dilation` are separate callables and execute in a fixed order, but they are consecutive transformations of the same 32x32 bottleneck and both change context aggregation. That makes them code-level distinct but conceptually coupled. No full-protocol anchor currently demonstrates that they support two independent research conclusions.

Treat `cv-inpaint-dilation` as provisional before runtime launch. The preferred resolution is to consolidate both into one context-aggregation question or replace dilation with a genuinely orthogonal inpainting decision. If no orthogonal replacement is justified, drop dilation and ship nine tasks rather than count two bottleneck-context variants as independent questions.
