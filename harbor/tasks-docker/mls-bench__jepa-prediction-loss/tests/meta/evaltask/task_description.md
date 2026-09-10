# Temporal JEPA Prediction Loss Optimization

## Research Question
Design a better prediction cost function for multi-step temporal Joint Embedding Predictive Architecture (JEPA). The prediction loss measures discrepancy between predicted and target representations in the latent space, directly influencing how well the predictor learns to model temporal dynamics.

## Background
JEPA-style self-supervised models (Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243) train an encoder and a predictor jointly, with the predictor matching latent representations of context and target. In the temporal extension used here, the encoder produces a spatial feature map for each frame and the predictor operates autoregressively over time on a Moving MNIST sequence. The training loss is the sum of:
- a **prediction loss** comparing predicted and target latent feature maps (the component you redesign), and
- a **VCLoss (Variance–Covariance) regularizer** that prevents collapse, in the spirit of VICReg (Bardes, Ponce, LeCun, ICLR 2022, arXiv:2105.04906).

The current baseline uses a plain `F.mse_loss(state, predicted)`, which treats every channel and spatial location identically and ignores temporal structure.

## What You Can Modify
The `CustomPredictionLoss` class in `custom_prediction_loss.py`. You may modify the `__init__` and `forward` methods, add helper methods, and import additional modules.

## Interface
```python
class CustomPredictionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        """
        Args:
            state:     [B, C, T, H, W] - target encoded representations from the encoder
            predicted: [B, C, T, H, W] - predicted representations from the predictor

        Returns:
            Scalar loss tensor (lower means predicted is closer to state)
        """
```
The loss is called during JEPA's `unroll()` method as `predcost(state, predicted_states)`, where both tensors share the same shape. The returned scalar is added to the regularization loss and backpropagated.

## Evaluation
Mean detection Average Precision (AP) across prediction timesteps on Moving MNIST. Higher is better. The model is trained for 50 epochs with the Adam optimizer (lr=1e-3) and the final mean detection AP is reported.

The prediction loss is evaluated across three model sizes to test generalization:
- **small**: henc=16, dstc=8, hpre=16
- **base**: henc=32, dstc=16, hpre=32
- **large**: henc=64, dstc=32, hpre=64

## Notes
- The encoder produces spatial feature maps (not just vectors), so spatial structure matters.
- The predictor operates autoregressively over time steps, so temporal weighting/ordering can be exploited.
- The VCLoss regularizer is computed separately and added by the trainer; you only design the prediction term.
