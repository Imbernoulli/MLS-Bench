"""Strong head baseline: BNNeck (BatchNorm1d, bias frozen) -- Luo Bag of Tricks 2019."""
import torch.nn as nn
class BNNeck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.bn.bias.requires_grad_(False)  # BNNeck: no learnable shift
    def forward(self, x):
        return self.bn(x)
def build_head(feat_dim):
    head = BNNeck(feat_dim)
    head.name = "bnneck"
    return head
