"""Head baseline (pitfall): L2-normalisation before a Euclidean triplet loss."""
import torch.nn as nn
class L2Norm(nn.Module):
    def forward(self, x):
        import torch.nn.functional as F
        return F.normalize(x, p=2, dim=1)
def build_head(feat_dim):
    head = L2Norm()
    head.name = "l2norm"
    return head
