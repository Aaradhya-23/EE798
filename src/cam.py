"""Class Activation Mapping (Zhou et al., CVPR 2016) for a GAP -> linear-classifier network."""
import torch

from .gradcam import normalize_map, upsample


def cam_heatmap(fc_weight: torch.Tensor, A: torch.Tensor, class_idx, size) -> torch.Tensor:
    """fc_weight: (C,K) weights of the linear layer after global average pooling; A: (B,K,h,w).

    CAM_c(i,j) = sum_k w_k^c A^k_ij   (the bias is ignored; no ReLU, min-max normalised as in CAM).
    """
    class_idx = torch.as_tensor(class_idx, device=A.device).long().reshape(-1).expand(A.size(0))
    w = fc_weight[class_idx]                                   # (B,K)
    cam = (w[:, :, None, None] * A).sum(dim=1, keepdim=True)   # (B,1,h,w)
    return normalize_map(upsample(cam, size))
