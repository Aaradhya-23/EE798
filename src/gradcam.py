"""Grad-CAM (Selvaraju et al., ICCV 2017) implemented from scratch with PyTorch.

Activations A^k are captured with a forward hook; the gradients dy^c/dA^k_ij are obtained
with torch.autograd.grad on the pre-softmax class score.  No third-party Grad-CAM code is used.
Grad-CAM++ (Chattopadhay et al., 2018) is included only as an extra comparison method; it
re-uses the same A and G, so it costs nothing extra.
"""
import torch
import torch.nn.functional as F


def normalize_map(m: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample min-max normalisation of a (B,1,H,W) map to [0,1]."""
    mn = m.amin(dim=(-2, -1), keepdim=True)
    mx = m.amax(dim=(-2, -1), keepdim=True)
    return (m - mn) / (mx - mn + eps)


def upsample(cam: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(cam, size=tuple(size), mode="bilinear", align_corners=False)


def gradcam_weights_and_map(A: torch.Tensor, G: torch.Tensor):
    """A, G: (B,K,h,w) activations and gradients of the class score w.r.t. A."""
    # BEGIN snippet:core
    alpha = G.mean(dim=(2, 3), keepdim=True)              # alpha_k^c = (1/Z) sum_ij dy^c/dA^k_ij
    cam = F.relu((alpha * A).sum(dim=1, keepdim=True))    # ReLU( sum_k alpha_k^c A^k )
    # END snippet:core
    return alpha, cam


def gradcampp_map(A: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Grad-CAM++ closed form (exponential class score assumption, as in the original paper)."""
    g2, g3 = G ** 2, G ** 3
    denom = 2 * g2 + A.sum(dim=(2, 3), keepdim=True) * g3
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    a = g2 / denom
    w = (a * F.relu(G)).sum(dim=(2, 3), keepdim=True)
    return F.relu((w * A).sum(dim=1, keepdim=True))


class GradCAM:
    """Explain a CNN at one or several layers: layers = {"layer4": model.layer4, ...}."""

    def __init__(self, model, layers: dict):
        self.model = model.eval()
        self._acts = {}
        self.A, self.alpha, self.raw = {}, {}, {}      # kept for tests / CAM baseline
        self._handles = [m.register_forward_hook(self._save(n)) for n, m in layers.items()]

    def _save(self, name):
        # BEGIN snippet:hook
        def hook(module, inputs, output):
            self._acts[name] = output        # keep the graph: we differentiate w.r.t. this tensor
        # END snippet:hook
        return hook

    def __call__(self, x: torch.Tensor, class_idx, pp: bool = True) -> dict:
        size = x.shape[-2:]
        x = x.clone().requires_grad_(True)            # guarantees a graph even if params are frozen
        class_idx = torch.as_tensor(class_idx, device=x.device).long().reshape(-1)
        if class_idx.numel() == 1:
            class_idx = class_idx.expand(x.size(0))
        self._acts.clear()
        logits = self.model(x)                         # forward pass; hooks fill self._acts
        # BEGIN snippet:backward
        score = logits[torch.arange(x.size(0), device=x.device), class_idx].sum()   # y^c, pre-softmax
        names = list(self._acts)
        grads = torch.autograd.grad(score, [self._acts[n] for n in names])          # dy^c / dA^k_ij
        # END snippet:backward
        out = {"gradcam": {}, "gradcampp": {}, "logits": logits.detach()}
        for n, G in zip(names, grads):
            A = self._acts[n].detach()
            alpha, cam = gradcam_weights_and_map(A, G)
            self.A[n], self.alpha[n], self.raw[n] = A, alpha, cam
            out["gradcam"][n] = normalize_map(upsample(cam, size))
            if pp:
                out["gradcampp"][n] = normalize_map(upsample(gradcampp_map(A, G), size))
        self._acts.clear()
        return out

    def remove(self):
        for h in self._handles:
            h.remove()
