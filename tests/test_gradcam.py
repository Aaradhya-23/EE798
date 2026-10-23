"""Correctness tests (random weights, synthetic input): these verify the code, they are NOT results."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cam import cam_heatmap
from src.evaluation import box_from_heatmap, box_iou, score_heatmap
from src.gradcam import GradCAM, normalize_map, upsample
from src.model import target_layers


def _setup(seed=0):
    torch.manual_seed(seed)
    m = torchvision.models.resnet50(weights=None, num_classes=20).eval()
    x = torch.randn(2, 3, 128, 128)
    return m, x


def test_gradcam_equals_cam_for_gap_linear_head():
    """Derivation: y^c = sum_k w_k (1/Z) sum_ij A_ij^k  =>  alpha_k = w_k / Z  =>  Grad-CAM = ReLU(CAM)/Z."""
    m, x = _setup()
    gc = GradCAM(m, {"layer4": m.layer4})
    cls = torch.tensor([3, 7])
    gc(x, cls)
    A, alpha, raw = gc.A["layer4"], gc.alpha["layer4"], gc.raw["layer4"]
    Z = A.shape[2] * A.shape[3]
    W = m.fc.weight[cls]
    assert torch.allclose(alpha.flatten(1), W / Z, atol=1e-6)
    cam = (W[:, :, None, None] * A).sum(1, keepdim=True)
    assert torch.allclose(raw, F.relu(cam) / Z, atol=1e-6)


def test_gradcam_matches_retain_grad_route_all_layers():
    """Independent mechanism: loss.backward() + retain_grad() must give the same alpha."""
    m, x = _setup(1)
    layers = target_layers(m)
    gc = GradCAM(m, layers)
    gc(x, 5)
    acts = {}

    def make(n):
        def hook(_m, _i, o):          # must return None, otherwise the layer output is replaced
            o.retain_grad()
            acts[n] = o
        return hook

    hs = [mod.register_forward_hook(make(n)) for n, mod in layers.items()]
    xx = x.clone().requires_grad_(True)
    m(xx)[:, 5].sum().backward()
    for h in hs:
        h.remove()
    for n in layers:
        assert torch.allclose(gc.alpha[n], acts[n].grad.mean(dim=(2, 3), keepdim=True), atol=1e-6), n


def test_class_specific_and_range():
    m, x = _setup(2)
    gc = GradCAM(m, {"layer4": m.layer4})
    a = gc(x, 1)["gradcam"]["layer4"]
    b = gc(x, 2)["gradcam"]["layer4"]
    assert a.shape == (2, 1, 128, 128)
    assert 0 <= a.min() and a.max() <= 1 + 1e-6
    assert not torch.allclose(a, b)


def test_cam_heatmap_shape_and_range():
    m, x = _setup(3)
    gc = GradCAM(m, {"layer4": m.layer4})
    gc(x, 4)
    h = cam_heatmap(m.fc.weight, gc.A["layer4"], 4, x.shape[-2:])
    assert h.shape == (2, 1, 128, 128) and h.min() >= 0 and h.max() <= 1 + 1e-6


def test_hooks_removed():
    m, x = _setup()
    gc = GradCAM(m, {"layer4": m.layer4})
    gc.remove()
    assert len(m.layer4._forward_hooks) == 0


def test_box_and_iou():
    h = np.zeros((100, 100)); h[20:60, 30:50] = 1.0; h[80:85, 80:85] = 1.0   # largest blob wins
    assert box_from_heatmap(h, 0.15) == (30.0, 20.0, 50.0, 60.0)
    assert abs(box_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    s = score_heatmap(h, [(30, 20, 50, 60)], [0.15, 0.5])
    assert s["point"] and s["iou"][0.15] == 1.0 and 0 < s["mass"] < 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
