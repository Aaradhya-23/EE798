"""Localisation metrics that turn a heat-map into a region and compare it with VOC boxes."""
import numpy as np
from scipy import ndimage as ndi


def box_from_heatmap(h: np.ndarray, thr: float):
    """Binarise at thr (fraction of the max), keep the largest 4-connected component, return its box."""
    mask = h >= thr
    labels, n = ndi.label(mask)
    if n == 0:
        return None
    areas = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    k = int(np.argmax(areas)) + 1
    ys, xs = np.where(labels == k)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, iw) * max(0.0, ih)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def gt_union_mask(boxes, H, W) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        m[max(0, int(round(y1))):min(H, int(round(y2))), max(0, int(round(x1))):min(W, int(round(x2)))] = True
    return m


def score_heatmap(h: np.ndarray, gt_boxes, thrs, tol: int = 0) -> dict:
    """gt_boxes: all (non-difficult) boxes of the explained class in this image."""
    H, W = h.shape
    empty = not bool(h.max() > 0)                  # e.g. Grad-CAM when no location supports class c
    y, x = np.unravel_index(int(np.argmax(h)), h.shape)
    point = (not empty) and any(b[0] - tol <= x < b[2] + tol and b[1] - tol <= y < b[3] + tol for b in gt_boxes)
    m = gt_union_mask(gt_boxes, H, W)
    mass = float(h[m].sum() / (h.sum() + 1e-12))          # fraction of heat-map energy inside GT boxes
    ious, areas = {}, {}
    for t in thrs:
        box = box_from_heatmap(h, t)
        if box is None:
            ious[t], areas[t] = 0.0, 0.0
        else:                                               # best IoU over instances of the class
            ious[t] = max(box_iou(box, g) for g in gt_boxes)
            areas[t] = box_area(box) / (H * W)
    return {"point": bool(point), "empty": empty, "mass": mass, "iou": ious, "area": areas, "gt_area": float(m.mean())}


def discrimination_index(h, own_boxes, other_boxes, min_frac: float = 0.01):
    """Mean heat inside own-class-only pixels / (own + other-class-only mean heat). 0.5 = chance."""
    H, W = h.shape
    mo, mx = gt_union_mask(own_boxes, H, W), gt_union_mask(other_boxes, H, W)
    own, oth = mo & ~mx, mx & ~mo
    if own.mean() < min_frac or oth.mean() < min_frac:
        return None
    d_own, d_oth = h[own].mean(), h[oth].mean()
    return float(d_own / (d_own + d_oth + 1e-12))


def center_gaussian(size, sigma_frac: float = 0.25) -> np.ndarray:
    """Model-free control: an isotropic Gaussian centred in the image, max = 1."""
    H, W = size
    yy, xx = np.mgrid[0:H, 0:W]
    s = sigma_frac * min(H, W)
    return np.exp(-(((yy - (H - 1) / 2) ** 2 + (xx - (W - 1) / 2) ** 2) / (2 * s * s)))


def bootstrap_ci(x, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    rng = np.random.RandomState(seed)
    means = x[rng.randint(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a, b, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(d.mean()), *bootstrap_ci(d, n, seed, alpha)
