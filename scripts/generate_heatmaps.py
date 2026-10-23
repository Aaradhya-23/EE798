"""Qualitative figures. Examples are chosen by fixed, seeded rules (no hand-picking); see README."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import explain_all
from src.dataset import VOC_CLASSES, VOCLocalization
from src.evaluation import box_from_heatmap, center_gaussian
from src.gradcam import GradCAM
from src.model import load_checkpoint, target_layers
from src.utils import denormalize, get_device, overlay, save_json, set_seed

COLS = ["Image + GT boxes", "Grad-CAM heat-map", "Grad-CAM overlay + pred. box", "Grad-CAM++", "CAM", "Vanilla saliency"]


def draw_boxes(ax, boxes, color):
    for (x1, y1, x2, y2) in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=2))


def render(rows, ds, model, gc, center, path, thr, title_fn):
    n = len(rows)
    fig, axes = plt.subplots(n, len(COLS), figsize=(2.6 * len(COLS), 2.6 * n), squeeze=False)
    for r, (i, c) in enumerate(rows):
        item = ds[i]
        x = item["image"][None].to(get_device())
        img = denormalize(item["image"])
        maps = explain_all(gc, model, x, c, x.shape[-2:], center)
        gt_c = [b[1:] for b in item["boxes"] if b[0] == c]
        gt_other = [b[1:] for b in item["boxes"] if b[0] != c]
        box = box_from_heatmap(maps["gradcam@layer4"], thr)
        ax = axes[r]
        ax[0].imshow(img); draw_boxes(ax[0], gt_c, "lime"); draw_boxes(ax[0], gt_other, "yellow")
        ax[1].imshow(maps["gradcam@layer4"], cmap="jet", vmin=0, vmax=1)
        ax[2].imshow(overlay(img, maps["gradcam@layer4"])); draw_boxes(ax[2], gt_c, "lime")
        if box is not None:
            draw_boxes(ax[2], [box], "red")
        for j, k in zip((3, 4, 5), ("gradcampp@layer4", "cam@layer4", "saliency")):
            ax[j].imshow(overlay(img, maps[k]) if k != "saliency" else maps[k], cmap="gray" if k == "saliency" else None)
        ax[0].set_ylabel(title_fn(item, c), fontsize=7)
        for a_ in ax:
            a_.set_xticks([]); a_.set_yticks([])
        if r == 0:
            for a_, t in zip(ax, COLS):
                a_.set_title(t, fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_layers(rows, ds, model, gc, center, path):
    fig, axes = plt.subplots(len(rows), 4, figsize=(10.4, 2.6 * len(rows)), squeeze=False)
    for r, (i, c) in enumerate(rows):
        item = ds[i]
        x = item["image"][None].to(get_device())
        img = denormalize(item["image"])
        maps = explain_all(gc, model, x, c, x.shape[-2:], center)
        axes[r][0].imshow(img); draw_boxes(axes[r][0], [b[1:] for b in item["boxes"] if b[0] == c], "lime")
        axes[r][0].set_ylabel(f"{VOC_CLASSES[c]}\n{item['image_id']}", fontsize=7)
        for j, l in enumerate(("layer2", "layer3", "layer4"), start=1):
            axes[r][j].imshow(overlay(img, maps[f"gradcam@{l}"]))
            if r == 0:
                axes[r][j].set_title(f"Grad-CAM @ {l}", fontsize=8)
        if r == 0:
            axes[r][0].set_title("Image + GT", fontsize=8)
        for a_ in axes[r]:
            a_.set_xticks([]); a_.set_yticks([])
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--ckpt", default="checkpoints/resnet50_voc07.pt")
    ap.add_argument("--res", default="results/quantitative")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--k", type=int, default=4, help="examples per category")
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    dev = get_device()
    model, ck = load_checkpoint(a.ckpt, dev)
    size = ck["args"]["img_size"]
    ds = VOCLocalization(a.root, "test", size, download=not a.no_download)
    gc = GradCAM(model, target_layers(model))
    center = center_gaussian((size, size))
    pp = pd.read_csv(f"{a.res}/per_pair.csv")
    probs = np.load(f"{a.res}/test_probs.npy")
    g = pp[pp.method == "gradcam@layer4"]
    ids = list(ds.indices)                                           # dataset index -> position in loader
    pick = lambda df, k: [(int(r.img_index), int(r.cls)) for r in df.sample(min(k, len(df)), random_state=rng).itertuples()]
    cats = {
        "success": g[(g.iou >= 0.5) & (g.point == 1)],
        "partial": g[(g.point == 1) & (g.pred_area < 0.5 * g.gt_area)],
        "background": g[(g.point == 0) & (g.mass < 0.3)],
    }
    chosen = {name: pick(df, a.k) for name, df in cats.items()}
    # misclassification: top-1 predicted class is not present in the image -> explain the *wrong* class
    mis = []
    for pos in range(len(ds)):
        top = int(np.argmax(probs[pos]))
        if top not in {b[0] for b in ds[pos]["boxes"]} and ds[pos]["boxes"]:
            mis.append((pos, top))
    chosen["misclassification"] = [mis[j] for j in rng.permutation(len(mis))[:a.k]]
    # multi-object: explain two different classes present in the same image
    multi = g.groupby("img_index").cls.nunique()
    multi = multi[multi >= 2].index.tolist()
    mrows = []
    for pos in [multi[j] for j in rng.permutation(len(multi))[:max(2, a.k // 2)]]:
        cs = sorted(g[g.img_index == pos].cls.unique())[:2]
        mrows += [(pos, int(cs[0])), (pos, int(cs[1]))]
    chosen["multiobject"] = mrows
    Path(a.fig_dir).mkdir(exist_ok=True)
    Path("results/qualitative").mkdir(parents=True, exist_ok=True)

    def title(item, c):
        pos = ids.index(item["index"])
        return f"{VOC_CLASSES[c]} (p={probs[pos, c]:.2f})\n{item['image_id']}"

    for name, rows in chosen.items():
        if rows:
            render(rows, ds, model, gc, center, f"{a.fig_dir}/fig_{name}.png", a.thr, title)
    if chosen["success"]:
        render_layers(chosen["success"], ds, model, gc, center, f"{a.fig_dir}/fig_layers.png")
    save_json({k: [[int(i), int(c)] for i, c in v] for k, v in chosen.items()},
              "results/qualitative/selected_examples.json")
    print({k: len(v) for k, v in chosen.items()})


if __name__ == "__main__":
    main()
