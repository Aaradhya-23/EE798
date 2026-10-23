"""Quantitative evaluation on VOC2007 test: localisation of Grad-CAM vs. baselines + ablations."""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import mean_ap, predict
from src.cam import cam_heatmap
from src.dataset import VOC_CLASSES, VOCLocalization, collate
from src.evaluation import (bootstrap_ci, center_gaussian, discrimination_index, paired_diff_ci, score_heatmap)
from src.gradcam import GradCAM
from src.model import load_checkpoint, target_layers
from src.saliency import vanilla_saliency
from src.utils import env_info, get_device, save_json, set_seed

THRS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
MAIN = ["gradcam@layer4", "gradcampp@layer4", "cam@layer4", "saliency", "center"]
ABLATION = ["gradcam@layer2", "gradcam@layer3", "gradcam@layer4"]


def explain_all(gc, model, x, c, size, center):
    res = gc(x, c)
    out = {f"gradcam@{l}": res["gradcam"][l] for l in res["gradcam"]}
    out["gradcampp@layer4"] = res["gradcampp"]["layer4"]
    out["cam@layer4"] = cam_heatmap(model.fc.weight.detach(), gc.A["layer4"], c, size)
    out["saliency"] = vanilla_saliency(model, x, c)
    maps = {k: v[0, 0].detach().cpu().numpy() for k, v in out.items()}
    maps["center"] = center
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--ckpt", default="checkpoints/resnet50_voc07.pt")
    ap.add_argument("--out", default="results/quantitative")
    ap.add_argument("--max-images", type=int, default=0, help="debug: evaluate only the first N test images")
    ap.add_argument("--thr-main", type=float, default=0.15)
    ap.add_argument("--tol", type=int, default=0, help="pointing-game tolerance in pixels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()

    set_seed(a.seed)
    dev = get_device()
    model, ck = load_checkpoint(a.ckpt, dev)
    size = ck["args"]["img_size"]
    ds = VOCLocalization(a.root, "test", size, download=not a.no_download)
    if a.max_images:
        ds.indices = ds.indices[:a.max_images]
    # --- classifier quality (context for the explanations) + probabilities for qualitative selection
    yt, pt = predict(model, DataLoader(ds, batch_size=32, num_workers=2, collate_fn=collate), dev)
    np.save(f"{a.out}/test_probs.npy", pt)
    test_map = mean_ap(yt, pt)
    print(f"classifier test mAP = {test_map:.4f}", flush=True)

    gc = GradCAM(model, target_layers(model))
    center = center_gaussian((size, size))
    keys = MAIN + [k for k in ABLATION if k not in MAIN]
    agg = {k: defaultdict(list) for k in keys}
    rows, disc = [], defaultdict(list)
    n_img = 0
    for i, b in enumerate(DataLoader(ds, batch_size=1, num_workers=2, collate_fn=collate)):
        boxes = b["boxes"][0]
        classes = sorted({bx[0] for bx in boxes})
        if not classes:
            continue
        n_img += 1
        x = b["image"].to(dev)
        maps_c = {}
        for c in classes:
            gt = [bx[1:] for bx in boxes if bx[0] == c]
            maps = explain_all(gc, model, x, c, x.shape[-2:], center)
            maps_c[c] = maps
            for k in keys:
                s = score_heatmap(maps[k], gt, THRS, a.tol)
                for t in THRS:
                    agg[k][f"iou@{t}"].append(s["iou"][t])
                agg[k]["point"].append(float(s["point"]))
                agg[k]["empty"].append(float(s["empty"]))
                agg[k]["mass"].append(s["mass"])
                rows.append({"img_index": i, "image_id": b["image_id"][0], "cls": c, "cls_name": VOC_CLASSES[c],
                             "method": k, "iou": s["iou"][a.thr_main], "point": int(s["point"]), "mass": s["mass"],
                             "pred_area": s["area"][a.thr_main], "gt_area": s["gt_area"],
                             "prob": float(pt[i, c])})
        for c in classes:                                    # class-discriminativeness on multi-class images
            for c2 in classes:
                if c2 == c:
                    continue
                own = [bx[1:] for bx in boxes if bx[0] == c]
                oth = [bx[1:] for bx in boxes if bx[0] == c2]
                for k in MAIN:
                    d = discrimination_index(maps_c[c][k], own, oth)
                    if d is not None:
                        disc[k].append(d)
        if n_img % 200 == 0:
            print(f"{n_img} images processed", flush=True)

    # --- aggregate
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_pairs = len(agg[keys[0]]["point"])
    with open(out / "metrics_by_threshold.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "thr", "n_pairs", "loc_acc", "loc_acc_lo", "loc_acc_hi", "mean_iou", "mean_iou_lo", "mean_iou_hi"])
        for k in keys:
            for t in THRS:
                iou = np.array(agg[k][f"iou@{t}"])
                acc = (iou >= 0.5).astype(float)
                (alo, ahi), (ilo, ihi) = bootstrap_ci(acc), bootstrap_ci(iou)
                w.writerow([k, t, n_pairs, acc.mean(), alo, ahi, iou.mean(), ilo, ihi])
    summ = {"n_images": n_img, "n_pairs": n_pairs, "test_mAP": test_map, "thr_main": a.thr_main, "tol": a.tol,
            "img_size": size, "ckpt_epoch": ck["epoch"], "methods": {}}
    for k in keys:
        pt_, ms_ = np.array(agg[k]["point"]), np.array(agg[k]["mass"])
        summ["methods"][k] = {"point": float(pt_.mean()), "point_ci": bootstrap_ci(pt_),
                              "mass": float(ms_.mean()), "mass_ci": bootstrap_ci(ms_),
                              "empty": float(np.mean(agg[k]["empty"]))}
    base = (np.array(agg["gradcam@layer4"][f"iou@{a.thr_main}"]) >= 0.5).astype(float)
    with open(out / "paired_diff.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "metric", "diff", "lo", "hi"])
        for k in keys:
            if k == "gradcam@layer4":
                continue
            other = (np.array(agg[k][f"iou@{a.thr_main}"]) >= 0.5).astype(float)
            w.writerow([f"gradcam@layer4 - {k}", "loc_acc@0.5", *paired_diff_ci(base, other)])
            w.writerow([f"gradcam@layer4 - {k}", "pointing", *paired_diff_ci(agg["gradcam@layer4"]["point"], agg[k]["point"])])
    with open(out / "discrimination.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_pairs", "index", "lo", "hi"])
        for k in MAIN:
            if disc[k]:
                w.writerow([k, len(disc[k]), np.mean(disc[k]), *bootstrap_ci(disc[k])])
    with open(out / "per_pair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summ["env"] = env_info()
    save_json(summ, out / "summary.json")
    print("done:", n_img, "images,", n_pairs, "image-class pairs")


if __name__ == "__main__":
    main()
