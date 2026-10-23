"""Extra breakdowns computed ONLY from results/quantitative/per_pair.csv (+ summary.json); writes LaTeX tables + extra.json."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results/quantitative")
VOC = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
       "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
NICE = {"gradcam@layer4": "Grad-CAM", "gradcampp@layer4": "Grad-CAM++", "cam@layer4": "CAM", "saliency": "Saliency",
        "center": "Centre"}
pp = pd.read_csv(R / "per_pair.csv")
g = pp[pp.method == "gradcam@layer4"].reset_index(drop=True)
by = {m: pp[pp.method == m].reset_index(drop=True) for m in NICE}


def tab(cap, label, header, rows, fmt, small=False):
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    inner = (f"\\begin{{tabular}}{{{fmt}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}")
    if small:
        inner = "\\resizebox{\\linewidth}{!}{%\n" + inner + "}"
    return f"\\begin{{table}}[t]\n\\centering\n\\caption{{{cap}}}\\label{{{label}}}\n{inner}\n\\end{{table}}\n"


out = {}
# (1) predicted-box size: explains threshold calibration
rows = []
for m, n in NICE.items():
    d = by[m]
    rows.append([n, f"{100*d.pred_area.median():.1f}", f"{100*d.pred_area.mean():.1f}", f"{100*(d.pred_area > 0.9).mean():.1f}",
                 f"{100*(d.iou >= 0.5).mean():.1f}"])
out["gt_area_mean"] = float(g.gt_area.mean()); out["gt_area_median"] = float(g.gt_area.median())
(R / "boxarea_table.tex").write_text(tab(
    f"Size of the predicted box at $\\tau=0.15$ as a percentage of the image area (ground-truth boxes cover on average "
    f"{100*g.gt_area.mean():.1f}\\% of the image, median {100*g.gt_area.median():.1f}\\%). ``$>90\\%$'' = share of pairs whose box covers more than "
    f"90\\% of the image.", "tab:boxarea", ["Method", "Median area (\\%)", "Mean area (\\%)", "$>90\\%$ of image (\\%)", "Loc. acc. (\\%)"],
    rows, "lrrrr"))
# (2) by object size
bins = [0, 0.1, 0.3, 0.6, 1.0001]; lab = ["$<10\\%$", "10--30\\%", "30--60\\%", "$>60\\%$"]
g["bin"] = pd.cut(g.gt_area, bins, right=False, labels=lab)
rows = []
for b in lab:
    idx = g.index[g.bin == b]
    r = [b, len(idx)]
    for m in ("gradcam@layer4", "gradcampp@layer4", "cam@layer4", "center"):
        r.append(f"{100*(by[m].loc[idx, 'iou'] >= 0.5).mean():.1f}")
    r.append(f"{100*g.loc[idx, 'point'].mean():.1f}"); r.append(f"{100*by['center'].loc[idx, 'point'].mean():.1f}")
    rows.append(r)
(R / "size_table.tex").write_text(tab(
    "Localisation accuracy (IoU $\\ge0.5$, \\%) by relative size of the ground-truth region (union of the class' boxes as a share of the image), "
    "and pointing accuracy (\\%) for Grad-CAM and the centre control, $\\tau=0.15$.", "tab:size",
    ["GT size", "\\# pairs", "Grad-CAM", "Grad-CAM++", "CAM", "Centre", "Point: Grad-CAM", "Point: Centre"], rows, "lrrrrrrr", small=True))
# (3) by classifier confidence for the explained class
pb = [0, 0.1, 0.5, 0.9, 1.0001]; pl = ["$<0.1$", "0.1--0.5", "0.5--0.9", "$\\ge0.9$"]
g["pbin"] = pd.cut(g.prob, pb, right=False, labels=pl)
rows = []
for b in pl:
    d = g[g.pbin == b]
    rows.append([b, len(d), f"{100*(d.iou >= 0.5).mean():.1f}", f"{100*d.point.mean():.1f}", f"{100*d.mass.mean():.1f}",
                 int((d.mass == 0).sum())])
out["gradcam_pred_area_by_gt_bin"] = {str(b): {"n": int((g.bin == b).sum()), "mean_pred_area": float(g.pred_area[g.bin == b].mean()),
                                                "mean_gt_area": float(g.gt_area[g.bin == b].mean())} for b in lab}
out["n_prob_lt_0.5"] = int((g.prob < 0.5).sum())
(R / "confidence_table.tex").write_text(tab(
    "Grad-CAM (layer4, $\\tau=0.15$) by the classifier's probability for the ground-truth class being explained. "
    "``Zero mass'' = pairs where the map has exactly zero energy inside the ground-truth boxes.", "tab:conf",
    ["$p(\\text{class})$", "\\# pairs", "Loc. acc. (\\%)", "Pointing (\\%)", "Mass (\\%)", "Zero mass"], rows, "lrrrrr"))
# (4) per class
rows = []
for c, name in enumerate(VOC):
    idx = g.index[g.cls == c]
    if len(idx) == 0:
        continue
    rows.append([name, len(idx)] + [f"{100*(by[m].loc[idx, 'iou'] >= 0.5).mean():.1f}" for m in ("gradcam@layer4", "cam@layer4", "center")]
                + [f"{100*g.loc[idx, 'point'].mean():.1f}", f"{100*g.loc[idx, 'gt_area'].mean():.0f}"])
(R / "class_table.tex").write_text(tab(
    "Per-class results ($\\tau=0.15$): localisation accuracy (\\%) of Grad-CAM, CAM and the centre control, Grad-CAM pointing accuracy (\\%), "
    "and mean ground-truth region size (\\% of image).", "tab:class",
    ["Class", "\\# pairs", "Loc: Grad-CAM", "Loc: CAM", "Loc: Centre", "Point: Grad-CAM", "GT size"], rows, "lrrrrrr", small=True))
# (5) category rates (the rules behind the qualitative figures)
succ = (g.iou >= 0.5) & (g.point == 1)
part = (g.point == 1) & (g.pred_area < 0.5 * g.gt_area)
back = (g.point == 0) & (g.mass < 0.3)
out["categories"] = {k: int(v.sum()) for k, v in {"success": succ, "partial": part, "background": back}.items()}
out["categories"]["n_pairs"] = int(len(g))
out["pointing_hit_but_iou_lt_0.5"] = int(((g.point == 1) & (g.iou < 0.5)).sum())
e = {m: int((by[m].mass == 0).sum()) for m in NICE}
out["zero_mass_pairs"] = e
out["zero_mass_max_prob"] = float(g.prob[g.mass == 0].max())
a, c = by["gradcam@layer4"], by["cam@layer4"]
out["pointing_disagreements_gradcam_vs_cam"] = int((a.point != c.point).sum())
out["pointing_disagreement_gradcam_mass"] = a.mass[a.point != c.point].tolist()
M = pd.read_csv(R / "metrics_by_threshold.csv")
best = M.loc[M.groupby("method").loc_acc.idxmax()][["method", "thr", "loc_acc", "loc_acc_lo", "loc_acc_hi"]]
out["best_threshold"] = best.set_index("method").to_dict("index")
json.dump(out, open(R / "extra.json", "w"), indent=2)
print(json.dumps(out, indent=1))
