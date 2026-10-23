"""Turn the measured CSV/JSON files into LaTeX tables and the threshold-sweep figure (no numbers typed by hand)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path("results/quantitative")
NICE = {"gradcam@layer4": "Grad-CAM (layer4)", "gradcampp@layer4": "Grad-CAM++ (layer4)", "cam@layer4": "CAM (layer4)",
        "saliency": "Vanilla saliency", "center": "Centre Gaussian (no model)",
        "gradcam@layer2": "Grad-CAM (layer2)", "gradcam@layer3": "Grad-CAM (layer3)"}
MAIN = ["gradcam@layer4", "gradcampp@layer4", "cam@layer4", "saliency", "center"]


def w(name, text):
    (R / name).write_text(text)


def table(caption, label, header, rows, colfmt):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    tab = (f"\\begin{{tabular}}{{{colfmt}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}")
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\resizebox{{\\linewidth}}{{!}}{{%\n{tab}}}\n\\end{{table}}\n")


def main():
    S = json.load(open(R / "summary.json"))
    M = pd.read_csv(R / "metrics_by_threshold.csv")
    t0 = S["thr_main"]
    timing = json.load(open(R / "timing.json")) if (R / "timing.json").exists() else None
    tmap = {"gradcam@layer4": "Grad-CAM (layer4)", "gradcampp@layer4": "Grad-CAM++ (layer4)", "cam@layer4": "CAM (layer4)",
            "saliency": "Vanilla saliency"}

    def row(k):
        m = M[(M.method == k) & (M.thr == t0)].iloc[0]
        s = S["methods"][k]
        ms = "--"
        if timing and k in tmap and tmap[k] in timing["results"]:
            ms = f"{timing['results'][tmap[k]]['ms_mean']:.1f}"
        return [NICE[k], f"{100*m.loc_acc:.1f} [{100*m.loc_acc_lo:.1f}, {100*m.loc_acc_hi:.1f}]", f"{m.mean_iou:.3f}",
                f"{100*s['point']:.1f}", f"{100*s['mass']:.1f}", f"{100*s['empty']:.1f}", ms]

    w("main_table.tex", table(
        f"Weakly-supervised localisation on VOC2007 test ({S['n_images']} images, {S['n_pairs']} image--class pairs; "
        f"threshold {t0:g} of the map maximum; brackets: 95\\% bootstrap CI). Loc.\\ acc.\\ = fraction of pairs with "
        f"box IoU $\\ge 0.5$; Mass = share of heat-map energy inside ground-truth boxes; Empty = map identically zero (counted as failure); time = ms per explanation "
        f"(batch size 1, measured separately, see Table~\\ref{{tab:timing}}).",
        "tab:main", ["Method", "Loc. acc.@0.5 (\\%)", "Mean IoU", "Pointing (\\%)", "Mass in GT (\\%)", "Empty maps (\\%)", "Time (ms)"],
        [row(k) for k in MAIN], "lrrrrrr"))
    # threshold sweep
    piv = M[M.method.isin(MAIN)].pivot(index="method", columns="thr", values="loc_acc").loc[MAIN]
    thrs = list(piv.columns)
    w("threshold_table.tex", table(
        "Sensitivity to the binarisation threshold: localisation accuracy (IoU $\\ge 0.5$, \\%) of each method.",
        "tab:thr", ["Method"] + [f"{t:g}" for t in thrs],
        [[NICE[k]] + [f"{100*v:.1f}" for v in piv.loc[k]] for k in MAIN], "l" + "r" * len(thrs)))
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for k in MAIN:
        ax.plot(thrs, 100 * piv.loc[k].values, marker="o", label=NICE[k])
    ax.set_xlabel("threshold (fraction of max)"); ax.set_ylabel("Loc. acc. @ IoU$\\geq$0.5 (%)"); ax.legend(fontsize=7)
    ax.grid(alpha=.3); fig.tight_layout(); Path("figures").mkdir(exist_ok=True); fig.savefig("figures/threshold_sweep.png", dpi=150)
    # layer ablation
    L = ["gradcam@layer2", "gradcam@layer3", "gradcam@layer4"]
    res = {"layer2": 56, "layer3": 28, "layer4": 14}
    rows = []
    for k in L:
        m = M[(M.method == k) & (M.thr == t0)].iloc[0]
        s = S["methods"][k]
        grid = S["img_size"] // {"layer2": 8, "layer3": 16, "layer4": 32}[k.split("@")[1]]
        rows.append([NICE[k], f"${grid}\\times{grid}$", f"{100*m.loc_acc:.1f}", f"{m.mean_iou:.3f}",
                     f"{100*s['point']:.1f}", f"{100*s['mass']:.1f}"])
    w("layer_table.tex", table(
        f"Layer-selection ablation (ResNet-50, input ${S['img_size']}\\times{S['img_size']}$, threshold {t0:g}).",
        "tab:layers", ["Target layer", "Map size", "Loc. acc. (\\%)", "Mean IoU", "Pointing (\\%)", "Mass (\\%)"], rows, "lrrrrr"))
    # discrimination
    D = pd.read_csv(R / "discrimination.csv")
    w("discrimination_table.tex", table(
        "Class-discrimination index on images that contain $\\ge 2$ classes (0.5 = the map cannot tell the classes apart; "
        "1 = all heat on the explained class' own box).", "tab:disc", ["Method", "\\# pairs", "Index [95\\% CI]"],
        [[NICE[r.method], str(int(r.n_pairs)), f"{r.index:.3f} [{r.lo:.3f}, {r.hi:.3f}]"] for r in D.itertuples()], "lrr"))
    # paired differences
    P = pd.read_csv(R / "paired_diff.csv")
    P = P[P.comparison.str.contains("|".join(MAIN[1:]))]
    w("paired_table.tex", table(
        f"Paired bootstrap differences (Grad-CAM layer4 minus method) at threshold {t0:g}, in percentage points.",
        "tab:paired", ["Comparison", "Metric", "Diff.", "95\\% CI"],
        [[NICE.get(r.comparison.split(" - ")[1], r.comparison), r.metric.replace("_", "\\_"), f"{100*r.diff:+.1f}",
          f"[{100*r.lo:+.1f}, {100*r.hi:+.1f}]"] for r in P.itertuples()], "llrr"))
    # timing
    if timing:
        rows = [[k, f"{v['ms_mean']:.1f} $\\pm$ {v['ms_std']:.1f}",
                 "--" if v["peak_mem_MB"] is None else f"{v['peak_mem_MB']:.0f}"] for k, v in timing["results"].items()]
        w("timing_table.tex", table(
            f"Measured cost per image (batch 1, {timing['img_size']}$\\times${timing['img_size']} input, "
            f"{timing['runs']} runs after warm-up, device: {timing['env']['device']}). Parameters: {timing['params']:,}.",
            "tab:timing", ["Method", "Time (ms)", "Peak GPU mem. (MB)"], rows, "lrr"))
    # setup / training facts
    tl = json.load(open(R / "train_log.json")) if (R / "train_log.json").exists() else None
    env = S["env"]
    lines = [f"Python {env['python']}", f"PyTorch {env['torch']}", f"torchvision {env['torchvision']}", f"device: {env['device']}"]
    if tl:
        lines += [f"train/val images: {tl['n_train']}/{tl['n_val']}", f"best val mAP: {tl['best_val_mAP']:.4f} (epoch {S['ckpt_epoch']})"]
    lines += [f"test mAP: {S['test_mAP']:.4f}", f"test images with labels: {S['n_images']}", f"image--class pairs: {S['n_pairs']}"]
    w("setup_facts.tex", "\\begin{itemize}\n" + "\n".join(f"\\item {l}" for l in lines) + "\n\\end{itemize}\n")
    print("tables written to", R)


if __name__ == "__main__":
    main()
