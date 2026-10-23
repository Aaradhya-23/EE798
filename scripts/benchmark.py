"""Wall-clock / memory / parameter-count measurements for each explanation method."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cam import cam_heatmap
from src.gradcam import GradCAM
from src.model import load_checkpoint, target_layers
from src.saliency import vanilla_saliency
from src.utils import env_info, get_device, save_json, set_seed


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/resnet50_voc07.pt")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="results/quantitative/timing.json")
    a = ap.parse_args()
    set_seed(0)
    dev = get_device()
    model, ck = load_checkpoint(a.ckpt, dev)
    size = ck["args"]["img_size"]
    x = torch.randn(1, 3, size, size, device=dev)      # timing does not depend on pixel content
    layers = target_layers(model)
    gc4 = GradCAM(model, {"layer4": layers["layer4"]})
    gc2 = GradCAM(model, {"layer2": layers["layer2"]})
    c = 0

    def fwd():
        with torch.no_grad():
            model(x)

    def gradcam4():
        gc4(x, c, pp=False)

    def gradcampp4():
        gc4(x, c, pp=True)

    def gradcam2():
        gc2(x, c, pp=False)

    def cam():
        with torch.no_grad():
            gc4._acts.clear()
            gc4.model(x)
            A = gc4._acts["layer4"]
            cam_heatmap(model.fc.weight, A, c, x.shape[-2:])

    def sal():
        vanilla_saliency(model, x, c)

    fns = {"forward only": fwd, "Grad-CAM (layer4)": gradcam4, "Grad-CAM (layer2)": gradcam2,
           "Grad-CAM++ (layer4)": gradcampp4, "CAM (layer4)": cam, "Vanilla saliency": sal}
    res = {}
    for name, fn in fns.items():
        for _ in range(a.warmup):
            fn()
        sync(dev)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(a.runs):
            sync(dev); t0 = time.perf_counter(); fn(); sync(dev); ts.append((time.perf_counter() - t0) * 1000)
        res[name] = {"ms_mean": float(np.mean(ts)), "ms_std": float(np.std(ts)),
                     "peak_mem_MB": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None}
        print(name, res[name], flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    save_json({"img_size": size, "runs": a.runs, "params": n_par, "results": res, "env": env_info()}, a.out)


if __name__ == "__main__":
    main()
