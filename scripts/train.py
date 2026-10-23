"""Fine-tune ResNet-50 (layer4 + new 20-way head) on VOC2007 trainval (90/10 train/val split)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import VOCLocalization, collate
from src.model import build_model
from src.utils import env_info, get_device, save_json, set_seed


def mean_ap(y_true, y_score):
    aps = [average_precision_score(y_true[:, c], y_score[:, c]) for c in range(y_true.shape[1])
           if y_true[:, c].sum() > 0]
    return float(np.mean(aps))


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    ys, ps = [], []
    for b in loader:
        ps.append(torch.sigmoid(model(b["image"].to(dev))).cpu().numpy())
        ys.append(b["labels"].numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-layer4", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="checkpoints/resnet50_voc07.pt")
    ap.add_argument("--max-train", type=int, default=0, help="debug: limit #train images")
    ap.add_argument("--no-pretrained", action="store_true", help="debug only")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()

    set_seed(a.seed)
    dev = get_device()
    dl = not a.no_download
    n = len(VOCLocalization(a.root, "trainval", a.img_size, download=dl))
    perm = np.random.RandomState(a.seed).permutation(n)
    n_val = int(0.1 * n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if a.max_train:
        train_idx, val_idx = train_idx[:a.max_train], val_idx[:max(8, a.max_train // 4)]
    mk = lambda idx, tr: DataLoader(VOCLocalization(a.root, "trainval", a.img_size, idx, train=tr, download=False),
                                    batch_size=a.bs, shuffle=tr, num_workers=a.workers, collate_fn=collate)
    train_loader, val_loader = mk(train_idx, True), mk(val_idx, False)

    model = build_model(pretrained=not a.no_pretrained).to(dev)
    for name, p in model.named_parameters():                   # freeze everything except layer4 and fc
        p.requires_grad = name.startswith(("layer4", "fc"))
    opt = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": a.lr_layer4},
        {"params": model.fc.parameters(), "lr": a.lr_head}], weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    crit = nn.BCEWithLogitsLoss()

    log, best = [], -1.0
    for ep in range(a.epochs):
        model.train()
        for m in (model.conv1, model.bn1, model.layer1, model.layer2, model.layer3):
            m.eval()                                            # frozen stages keep ImageNet BN statistics
        t0, tot = time.time(), 0.0
        for b in train_loader:
            x, y = b["image"].to(dev), b["labels"].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * x.size(0)
        yv, pv = predict(model, val_loader, dev)
        v_map = mean_ap(yv, pv)
        log.append({"epoch": ep + 1, "train_loss": tot / len(train_idx), "val_mAP": v_map,
                    "seconds": time.time() - t0})
        print(log[-1], flush=True)
        if v_map > best:
            best = v_map
            Path(a.ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "args": vars(a), "val_mAP": v_map, "epoch": ep + 1}, a.ckpt)
    save_json({"args": vars(a), "n_train": len(train_idx), "n_val": len(val_idx), "log": log,
               "best_val_mAP": best, "env": env_info()}, "results/quantitative/train_log.json")


if __name__ == "__main__":
    main()
