import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from data import GAICD, load_split
from model import GAIC


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    """GAICD protocol: mean per-image SRCC and Acc_K/N (K=1..4, N=5,10) in percent."""
    model.eval()
    srcc, hits = [], {n: np.zeros(4) for n in (5, 10)}
    for x, boxes, mos in loader:
        pred = model(x.to(device), boxes.to(device)).cpu().numpy()
        gt = mos.numpy()
        srcc.append(spearmanr(pred, gt).statistic)
        pred_rank = np.argsort(-pred)
        for n in hits:
            top_gt = set(np.argsort(-gt)[:n].tolist())
            for k in range(1, 5):
                hits[n][k - 1] += np.mean([c in top_gt for c in pred_rank[:k]])
    count = len(loader)
    metrics = {"srcc": float(np.mean(srcc))}
    for n, h in hits.items():
        acc = h / count * 100
        metrics.update({f"acc{k}/{n}": float(acc[k - 1]) for k in range(1, 5)})
        metrics[f"acc{n}"] = float(acc.mean())
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/GAIC")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val", type=int, default=100, help="train images held out for checkpoint selection")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_items = load_split(args.data, "train")
    random.Random(args.seed).shuffle(train_items)
    val_items, train_items = train_items[: args.val], train_items[args.val:]
    test_items = load_split(args.data, "test")
    all_mos = np.concatenate([m for _, _, m in train_items])
    mos_mean, mos_std = float(all_mos.mean()), float(all_mos.std(ddof=1))
    print(f"device={device} train={len(train_items)} val={len(val_items)} test={len(test_items)} "
          f"mos={mos_mean:.3f}±{mos_std:.3f}")

    def loader(items, train):
        ds = GAICD(items, mos_mean, mos_std, train)
        return DataLoader(ds, batch_size=None, shuffle=train, num_workers=args.workers,
                          persistent_workers=True)

    train_dl, val_dl, test_dl = loader(train_items, True), loader(val_items, False), loader(test_items, False)

    model = GAIC().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.05)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        # One image per step: keep ImageNet BatchNorm statistics frozen.
        model.apply(lambda m: m.eval() if isinstance(m, torch.nn.BatchNorm2d) else None)
        t0, total = time.time(), 0.0
        for x, boxes, mos in train_dl:
            loss = F.huber_loss(model(x.to(device), boxes.to(device)), mos.to(device), delta=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        sched.step()
        val = evaluate(model, val_dl, device)
        select = val["srcc"] * 100 + val["acc5"]
        tag = ""
        if select > best:
            best = select
            torch.save({"model": model.state_dict(), "mos_mean": mos_mean, "mos_std": mos_std, "epoch": epoch},
                       out / "best.pt")
            tag = " *"
        print(f"epoch {epoch:3d} loss {total / len(train_dl):.4f} val srcc {val['srcc']:.3f} "
              f"acc5 {val['acc5']:.1f} acc10 {val['acc10']:.1f} ({time.time() - t0:.0f}s){tag}", flush=True)

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_dl, device)
    test["epoch"] = ckpt["epoch"]
    (out / "test_metrics.json").write_text(json.dumps(test, indent=2))
    print("test", json.dumps(test))


if __name__ == "__main__":
    main()
