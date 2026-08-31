#!/usr/bin/env python
"""
Fine-tune the AIGC detector.

Example:
    python -m src.train \
        --train_manifest data/processed/train.csv \
        --val_manifest data/processed/val.csv \
        --backbone vit_base_patch16_clip_224.openai \
        --epochs 5 --batch_size 32 --lr 3e-5 \
        --checkpoint_out checkpoints/best.pt
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import AIGCManifestDataset
from src.models.model import build_model


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = sum(int(p == l) for p, l in zip(preds, all_labels)) / len(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")  # only one class present in this split
    return acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--backbone", default="vit_base_patch16_clip_224.openai")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--augment_p", type=float, default=0.5,
                     help="probability of applying a random robustness transform per training image")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=1,
                     help="train only the classifier head for this many initial epochs")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--checkpoint_out", default="checkpoints/best.pt")
    ap.add_argument("--log_csv", default="outputs/train_log.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = AIGCManifestDataset(args.train_manifest, train=True, augment_p=args.augment_p)
    val_ds = AIGCManifestDataset(args.val_manifest, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.backbone, pretrained=True,
                         freeze_backbone=args.freeze_backbone_epochs > 0).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
    log_rows = []
    best_auc = -1.0

    for epoch in range(args.epochs):
        if epoch == args.freeze_backbone_epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            print("unfroze backbone")

        model.train()
        running_loss, n = 0.0, 0
        t0 = time.time()
        for x, y, _ in tqdm(train_loader, desc=f"epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = running_loss / n
        val_acc, val_auc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} ({elapsed:.0f}s)")
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc,
                          "val_auc": val_auc, "seconds": elapsed})

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model_state": model.state_dict(), "backbone": args.backbone,
                        "epoch": epoch, "val_auc": float(val_auc)}, args.checkpoint_out)
            print(f"  saved new best checkpoint (val_auc={val_auc:.4f}) -> {args.checkpoint_out}")

    with open(args.log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)


if __name__ == "__main__":
    main()
