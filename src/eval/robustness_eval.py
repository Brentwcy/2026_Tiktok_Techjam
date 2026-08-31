#!/usr/bin/env python
"""
Robustness evaluation harness -- produces the two required deliverables:
  1. outputs/robustness_table.csv  (clean vs. each transform x severity)  [deliverable #4]
  2. outputs/error_analysis.csv    (false positives / false negatives)    [deliverable #5]

Example:
    python -m src.eval.robustness_eval \
        --test_manifest data/processed/test.csv \
        --checkpoint checkpoints/best.pt \
        --out_dir outputs
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.data.augmentations import apply_transform, iter_all_transform_settings
from src.data.constants import IMAGE_SIZE
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.models.model import build_model

POST_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class EvalDataset(Dataset):
    """Applies exactly one (transform, level) setting to every image, or none if setting is None."""

    def __init__(self, manifest_path: str, setting):
        self.df = pd.read_csv(manifest_path)
        self.setting = setting  # (name, level) or None for clean

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.setting is not None:
            name, level = self.setting
            img = apply_transform(img, name, level)
        tensor = POST_TRANSFORM(img)
        return tensor, int(row["label"]), row["image_path"]


def run_pass(model, manifest_path, setting, device, batch_size, num_workers):
    ds = EvalDataset(manifest_path, setting)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels, probs, paths = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y, p in loader:
            x = x.to(device)
            logit = model(x)
            prob = torch.sigmoid(logit).cpu().numpy()
            probs.extend(prob.tolist())
            labels.extend(y.numpy().tolist())
            paths.extend(list(p))
    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = sum(int(p == l) for p, l in zip(preds, labels)) / len(labels)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return acc, auc, labels, preds, probs, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = [("clean", None)] + [(f"{name}_{level}", (name, level))
                                     for name, level in iter_all_transform_settings()]

    table_rows = []
    error_rows = []
    for label_str, setting in tqdm(settings, desc="robustness sweep"):
        acc, auc, labels, preds, probs, paths = run_pass(
            model, args.test_manifest, setting, device, args.batch_size, args.num_workers
        )
        table_rows.append({"setting": label_str, "accuracy": acc, "auc": auc, "n": len(labels)})

        for l, pr, prob, path in zip(labels, preds, probs, paths):
            if l != pr:
                error_rows.append({
                    "setting": label_str, "image_path": path, "true_label": l,
                    "pred_label": pr, "pred_confidence": prob,
                    "error_type": "false_positive" if (l == 0 and pr == 1) else "false_negative",
                })

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(out_dir / "robustness_table.csv", index=False)
    print("\nRobustness table:")
    print(table_df.to_string(index=False))

    error_df = pd.DataFrame(error_rows)
    error_df.to_csv(out_dir / "error_analysis.csv", index=False)
    print(f"\n{len(error_df)} misclassified examples -> {out_dir / 'error_analysis.csv'}")


if __name__ == "__main__":
    main()
