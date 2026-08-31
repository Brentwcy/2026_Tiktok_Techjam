#!/usr/bin/env python
"""
Build a unified train/val manifest CSV (image_path,label) from one or more
downloaded datasets.

IMPORTANT: never point this at the WildFake demo subset (COCO val2017 reals +
DALL-E Advanced fakes) -- that split is reserved by the problem statement as a
held-out benchmark, not training data. Build it with this same script, but into
a separate --add_to file (e.g. data/processed/demo.csv) that never gets merged
into train.csv/val.csv.

Usage (generic real/fake folder layout, e.g. CIFAKE's train/REAL, train/FAKE):
    python scripts/build_manifest.py \
        --root data/raw/cifake/train --real_subdir REAL --fake_subdir FAKE \
        --add_to data/processed/train_manifest.csv

Run once per source dataset with --add_to pointing at the same file to
accumulate a combined manifest, then split it:
    python scripts/build_manifest.py --split data/processed/train_manifest.csv \
        --val_fraction 0.15 \
        --train_out data/processed/train.csv --val_out data/processed/val.csv
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.model_selection import train_test_split

from src.data.constants import LABEL_AIGC, LABEL_REAL

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scan_binary_folders(root: Path, real_subdir: str, fake_subdir: str):
    rows = []
    for subdir, label in [(real_subdir, LABEL_REAL), (fake_subdir, LABEL_AIGC)]:
        folder = root / subdir
        if not folder.exists():
            raise FileNotFoundError(f"expected folder {folder} not found")
        for p in folder.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                rows.append((str(p.resolve()), label))
    return rows


def append_rows(csv_path: Path, rows):
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["image_path", "label"])
        writer.writerows(rows)


def split_manifest(src_csv: Path, val_fraction: float, train_out: Path, val_out: Path, seed: int = 42):
    import pandas as pd

    df = pd.read_csv(src_csv).drop_duplicates(subset="image_path")
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, random_state=seed, stratify=df["label"]
    )
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    print(f"train={len(train_df)} val={len(val_df)} -> {train_out}, {val_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, help="dataset root containing real/fake subfolders")
    ap.add_argument("--real_subdir", default="real")
    ap.add_argument("--fake_subdir", default="fake")
    ap.add_argument("--add_to", type=Path, help="manifest CSV to append scanned rows to")

    ap.add_argument("--split", type=Path, help="existing manifest CSV to split into train/val")
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--train_out", type=Path)
    ap.add_argument("--val_out", type=Path)
    args = ap.parse_args()

    if args.split:
        split_manifest(args.split, args.val_fraction, args.train_out, args.val_out)
        return

    rows = scan_binary_folders(args.root, args.real_subdir, args.fake_subdir)
    print(f"found {len(rows)} images under {args.root}")
    append_rows(args.add_to, rows)
    print(f"appended to {args.add_to}")


if __name__ == "__main__":
    main()
