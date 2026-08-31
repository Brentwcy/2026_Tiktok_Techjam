#!/usr/bin/env python
"""
Required inference script (Track 5 deliverable #2): takes a directory of
images and outputs a JSON file with an image_path and pred (confidence that
the image is AIGC-generated, in [0, 1]) for each image.

Usage:
    python scripts/infer.py --input_dir path/to/images \
        --checkpoint checkpoints/best.pt \
        --output preds.json
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.data.constants import IMAGE_SIZE
from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.models.model import build_model

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", default="preds.json", type=Path)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    paths = sorted(p for p in args.input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise SystemExit(f"no images found under {args.input_dir}")

    results = []
    batch_imgs, batch_paths = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy()
        for p, prob in zip(batch_paths, probs):
            results.append({"image_path": str(p), "pred": float(prob)})
        batch_imgs.clear()
        batch_paths.clear()

    for p in tqdm(paths, desc="inference"):
        img = Image.open(p).convert("RGB")
        batch_imgs.append(TRANSFORM(img))
        batch_paths.append(p)
        if len(batch_imgs) == args.batch_size:
            flush()
    flush()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
