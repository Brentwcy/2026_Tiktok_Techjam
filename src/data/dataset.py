"""
Manifest-driven dataset. A manifest is just a CSV with columns:
    image_path,label
where label is 0 (real) or 1 (AIGC), matching src.data.constants.LABEL_*.

Build manifests with scripts/build_manifest.py once datasets are downloaded --
this class doesn't care which source dataset (SID_Set, CIFAKE, ...) a row
came from.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.augmentations import RandomRobustnessAugment
from src.data.constants import IMAGE_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AIGCManifestDataset(Dataset):
    def __init__(self, manifest_path: str, train: bool = True, augment_p: float = 0.5):
        self.df = pd.read_csv(manifest_path)
        assert {"image_path", "label"}.issubset(self.df.columns), (
            f"manifest {manifest_path} must have columns image_path,label"
        )
        self.train = train
        self.robust_aug = RandomRobustnessAugment(p=augment_p) if train else None

        resize_and_tensor = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.post_transform = transforms.Compose(resize_and_tensor)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = int(row["label"])

        img = Image.open(path).convert("RGB")
        if self.robust_aug is not None:
            img = self.robust_aug(img)
        tensor = self.post_transform(img)
        return tensor, torch.tensor(label, dtype=torch.float32), path


def load_manifest(manifest_path: str) -> pd.DataFrame:
    return pd.read_csv(manifest_path)
