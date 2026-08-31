"""
Implements the exact robustness transforms from the Track 5 problem statement
(section 5.2), each as a pure function: PIL.Image -> PIL.Image.

Design notes / assumptions (call these out in the README limitations section):
- "Resize scale 0.5x / 0.25x then upscale": downscale then upscale back to the
  original size with bilinear interpolation, mimicking a thumbnail round-trip.
- "Center crop 80%": interpreted as cropping to 80% of each side length
  (not 80% area), then resizing back to the original size so images still
  batch together. If judges intended 80% area, swap KEEP_FRACTION handling
  in center_crop() for math.sqrt(keep_fraction).
- "Gaussian noise sigma": applied in normalized [0, 1] pixel space, matching
  common robustness-benchmark convention (e.g. ImageNet-C).
"""

import io
import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter

from src.data.constants import ROBUSTNESS_TRANSFORMS


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_roundtrip(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def color_jitter(img: Image.Image, strength: float = 0.20) -> Image.Image:
    jitter = ColorJitter(brightness=strength, contrast=strength, saturation=strength)
    return jitter(img.convert("RGB"))


def center_crop(img: Image.Image, keep_fraction: float = 0.80) -> Image.Image:
    w, h = img.size
    new_w, new_h = int(w * keep_fraction), int(h * keep_fraction)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)


TRANSFORM_FNS = {
    "jpeg": lambda img, level: jpeg_compress(img, quality=level),
    "blur": lambda img, level: gaussian_blur(img, sigma=level),
    "resize": lambda img, level: resize_roundtrip(img, scale=level),
    "noise": lambda img, level: gaussian_noise(img, sigma=level),
    "color_jitter": lambda img, level: color_jitter(img, strength=level),
    "center_crop": lambda img, level: center_crop(img, keep_fraction=level),
}


def apply_transform(img: Image.Image, name: str, level) -> Image.Image:
    return TRANSFORM_FNS[name](img, level)


def iter_all_transform_settings():
    """Yield (name, level) for every (transform, severity) cell in the robustness table."""
    for name, spec in ROBUSTNESS_TRANSFORMS.items():
        for level in spec["levels"]:
            yield name, level


class RandomRobustnessAugment:
    """
    Training-time augmentation: with probability p, apply one randomly chosen
    (transform, severity) pair from the robustness table. Otherwise pass the
    image through clean. This is what actually buys robustness at eval time --
    the model has to have seen these degradations during training.
    """

    def __init__(self, p: float = 0.5):
        self.p = p
        self.settings = list(iter_all_transform_settings())

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        name, level = random.choice(self.settings)
        return apply_transform(img, name, level)
