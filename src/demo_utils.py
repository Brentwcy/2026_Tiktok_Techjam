"""Shared inference helpers for the Gradio and Streamlit demos."""

import urllib.request
from pathlib import Path

import torch
from PIL import Image

from scripts.infer import TRANSFORM, load_model


DEFAULT_CHECKPOINT = Path("checkpoints/best.pt")
CHECKPOINT_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1l4Kw8aW6vv8uzTmvM3Lzhqfz_zchWUpN&export=download&confirm=t"
)
MIN_CHECKPOINT_BYTES = 10_000_000


def ensure_checkpoint(path: Path = DEFAULT_CHECKPOINT) -> Path:
    """Download the public checkpoint when it is not already available."""
    if path.exists() and path.stat().st_size >= MIN_CHECKPOINT_BYTES:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_suffix(path.suffix + ".part")
    partial_path.unlink(missing_ok=True)
    print(f"downloading trained checkpoint -> {path}")
    try:
        urllib.request.urlretrieve(CHECKPOINT_DOWNLOAD_URL, partial_path)
        if partial_path.stat().st_size < MIN_CHECKPOINT_BYTES:
            raise RuntimeError(
                "checkpoint download is unexpectedly small: "
                f"{partial_path.stat().st_size} bytes"
            )
        partial_path.replace(path)
    finally:
        partial_path.unlink(missing_ok=True)
    return path


class Detector:
    """Load the submitted model once and expose UI-friendly predictions."""

    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(ensure_checkpoint(checkpoint), self.device)

    def predict(self, image: Image.Image, threshold: float = 0.5) -> dict:
        image = image.convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            ai_probability = torch.sigmoid(self.model(tensor)).item()

        real_probability = 1.0 - ai_probability
        is_ai = ai_probability >= threshold
        return {
            "decision": "AI-generated" if is_ai else "Likely real",
            "confidence": ai_probability if is_ai else real_probability,
            "ai_probability": ai_probability,
            "real_probability": real_probability,
            "threshold": threshold,
        }
