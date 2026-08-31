#!/usr/bin/env python
"""Gradio demo for the submitted EfficientNet-B0 AIGC detector."""

import argparse
import os
import urllib.request
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from scripts.infer import TRANSFORM, load_model


DEFAULT_CHECKPOINT = Path("checkpoints/best.pt")
CHECKPOINT_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1l4Kw8aW6vv8uzTmvM3Lzhqfz_zchWUpN&export=download&confirm=t"
)
MIN_CHECKPOINT_BYTES = 10_000_000


def ensure_checkpoint(path: Path) -> Path:
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
                f"checkpoint download is unexpectedly small: "
                f"{partial_path.stat().st_size} bytes"
            )
        partial_path.replace(path)
    finally:
        partial_path.unlink(missing_ok=True)
    return path


def build_demo(checkpoint: Path, threshold: float = 0.5) -> gr.Blocks:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ensure_checkpoint(checkpoint), device)

    def predict(image: Image.Image | None):
        if image is None:
            raise gr.Error("Upload an image before clicking Analyze.")

        image = image.convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            ai_probability = torch.sigmoid(model(tensor)).item()

        real_probability = 1.0 - ai_probability
        decision = "AI-generated" if ai_probability >= threshold else "Likely real"
        confidence = ai_probability if decision == "AI-generated" else real_probability

        scores = {
            "AI-generated": ai_probability,
            "Likely real": real_probability,
        }
        explanation = (
            f"### Decision: {decision}\n\n"
            f"**Decision confidence:** {confidence:.1%}  \n"
            f"**AI-generated probability:** {ai_probability:.1%}  \n"
            f"**Threshold:** {threshold:.0%}"
        )
        return scores, explanation

    with gr.Blocks(
        title="AI-Generated Image Detector",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# Robust AI-Generated Image Detector\n"
            "Upload an image to estimate whether it is AI-generated or real. "
            "The submitted EfficientNet-B0 model runs locally on the active runtime."
        )
        with gr.Row():
            image_input = gr.Image(
                type="pil",
                label="Upload an image",
                sources=["upload", "clipboard", "webcam"],
            )
            with gr.Column():
                result_label = gr.Label(
                    label="Model scores",
                    num_top_classes=2,
                )
                explanation = gr.Markdown("Upload an image and click **Analyze**.")

        with gr.Row():
            analyze_button = gr.Button("Analyze image", variant="primary")
            gr.ClearButton(
                components=[image_input, result_label, explanation],
                value="Clear",
            )

        analyze_button.click(
            fn=predict,
            inputs=image_input,
            outputs=[result_label, explanation],
        )

        gr.Markdown(
            "**Important limitation:** this prototype was trained only on "
            "32×32 CIFAKE images (CIFAR-10 real images versus Stable Diffusion "
            "1.4 images). Treat the score as an experimental signal, not proof "
            "of authenticity."
        )
        gr.Markdown(f"Runtime device: `{device}`")

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    demo = build_demo(args.checkpoint, args.threshold)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
