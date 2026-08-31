#!/usr/bin/env python
"""Gradio demo for the submitted EfficientNet-B0 AIGC detector."""

import argparse
import os
from pathlib import Path

import gradio as gr
from PIL import Image

from src.demo_utils import DEFAULT_CHECKPOINT, Detector


def build_demo(checkpoint: Path, threshold: float = 0.5) -> gr.Blocks:
    detector = Detector(checkpoint)

    def predict(image: Image.Image | None):
        if image is None:
            raise gr.Error("Upload an image before clicking Analyze.")

        result = detector.predict(image, threshold)

        scores = {
            "AI-generated": result["ai_probability"],
            "Likely real": result["real_probability"],
        }
        explanation = (
            f"### Decision: {result['decision']}\n\n"
            f"**Decision confidence:** {result['confidence']:.1%}  \n"
            f"**AI-generated probability:** {result['ai_probability']:.1%}  \n"
            f"**Threshold:** {result['threshold']:.0%}"
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
        gr.Markdown(f"Runtime device: `{detector.device}`")

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
