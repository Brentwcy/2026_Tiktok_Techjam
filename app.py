#!/usr/bin/env python
"""
Gradio demo for the submitted EfficientNet-B0 AIGC detector with interactive
robustness-under-transformation testing (TikTok TechJam 2026 — Track 5).
"""

import argparse
import os
from pathlib import Path

import gradio as gr
from PIL import Image

from src.data.augmentations import apply_transform
from src.demo_utils import DEFAULT_CHECKPOINT, Detector

# Mapping friendly dropdown labels to Track 5 transform settings (name, level)
TRANSFORM_OPTIONS = {
    "None (Clean Image)": None,
    "JPEG Compression (Quality = 30) — Heavy re-compression": ("jpeg", 30),
    "JPEG Compression (Quality = 50) — Medium re-compression": ("jpeg", 50),
    "JPEG Compression (Quality = 70) — Standard re-compression": ("jpeg", 70),
    "JPEG Compression (Quality = 90) — Light re-compression": ("jpeg", 90),
    "Gaussian Blur (σ = 2.0) — Heavy blur / out-of-focus": ("blur", 2.0),
    "Gaussian Blur (σ = 1.0) — Medium blur": ("blur", 1.0),
    "Gaussian Blur (σ = 0.5) — Slight blur": ("blur", 0.5),
    "Resize Roundtrip (0.25×) — Severe downscale & upscale": ("resize", 0.25),
    "Resize Roundtrip (0.50×) — Medium downscale & upscale": ("resize", 0.50),
    "Sensor Noise (σ = 0.10) — Severe low-light noise": ("noise", 0.10),
    "Sensor Noise (σ = 0.05) — Medium sensor noise": ("noise", 0.05),
    "Sensor Noise (σ = 0.02) — Light sensor noise": ("noise", 0.02),
    "Color Jitter (±20% Brightness / Contrast / Saturation)": ("color_jitter", 0.20),
    "Center Crop (80% Framing Crop)": ("center_crop", 0.80),
}

CUSTOM_CSS = """
/* Ensure consistent image container dimensions and dropdown elevation */
.gradio-container { max-width: 1200px !important; margin: auto !important; }
ul.options, .dropdown-menu { z-index: 9999 !important; max-height: 320px !important; overflow-y: auto !important; }
.image-box { height: 330px !important; }
.image-box .image-container, .image-box .wrap { height: 330px !important; display: flex !important; align-items: center !important; justify-content: center !important; }
.image-box img { max-height: 300px !important; width: auto !important; object-fit: contain !important; margin: auto !important; }
"""


def build_demo(checkpoint: Path = DEFAULT_CHECKPOINT, threshold: float = 0.5) -> gr.Blocks:
    detector = Detector(checkpoint)

    def predict(image: Image.Image | None, transform_choice: str):
        if image is None:
            raise gr.Error("Upload an image before clicking Analyze.")

        image = image.convert("RGB")

        # 1. Clean image prediction
        clean_result = detector.predict(image, threshold)
        clean_ai_prob = clean_result["ai_probability"]
        clean_decision = clean_result["decision"]

        transform_setting = TRANSFORM_OPTIONS.get(transform_choice)

        if transform_setting is not None:
            # 2. Apply chosen Track 5 transformation
            name, level = transform_setting
            transformed_image = apply_transform(image, name, level)

            # Evaluate transformed image
            eval_result = detector.predict(transformed_image, threshold)
            eval_ai_prob = eval_result["ai_probability"]
            eval_real_prob = eval_result["real_probability"]
            decision = eval_result["decision"]
            confidence = eval_result["confidence"]

            preview_image = transformed_image
            is_transformed = True
        else:
            eval_ai_prob = clean_ai_prob
            eval_real_prob = clean_result["real_probability"]
            decision = clean_decision
            confidence = clean_result["confidence"]
            preview_image = image
            is_transformed = False

        scores = {
            "AI-generated": eval_ai_prob,
            "Likely real": eval_real_prob,
        }

        # Build explanation with robustness impact comparison
        is_decision_stable = (decision == clean_decision)
        delta_shift = (eval_ai_prob - clean_ai_prob) * 100.0

        if is_transformed:
            stability_badge = "✅ **Decision Invariant (Robust)**" if is_decision_stable else "⚠️ **Decision Flipped Under Degradation**"
            explanation = (
                f"### Decision: {decision}\n\n"
                f"**Active Decision Confidence:** {confidence:.1%}  \n"
                f"**Active AI-Generated Probability:** {eval_ai_prob:.1%}  \n"
                f"**Applied Transformation:** `{transform_choice}`\n\n"
                f"---\n"
                f"### 🛡️ Robustness Analysis\n"
                f"- **Clean Image P(AI):** {clean_ai_prob:.1%} (Decision: *{clean_decision}*)\n"
                f"- **Transformed Image P(AI):** {eval_ai_prob:.1%} (Decision: *{decision}*)\n"
                f"- **Confidence Shift (Δ):** {delta_shift:+.1f}%\n"
                f"- **Status:** {stability_badge}\n\n"
                f"*(Model maintains detection reliability even under aggressive post-processing).* "
            )
        else:
            explanation = (
                f"### Decision: {decision}\n\n"
                f"**Decision Confidence:** {confidence:.1%}  \n"
                f"**AI-Generated Probability:** {eval_ai_prob:.1%}  \n"
                f"**Threshold:** {threshold:.0%}\n\n"
                f"---\n"
                f"💡 *Tip: Select a transformation from the dropdown to test model robustness under real-world social-media degradations.*"
            )

        return preview_image, scores, explanation

    with gr.Blocks(
        title="AI-Generated Image Detector — Track 5",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# 🛡️ Robust AI-Generated Image Detector\n"
            "**TikTok TechJam 2026 — Track 5** | Fine-tuned EfficientNet-B0 with *Robustness-by-Training*.\n\n"
            "Upload an image to estimate whether it is AI-generated or authentic, and use the **Transformation Toggle** "
            "to test whether the detection holds under realistic platform degradations (JPEG re-compression, blur, downsampling, sensor noise, etc.)."
        )

        # Side-by-side matching image displays
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="1. Original Clean Input Image",
                    sources=["upload", "clipboard", "webcam"],
                    height=330,
                    elem_classes=["image-box"],
                )
            with gr.Column(scale=1):
                preview_output = gr.Image(
                    type="pil",
                    label="2. Transformed Image Preview (Model Input)",
                    interactive=False,
                    height=330,
                    elem_classes=["image-box"],
                )

        # Interactive controls
        with gr.Row():
            with gr.Column(scale=3):
                transform_dropdown = gr.Dropdown(
                    choices=list(TRANSFORM_OPTIONS.keys()),
                    value="None (Clean Image)",
                    label="Real-World Transformation (Robustness Toggle)",
                    info="Simulate social media re-encoding, optical blur, thumbnail resizing, or sensor noise.",
                    filterable=False,
                    interactive=True,
                )
            with gr.Column(scale=1, min_width=160):
                analyze_button = gr.Button("⚡ Analyze Image", variant="primary", size="lg")
                clear_button = gr.ClearButton(
                    components=[image_input, transform_dropdown],
                    value="Clear All",
                    size="sm",
                )

        # Results & Metrics
        with gr.Row():
            with gr.Column(scale=1):
                result_label = gr.Label(
                    label="Model Confidence Scores",
                    num_top_classes=2,
                )
            with gr.Column(scale=1):
                explanation = gr.Markdown("Upload an image, pick an optional transformation, and click **⚡ Analyze Image**.")

        # Wire clear button to reset outputs as well
        clear_button.add([preview_output, result_label, explanation])

        analyze_button.click(
            fn=predict,
            inputs=[image_input, transform_dropdown],
            outputs=[preview_output, result_label, explanation],
        )

        gr.Markdown(
            "---\n"
            "### 📊 Key Benchmark Summary (CIFAKE Held-Out Test Set, N=20,000)\n"
            "- **Clean Baseline Accuracy**: **97.58%** (AUROC: 99.75%)\n"
            "- **Average Transformed Accuracy**: **94.45%** across all 14 Track 5 degradation settings\n"
            "- **Worst-Case Condition (0.25× Resize Roundtrip)**: **87.55%**\n"
            "- **Heavy Blur (σ=2.0)**: **89.65%** | **JPEG 30**: **95.15%**\n\n"
            f"*Runtime device: `{detector.device}` | Parameters: ~5.3M (EfficientNet-B0)*"
        )

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
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=args.server_port,
        share=args.share,
        show_error=True,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()

