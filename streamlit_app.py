#!/usr/bin/env python
"""Streamlit interface for the submitted EfficientNet-B0 AIGC detector."""

from pathlib import Path

import streamlit as st
from PIL import Image

from src.demo_utils import DEFAULT_CHECKPOINT, Detector


st.set_page_config(
    page_title="Robust AI Image Detector",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the EfficientNet-B0 detector…")
def load_detector(checkpoint: str) -> Detector:
    return Detector(Path(checkpoint))


st.title("Robust AI-Generated Image Detector")
st.caption(
    "Upload an image to estimate whether it is AI-generated or real. "
    "Inference runs locally with the submitted EfficientNet-B0 checkpoint."
)

with st.sidebar:
    st.header("Model settings")
    threshold = st.slider(
        "AI decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Images at or above this AI probability are labelled AI-generated.",
    )
    st.markdown(
        "**Final model:** EfficientNet-B0  \n"
        "**Clean accuracy:** 97.58%  \n"
        "**Average transformed accuracy:** 94.45%"
    )

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    image_column, result_column = st.columns([1.1, 1])

    with image_column:
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with result_column:
        if st.button("Analyze image", type="primary", use_container_width=True):
            try:
                detector = load_detector(str(DEFAULT_CHECKPOINT))
                result = detector.predict(image, threshold)
            except Exception as exc:
                st.error(f"The model could not be loaded or run: {exc}")
            else:
                if result["decision"] == "AI-generated":
                    st.error(f"Decision: {result['decision']}")
                else:
                    st.success(f"Decision: {result['decision']}")

                st.metric("Decision confidence", f"{result['confidence']:.1%}")
                st.write(f"AI-generated probability: **{result['ai_probability']:.1%}**")
                st.progress(result["ai_probability"], text="AI-generated")
                st.write(f"Likely-real probability: **{result['real_probability']:.1%}**")
                st.progress(result["real_probability"], text="Likely real")
                st.caption(
                    f"Threshold: {result['threshold']:.0%} · "
                    f"Runtime device: {detector.device}"
                )

st.divider()
st.warning(
    "Important limitation: this prototype was trained only on 32×32 CIFAKE "
    "images (CIFAR-10 real images versus Stable Diffusion 1.4 images). Treat "
    "the score as an experimental signal, not proof of authenticity."
)
