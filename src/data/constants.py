"""
Single source of truth for the robustness transform table from the Track 5
problem statement (section 5.2). Both training-time augmentation and the
robustness eval harness import this so they can never drift apart.
"""

LABEL_REAL = 0
LABEL_AIGC = 1

# transform name -> list of severity params, taken verbatim from the problem statement table
ROBUSTNESS_TRANSFORMS = {
    "jpeg": {
        "param_name": "quality",
        "levels": [90, 70, 50, 30],
        "real_world_analog": "Social-media re-encode, messaging",
    },
    "blur": {
        "param_name": "sigma",
        "levels": [0.5, 1.0, 2.0],
        "real_world_analog": "Out-of-focus",
    },
    "resize": {
        "param_name": "scale",
        "levels": [0.5, 0.25],
        "real_world_analog": "Thumbnail generation",
    },
    "noise": {
        "param_name": "sigma",
        "levels": [0.02, 0.05, 0.10],
        "real_world_analog": "Low-light sensor noise",
    },
    "color_jitter": {
        "param_name": "strength",
        "levels": [0.20],
        "real_world_analog": "Filter apps, auto-enhance",
    },
    "center_crop": {
        "param_name": "keep_fraction",
        "levels": [0.80],
        "real_world_analog": "Profile-picture cropping, framing",
    },
}

IMAGE_SIZE = 224  # backbone input resolution; resize-back-to-this after any transform that changes size
