# Demo video script (about 90 seconds)

Use your own or clearly licensed sample images. Keep the Gradio Colab cell
running throughout the recording.

## 0:00–0:12 — Problem

“AI-image detection often performs well on clean benchmark images but fails
after images are compressed, cropped, filtered, resized, or reposted. Our
Track 5 solution is trained and evaluated specifically for those conditions.”

## 0:12–0:27 — Approach

Show the repository and architecture summary.

“We fine-tuned EfficientNet-B0 for binary real-versus-AIGC classification.
Training randomly applies the exact six transformation families from the
challenge, while a separate harness evaluates every required severity.”

## 0:27–0:52 — Live Gradio demo

Upload one image, click **Analyze image**, and show the decision plus both
confidence scores. If possible, repeat with a compressed or resized version.

“The same checkpoint used for our reported evaluation powers this interface.
It runs locally without an external inference API and can also run on CPU.”

## 0:52–1:12 — Results

Show `results/robustness_table.csv` or the README table.

“Clean accuracy is 97.58% with 99.75% AUROC. Average transformed accuracy is
94.45%. The hardest condition is a 0.25-times resize round-trip at 87.55%,
which makes the remaining generalization gap explicit.”

## 1:12–1:25 — Model-selection insight

“We also tested frozen-feature DINOv2 and DINOv3 baselines. EfficientNet
outperformed both, showing that a larger semantic backbone is not necessarily
better for local synthetic-image artifacts.”

## 1:25–1:30 — Close

“The repository, inference script, public checkpoint, reproducibility steps,
and complete robustness results are available with the submission.”
