# Devpost description — Track 5

Submission-ready copy. Add the team names/contributions and video URL in the
Devpost form before publishing.

## Inspiration / problem

Generative AI makes it trivial to produce realistic synthetic images at
scale, and detection has to survive what actually happens to an image after
upload — re-compression, cropping, filters, resizing — not just clean lab
conditions. We built a detector that's trained to be robust to exactly the
transformations TikTok's problem statement called out: JPEG re-compression,
blur, resize round-trips, sensor noise, color jitter, and center cropping.

## What it does

Given a directory of images, the model outputs a confidence score (0–1) per
image indicating the likelihood it's AI-generated, as a JSON file of
`{image_path, pred}` entries — usable as a drop-in moderation/trust-and-safety
signal. For an accessible demonstration, our Gradio interface lets a user
upload an image and immediately see the real-versus-AIGC confidence scores.

## How we built it

- **Model**: an EfficientNet-B0 backbone (pretrained, via `timm`) fine-tuned
  with a binary classification head, using a freeze-then-unfreeze schedule
  (one head-only warmup epoch, then full-backbone fine-tuning).
- **Robustness-by-training, not just robustness-by-evaluation**: rather than
  training clean and hoping it generalizes, every training batch has a
  random chance of being run through one of the six problem-statement
  transforms at one of its specified severities before being fed to the
  model. The same transform implementations are shared between training and
  evaluation (`src/data/constants.py`, `src/data/augmentations.py`), so
  there's no drift between what the model was trained to withstand and what
  it's scored against.
- **Evaluation harness**: sweeps the held-out test set through clean plus
  every (transform, severity) combination and reports accuracy/AUROC for
  each, plus a full false-positive/false-negative breakdown with confidence
  scores.
- **Model selection by evidence**: we also evaluated frozen-feature DINOv2
  and DINOv3 baselines. EfficientNet-B0 remained the final model because it
  produced the best clean and transformed-image results while being faster
  and lighter for interactive inference.
- **Interactive demo**: a Gradio app automatically downloads the public
  checkpoint, accepts an uploaded image, and returns the model decision and
  both class confidence scores. It runs on either CPU or GPU.

## Results

Trained on 85,000 images from CIFAKE on GPU (Colab).

- Clean-eval accuracy: **97.58%**
- Clean-eval AUROC: **99.75%**
- Average accuracy across all transformed test sets: **94.45%**
- Worst-case accuracy: **87.55%** under a 0.25× resize round-trip
- Severe-blur accuracy: **89.65%** at Gaussian blur σ=2.0
- Full per-transform robustness table: see [`results/robustness_table.csv`](https://github.com/Brentwcy/2026_Tiktok_Techjam/blob/main/results/robustness_table.csv)
- Ran inference end-to-end on 20,000 images to validate the required output
  schema at scale.

Backbone comparison:

| Backbone | Clean accuracy | Clean AUROC | Average transformed accuracy | Worst case |
| --- | ---: | ---: | ---: | ---: |
| **EfficientNet-B0 (final)** | **97.58%** | **99.75%** | **94.45%** | **87.55%** |
| DINOv2 ViT-S/14 (frozen backbone) | 95.47% | 99.24% | Not fully evaluated | Not fully evaluated |
| DINOv3 ViT-S/16 (frozen backbone) | 91.80% | 97.63% | 87.93% | 77.47% |

The DINO results are frozen-backbone engineering baselines, not fully
fine-tuned foundation-model results.

## Development tools

- VS Code (local development)
- Google Colab (GPU training/inference)
- Gradio (interactive image-upload demo)
- Git / GitHub

## Models / APIs used

- EfficientNet-B0 pretrained weights (`efficientnet_b0`), loaded via `timm`.
  No external inference APIs — everything runs locally against the
  checkpoint.
- DINOv2 ViT-S/14 and DINOv3 ViT-S/16 were tested as frozen-backbone
  alternatives for model-selection evidence; neither replaced EfficientNet.

## Libraries and frameworks

PyTorch, torchvision, timm, scikit-learn, pandas, NumPy, Pillow, tqdm, and
Gradio.

## Datasets and assets used

- **CIFAKE** (Kaggle, birdy654) — 85,000 training images (42,500 real and
  42,500 AI-generated), plus separate validation and test splits.
- WildFake demo subset (COCO val2017 reals + DALL·E Advanced fakes) —
  reserved by the problem statement as a held-out benchmark; **not** used in
  training and not evaluated in the submitted results.
- SID_Set was considered but not incorporated into training given the time
  budget — noted as a limitation/future-work item.

## Challenges we ran into

- PyTorch 2.6 changed `torch.load`'s default to `weights_only=True`, which
  broke loading our own checkpoints because the saved dict carries plain
  Python values (backbone name, epoch, AUROC) alongside tensors — fixed by
  explicitly passing `weights_only=False` for checkpoints this project
  produces itself, and by casting metrics to plain `float` before saving so
  the pickle doesn't carry numpy scalars in the first place.
- Larger vision-foundation-model features were not automatically better for
  this forensic task. The DINO baselines underperformed EfficientNet,
  reinforcing our decision to prioritize measured robustness over model size.
- We packaged a public checkpoint and verified it from a fresh Colab clone,
  including dependency installation, checkpoint loading, inference, and the
  required JSON schema.

## What's next

- Evaluate against the WildFake demo subset to measure out-of-distribution
  generalization beyond CIFAKE.
- Add SID_Set for higher-resolution, more diverse generator coverage.
- Error-pattern analysis across `results/error_analysis.csv` to identify
  which transform/severity combinations drive the most false
  positives/negatives.

## Team

Add each team member's name and actual ownership before submission: data,
training, robustness evaluation, demo/integration, and presentation.

## Links

- Repository: https://github.com/Brentwcy/2026_Tiktok_Techjam
- Public checkpoint: https://drive.google.com/file/d/1l4Kw8aW6vv8uzTmvM3Lzhqfz_zchWUpN/view
- Demo video: add the final public video URL in Devpost
