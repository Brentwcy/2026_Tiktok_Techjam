# Devpost description draft — Track 5

_Draft for copy-paste into Devpost. Fill in the `<TODO>` spots before
submitting (video link, exact per-transform robustness numbers once
`results/robustness_table.csv` is committed, team names)._

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
signal.

## How we built it

- **Model**: a CLIP ViT-B/16 backbone (pretrained, via `timm`) fine-tuned with
  a binary classification head, using a freeze-then-unfreeze schedule
  (head-only warmup, then full fine-tuning).
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

## Results

Trained on 85,000 images from CIFAKE on GPU (Colab).

- Clean-eval accuracy: **97.58%**
- Clean-eval AUROC: **99.75%**
- Full per-transform robustness table: `<TODO: paste headline numbers from
  results/robustness_table.csv once committed>`
- Ran inference end-to-end on 20,000 images to validate the required output
  schema at scale.

## Development tools

- VS Code (local development)
- Google Colab (GPU training/inference)
- Git / GitHub

## Models / APIs used

- CLIP ViT-B/16 pretrained weights (`vit_base_patch16_clip_224.openai`),
  loaded via `timm`. No external inference APIs — everything runs locally
  against the checkpoint.

## Libraries and frameworks

PyTorch, torchvision, timm, scikit-learn, pandas, NumPy, Pillow, tqdm.

## Datasets and assets used

- **CIFAKE** (Kaggle, birdy654) — training data, 85,000 images.
- WildFake demo subset (COCO val2017 reals + DALL·E Advanced fakes) —
  reserved by the problem statement as a held-out benchmark; **not** used in
  training. `<TODO: confirm whether an evaluation pass against this subset
  was completed before submission, and add the result here if so>`.
- SID_Set was considered but not incorporated into training given the time
  budget — noted as a limitation/future-work item.

## Challenges we ran into

- PyTorch 2.6 changed `torch.load`'s default to `weights_only=True`, which
  broke loading our own checkpoints because the saved dict carries plain
  Python values (backbone name, epoch, AUROC) alongside tensors — fixed by
  explicitly passing `weights_only=False` for checkpoints this project
  produces itself, and by casting metrics to plain `float` before saving so
  the pickle doesn't carry numpy scalars in the first place.

## What's next

- Evaluate against the WildFake demo subset to measure out-of-distribution
  generalization beyond CIFAKE.
- Add SID_Set for higher-resolution, more diverse generator coverage.
- Error-pattern analysis across `results/error_analysis.csv` to identify
  which transform/severity combinations drive the most false
  positives/negatives.

## Team

`<TODO: names + who owned what — data pipeline, training, eval, packaging>`
