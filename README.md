# Robust Detection of AI-Generated Images Under Real-World Transformations

TikTok TechJam 2026 — Track 5.

Detects whether an image is AI-generated (AIGC) or authentic, and stays
accurate after realistic post-processing: JPEG re-compression, blur, resize
round-trips, sensor noise, color jitter, and center cropping.

## Project overview

- **Model**: a pretrained EfficientNet-B0 (`timm`) fine-tuned with a binary
  classification head (real vs. AIGC). The classifier head was warmed up for
  one epoch before the full backbone was unfrozen.
- **Robustness strategy**: the six transforms from the problem statement's
  robustness table (see `src/data/constants.py`) are applied *during
  training* at random, not just at eval time — the model has to have seen
  these degradations to be robust to them.
- **Evaluation**: a dedicated harness (`src/eval/robustness_eval.py`) scores
  the held-out test set both clean and under every (transform, severity)
  combination from the table, producing the robustness summary and an error
  analysis of false positives/negatives.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA GPU for reasonable fine-tuning time; runs on CPU but slowly.

## Supported datasets

The pipeline supports the datasets below (download separately; they are not
committed to this repo). The reported checkpoint and results used **CIFAKE
only**.

- SID_Set — https://huggingface.co/datasets/saberzl/SID_Set
- CIFAKE — https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- WildFake (optional extra training diversity, excluding the demo subset below) — https://modelscope.cn/datasets/hy2628982280/WildFake/summary

**Do not train on the WildFake demo/validation subset** (COCO val2017 reals +
DALL·E Advanced fakes) — the problem statement reserves that split purely as
a held-out benchmark for demoing progress, not for training.

### Building manifests

Each dataset gets scanned into a shared `image_path,label` CSV:

```bash
# repeat --add_to the same file per source dataset to accumulate
python scripts/build_manifest.py --root data/raw/cifake/train \
    --real_subdir REAL --fake_subdir FAKE \
    --add_to data/processed/all.csv

python scripts/build_manifest.py --root data/raw/sid_set \
    --real_subdir real --fake_subdir fake \
    --add_to data/processed/all.csv

# then split into train/val
python scripts/build_manifest.py --split data/processed/all.csv \
    --val_fraction 0.15 \
    --train_out data/processed/train.csv --val_out data/processed/val.csv
```

Build the held-out demo manifest the same way, into a **separate** file
(e.g. `data/processed/demo.csv`) — never merge it into `train.csv`/`val.csv`.

## Training

```bash
python -m src.train \
    --train_manifest data/processed/train.csv \
    --val_manifest data/processed/val.csv \
    --backbone efficientnet_b0 \
    --epochs 4 --batch_size 64 --lr 3e-5 \
    --augment_p 0.5 --freeze_backbone_epochs 1 \
    --checkpoint_out checkpoints/best.pt
```

Backbone is frozen for the first epoch (head-only warmup), then unfrozen.
Best checkpoint is selected by validation AUROC.

## Robustness evaluation (deliverables #4 and #5)

```bash
python -m src.eval.robustness_eval \
    --test_manifest data/processed/test.csv \
    --checkpoint checkpoints/best.pt \
    --out_dir outputs
```

Produces:
- `outputs/robustness_table.csv` — accuracy/AUROC for clean and every
  transform × severity setting.
- `outputs/error_analysis.csv` — every misclassified image with its
  confidence score and error type (false positive / false negative).

`outputs/` itself is gitignored (scratch space). Once a run is final, copy
its CSVs into the tracked `results/` folder so they ship with the repo —
see [Results](#results).

## Inference (deliverable #2 — required script)

Takes a directory of images, outputs a JSON file of `{image_path, pred}`
(confidence in `[0, 1]` that the image is AIGC):

```bash
python scripts/infer.py --input_dir path/to/images \
    --checkpoint checkpoints/best.pt \
    --output preds.json
```

## Reproducing results

1. Download the datasets above into `data/raw/`.
2. Build manifests as shown.
3. Run training.
4. Run the robustness eval harness.
5. Run inference on a held-out image directory to sanity-check the output
   schema.

## Results

Trained on GPU (Colab) on 85,000 CIFAKE images.

| Metric (clean eval) | Score |
| --- | --- |
| Accuracy | 97.58% |
| AUROC | 99.75% |
| Average transformed accuracy | 94.45% |
| Worst-case accuracy (0.25× resize) | 87.55% |
| Severe-blur accuracy (σ=2.0) | 89.65% |

Full per-transform results are available in
[`results/robustness_table.csv`](results/robustness_table.csv). The tracked
[`results/train_log.csv`](results/train_log.csv) records the four training
epochs, while [`results/error_summary.csv`](results/error_summary.csv)
provides compact per-condition misclassification counts.

Inference was run on 20,000 images to produce the required `{image_path, pred}`
JSON output.

**Trained checkpoint**: `checkpoints/best.pt` is not committed to this repo
(binary, gitignored). Download it from: `<TODO: GitHub Release asset or Drive
link>`.

## Limitations / future work

- **Trained only on CIFAKE**, not SID_Set — CIFAKE alone got the pipeline to a
  working, well-scoring model inside the time budget; SID_Set integration
  (higher-resolution, more diverse generators) was scoped out as a stretch
  goal, not attempted.
- **CIFAKE images are 32×32** (CIFAR-based), upscaled to 224×224 for the
  backbone. This is a real resolution mismatch versus full-size real-world
  images and is the most likely source of any generalization gap.
- **Generalization to the WildFake/COCO–DALL·E demo subset is untested** as
  of this writing — that subset was correctly excluded from training per the
  problem statement, but a held-out evaluation pass against it hasn't been
  run yet. Numbers above are CIFAKE-internal only.
- With more time: add SID_Set for resolution/generator diversity, run the
  WildFake demo-subset evaluation, and inspect the false-positive/negative
  gallery in `results/error_analysis.csv` for systematic failure patterns.

## Team contributions

_TODO: list each team member and what they owned._
