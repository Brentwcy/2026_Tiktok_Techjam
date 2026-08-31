# Robust Detection of AI-Generated Images Under Real-World Transformations

TikTok TechJam 2026 — Track 5.

Detects whether an image is AI-generated (AIGC) or authentic, and stays
accurate after realistic post-processing: JPEG re-compression, blur, resize
round-trips, sensor noise, color jitter, and center cropping.

## Project overview

- **Model**: a pretrained CLIP ViT-B/16 (`timm`) fine-tuned with a binary
  classification head (real vs. AIGC).
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

## Datasets

Training data (download separately, not committed to this repo — see
`.gitignore`):

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
    --epochs 5 --batch_size 32 --lr 3e-5 \
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

## Limitations / future work

_TODO before submission: fill in after the training run — e.g. dataset
resolution mismatch (CIFAKE is 32×32), which transforms hurt accuracy most,
generalization gap on the WildFake demo subset vs. training distribution,
what a second day of work would add._

## Team contributions

_TODO: list each team member and what they owned._
