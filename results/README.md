Final result artifacts go here, tracked in git (unlike `outputs/`, which is
scratch space for local runs):

- `robustness_table.csv` — accuracy/AUROC for clean + every transform×severity
  setting, produced by `src/eval/robustness_eval.py`.
- `error_summary.csv` — compact per-condition misclassification counts
  derived from the final robustness sweep. The full 16,034-row image-level
  error file remains in Drive to avoid committing machine-specific paths.
- `error_examples.csv` — two highest-confidence false positives and two
  highest-confidence false negatives for each evaluation condition, with
  machine-specific path prefixes removed.
- `train_log.csv` — per-epoch train loss / val accuracy / val AUROC, produced
  by `src/train.py`.
- `model_comparison.csv` — compact final comparison across EfficientNet-B0,
  DINOv2, and DINOv3.
- `dinov2_clean_results.csv` — DINOv2 frozen-backbone clean-test metrics.
- `dinov3_robustness_table.csv` — DINOv3 frozen-backbone clean and complete
  per-transform metrics.
- `dinov3_train_log.csv` — DINOv3 binary-head training log.

These files capture the final Colab run. See the "Results" section in the
top-level README.

The values were recovered from the saved, executed Colab notebooks and their
CSV outputs in Google Drive. `error_summary.csv` is derived directly from each
condition's reported accuracy and `n=20,000`; its counts sum to the notebook's
reported 16,034 condition-level misclassifications. `error_examples.csv` is
derived from that run's full image-level error file and keeps the most
confident errors for useful, compact inspection.
