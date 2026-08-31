Final result artifacts go here, tracked in git (unlike `outputs/`, which is
scratch space for local runs):

- `robustness_table.csv` — accuracy/AUROC for clean + every transform×severity
  setting, produced by `src/eval/robustness_eval.py`.
- `error_summary.csv` — compact per-condition misclassification counts
  derived from the final robustness sweep. The full 16,034-row image-level
  error file remains in Drive to avoid committing machine-specific paths.
- `train_log.csv` — per-epoch train loss / val accuracy / val AUROC, produced
  by `src/train.py`.

These files capture the final Colab run. See the "Results" section in the
top-level README.

The values were recovered from the saved, executed Colab notebook in Google
Drive. `error_summary.csv` is derived directly from each condition's reported
accuracy and `n=20,000`; its counts sum to the notebook's reported 16,034
condition-level misclassifications.
