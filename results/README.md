Final result artifacts go here, tracked in git (unlike `outputs/`, which is
scratch space for local runs):

- `robustness_table.csv` — accuracy/AUROC for clean + every transform×severity
  setting, produced by `src/eval/robustness_eval.py`.
- `error_analysis.csv` — every misclassified image with its confidence and
  error type, produced by the same script.
- `train_log.csv` — per-epoch train loss / val accuracy / val AUROC, produced
  by `src/train.py`.

Copy these in from the Colab run's `outputs/` directory once training is
final. See the "Results" section in the top-level README.
