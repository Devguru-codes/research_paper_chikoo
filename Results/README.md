# Results of the revised analysis (ICMLDE 2026, Paper ID 220)

Every number, table and data figure in the revised manuscript is generated from these files by
`Experiments/make_tex_numbers.py` and `Experiments/make_figures.py`.

| File | Produced by | Where it was run | Used in the paper |
|---|---|---|---|
| `main.json`, `xai_profiles.json` | `run_experiments.py main` | local (NVIDIA RTX 3050 Laptop) | ripeness classification table, fold box plot, confusion matrices |
| `stats.json` | `run_experiments.py stats` | local | bootstrap CIs, corrected t-test / Wilcoxon (Holm) |
| `bandsel.json` | `run_experiments.py bandsel` | local | attribution stability and band-selection test |
| `ablation.json` | `run_experiments.py ablation` | local | ablation table |
| `augment.json` | `run_experiments.py augment` | local | augmentation-sensitivity table |
| `improve.json`, `improve_stats.json` | `run_experiments.py improve` / `improve_stats` | Kaggle T4 (`Kaggle/sapota_improve.ipynb`), using the local folds in `splits_image.json` | MT-ST+ table |
| `csv_main.json`, `csv_stats.json` | `run_csv_multitask.py main` / `stats` | Kaggle T4 (`Kaggle/sapota_overnight.ipynb`) | regression table, loss-weighting table, regression scatter |
| `compute.json`, `kaggle_compute/*` | `run_experiments.py compute`, 3 runs, median | Kaggle T4 + Intel Xeon (`Kaggle/sapota_compute.ipynb`) | computational-cost table |
| `compute_laptop_rtx3050.json` | `run_experiments.py compute` | local | not used in the paper (kept for reference) |
| `splits_image.json` | outer folds of the 5x5 fruit-grouped CV | local scikit-learn 1.9 | guarantees identical folds on other machines (`SPLITS_JSON`) |
| `logs/*.log`, `logs/kaggle/*.log` | console output of the runs above | — | — |

Protocol: stratified 5-fold cross-validation grouped by fruit, repeated 5 times (25 outer folds); identical
preprocessing (SNV + Savitzky-Golay 1st derivative + training-fold z-scoring) for every model.

Note on the tabular data set (`Dataset/dataset.csv`, used only by `csv_*`): its spectra and sugar/carbohydrate values
are model-generated (LLM) proxy values, not laboratory measurements of the imaged fruits; see the limitations section
of the revised paper.
