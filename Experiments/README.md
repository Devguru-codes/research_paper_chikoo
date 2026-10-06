# Experiments for the revised ICMLDE paper (PID 220)

Every number, table and data figure in `Paper/icmlde_manuscript.tex` is produced by the scripts below.
Run from `Submission_Package/`. Tested with Python 3.11, PyTorch 2.5.1 (CUDA), scikit-learn 1.9, XGBoost 3.2, `spectral` 0.24.

| Step | Command | Output |
|---|---|---|
| 1 | `python Experiments/extract_spectra.py` | `Experiments/data/spectra.npz`, `extraction_log.json` (per-fruit ROI spectra from the raw Pika NIR-320 cubes) |
| 2 | `python Experiments/extract_unlabelled.py` | `Experiments/data/unlabelled_pixels.npz` (pixel spectra of the *unlabelled* March sessions, for pretraining only) |
| 3 | `python Experiments/run_experiments.py main` | `Results/main.json`, `Results/xai_profiles.json` — 9 models × 5×5 fruit-grouped CV |
| 4 | `python Experiments/run_experiments.py stats` | `Results/stats.json` — fruit-level bootstrap CIs, corrected t-test / Wilcoxon (Holm) |
| 5 | `python Experiments/run_experiments.py bandsel` | `Results/bandsel.json` — attribution stability, top-k vs random band test |
| 6 | `python Experiments/run_experiments.py ablation` | `Results/ablation.json` |
| 7 | `python Experiments/run_experiments.py augment` | `Results/augment.json` |
| 8 | `python Experiments/run_experiments.py compute` | `Results/compute.json` (run on an otherwise idle machine) |
| 9 | `python Experiments/run_experiments.py improve` then `improve_stats` | `Results/improve.json`, `Results/improve_stats.json` — pre-specified MT-ST+ |
| 9b | `cd Experiments && python run_csv_multitask.py main` then `stats` | `Results/csv_main.json`, `Results/csv_stats.json` — multi-task sugar/carbohydrate regression + stage on `Dataset/dataset.csv` (proxy, model-generated targets; grouped by source record), λ_cls sweep |
| 10 | `python Experiments/make_figures.py` | `Images/revised/*.png` (300 dpi) |
| 11 | `python Experiments/make_tex_numbers.py` | `Paper/results_macros.tex`, `Paper/tab_*.tex` |
| 12 | in `Paper/`: `pdflatex` → `bibtex` → `pdflatex` ×2 on `icmlde_manuscript.tex` | `icmlde_manuscript.pdf` |

`Results/logs/*.log` hold the console output of the runs used in the paper.

**Regression (sugar / carbohydrate):** add `--labels reference_values.csv` (columns `fruit_id,sugar,carbohydrate`,
laboratory-measured values of the imaged fruits; fruit ids as in `extraction_log.json`) to `main`. Never use the values
in `Dataset/dataset.csv`: they are not linked to the imaged fruits.

**Kaggle:** `python Experiments/build_kaggle_notebook.py` rebuilds `Kaggle/sapota_mtst_kaggle.ipynb` from these scripts.
