# Running the sapota experiments on Kaggle

## Files in this folder
| File | Size | What it is |
|---|---|---|
| `sapota_mtst_kaggle.ipynb` | ~70 KB | Self-contained notebook: embeds `models.py`, `extract_spectra.py`, `run_experiments.py`, `make_figures.py` |
| `sapota_spectra.zip` | ~1 MB | Per-fruit spectra already extracted from the raw cubes (`spectra.npz` + `extraction_log.json`) — **enough to run everything** |
| `sapota_raw_hypercubes.zip` | ~630 MB | The raw Resonon Pika NIR-320 ENVI cubes (`CHIKU/...`) — only needed to re-do the spectrum extraction |

## Steps
1. kaggle.com → **Datasets → New Dataset** → upload `sapota_spectra.zip` (and optionally `sapota_raw_hypercubes.zip`). Kaggle unzips it automatically.
2. **Code → New Notebook → File → Import Notebook** → choose `sapota_mtst_kaggle.ipynb`.
3. Right panel: **Add Input** → your dataset. **Settings → Accelerator → GPU T4 x1**. (Internet ON only if you set `USE_RAW_CUBES = True`, because it runs `pip install spectral`.)
4. Optional quick check: set `QUICK_TEST = True` in the first code cell (≈ 5 min). Then set it back to `False`.
5. **Run All**. Full run ≈ 1.5–3 h on a T4. Outputs appear in `/kaggle/working`: `Results/*.json`, `figures/*.png`, `results_bundle.zip` (download it from the Output tab).

## Notes
* Results on Kaggle will differ slightly from the local RTX 3050 run (different GPU kernels), but should agree within the reported SDs. Latency numbers in `compute.json` are hardware-specific — report the hardware they were measured on.
* `LABELS_CSV`: only set this to a CSV of **laboratory-measured** sugar/carbohydrate values for the imaged fruits (`fruit_id,sugar,carbohydrate`; fruit ids are listed in `extraction_log.json`, e.g. `08-04-2022/Ripe/B#2`).
* If you edit any script in `Submission_Package/Experiments/`, rebuild the notebook with `python Experiments/build_kaggle_notebook.py`.
