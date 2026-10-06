"""Build Kaggle/sapota_mtst_kaggle.ipynb: a self-contained notebook that embeds the exact experiment scripts
(%%writefile cells) so Kaggle runs are identical to local runs. Re-run this after editing any script."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Kaggle", "sapota_mtst_kaggle.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def writefile(name):
    src = open(os.path.join(HERE, name), encoding="utf-8").read()
    return code(f"%%writefile {name}\n{src}")


cells = [
    md("""
# Sapota ripeness from NIR hyperspectral imaging — MT-ST and baselines (reproducible run)

**Inputs (attach as a Kaggle Dataset — either one works):**
* `sapota_spectra.zip` → contains `sapota_spectra/spectra.npz` (per-fruit spectra already extracted, ~1 MB), **or**
* `sapota_raw_hypercubes.zip` → the raw Resonon Pika NIR-320 ENVI cubes (`CHIKU/...`); spectra are then
  extracted inside this notebook (needs Internet ON once for `pip install spectral`).

**Settings:** Accelerator → GPU (T4 or P100). Full run ≈ 1–2 h on a T4.

**Outputs** (in `/kaggle/working`): `Results/*.json` (every number used in the paper), `figures/*.png` (300 dpi),
`results_bundle.zip`.

Optional: upload a CSV with columns `fruit_id,sugar,carbohydrate` containing **laboratory-measured** reference
values and set `LABELS_CSV` below to enable the multi-task regression heads and the sugar/carbohydrate metrics.
"""),
    code("""
import glob, os, shutil, subprocess, sys, json
USE_RAW_CUBES = False      # True: re-extract spectra from the raw hypercubes instead of using spectra.npz
LABELS_CSV = None          # e.g. '/kaggle/input/<your-dataset>/reference_values.csv' (measured values only)
QUICK_TEST = False         # True: 2 epochs x 1 repeat, just to check that everything runs

WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
INP = '/kaggle/input' if os.path.isdir('/kaggle/input') else os.getcwd()
os.chdir(WORK)
os.environ.update(DATA_DIR=f'{WORK}/data', RESULTS_DIR=f'{WORK}/Results', FIG_DIR=f'{WORK}/figures')
if QUICK_TEST:
    os.environ.update(EPOCHS='2', N_REPEATS='2')
os.makedirs(f'{WORK}/data', exist_ok=True)
npz = glob.glob(f'{INP}/**/spectra.npz', recursive=True)
hdr = glob.glob(f'{INP}/**/CHIKU/**/*.hdr', recursive=True)
print('spectra.npz found:', npz[:1], '| raw cube headers found:', len(hdr))
import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
"""),
    md("## 1. Code (identical to `Submission_Package/Experiments/*.py`)"),
    writefile("models.py"),
    writefile("extract_spectra.py"),
    writefile("extract_unlabelled.py"),
    writefile("run_experiments.py"),
    writefile("make_figures.py"),
    md("## 2. Data: extract spectra from raw cubes, or use the pre-extracted file"),
    code("""
if USE_RAW_CUBES or not npz:
    assert hdr, 'Attach sapota_raw_hypercubes.zip or sapota_spectra.zip as a dataset'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'spectral'], check=True)
    root = hdr[0][:hdr[0].index('CHIKU') + len('CHIKU')]
    os.environ['CUBE_ROOT'] = root
    !python extract_spectra.py
else:
    shutil.copy(npz[0], f'{WORK}/data/spectra.npz')
    log = os.path.join(os.path.dirname(npz[0]), 'extraction_log.json')
    if os.path.exists(log): shutil.copy(log, f'{WORK}/data/')
    un = os.path.join(os.path.dirname(npz[0]), 'unlabelled_pixels.npz')
    if os.path.exists(un): shutil.copy(un, f'{WORK}/data/')
if not os.path.exists(f'{WORK}/data/unlabelled_pixels.npz') and hdr:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'spectral'], check=True)
    os.environ['CUBE_ROOT'] = hdr[0][:hdr[0].index('CHIKU') + len('CHIKU')]
    !python extract_unlabelled.py
import numpy as np
d = np.load(f'{WORK}/data/spectra.npz')
print({k: d[k].shape for k in ['wl', 'X_mean', 'X_patch']}, 'stages:', dict(zip(*np.unique(d['stage'], return_counts=True))))
"""),
    md("## 3. Experiments (each cell writes one JSON in `Results/`)"),
    code("""
LABARGS = f'--labels {LABELS_CSV}' if LABELS_CSV else ''
!python run_experiments.py main $LABARGS
"""),
    code("!python run_experiments.py stats      # fruit-level bootstrap 95% CIs + corrected t-test / Wilcoxon (Holm) vs MT-ST"),
    code("!python run_experiments.py bandsel    # attribution stability + top-k band retraining test"),
    code("!python run_experiments.py ablation   # architectural ablations of MT-ST"),
    code("!python run_experiments.py augment    # noise-augmentation sensitivity"),
    code("!python run_experiments.py compute    # params, FLOPs, latency (GPU / CPU), time per epoch, peak VRAM"),
    code("!python run_experiments.py improve    # pre-specified MT-ST+ (balanced loss + self-supervised pretraining + 5-seed ensemble)"),
    code("!python run_experiments.py improve_stats"),
    md("## 4. Tables and figures"),
    code("""
import pandas as pd
R = os.environ['RESULTS_DIR']
M = json.load(open(f'{R}/main.json')); S = json.load(open(f'{R}/stats.json'))
print(json.dumps(M['protocol'], indent=1))
rows = []
for m, s in M['summary'].items():
    ci = S['bootstrap_ci_95'][m]['acc']; t = S['tests_vs_mtst'].get(m, {})
    rows.append(dict(model=m, acc=f"{100*s['acc'][0]:.1f} ± {100*s['acc'][1]:.1f}", acc_CI95=f"[{100*ci[0]:.1f}, {100*ci[1]:.1f}]",
                     macroF1=f"{100*s['f1'][0]:.1f}", kappa=f"{s['kappa'][0]:.2f}",
                     p_corrected_t_holm=t.get('p_t_holm'), p_wilcoxon_holm=t.get('p_wilcoxon_holm')))
display(pd.DataFrame(rows))
for f in ['ablation.json', 'augment.json']:
    A = json.load(open(f'{R}/{f}'))
    display(pd.DataFrame({k: {m: f"{100*v[0]:.1f} ± {100*v[1]:.1f}" for m, v in a['summary'].items()} for k, a in A.items()}).T)
print(json.dumps(json.load(open(f'{R}/bandsel.json'))['band_selection_acc'], indent=1))
print(json.dumps({k: v for k, v in json.load(open(f'{R}/compute.json')).items()}, indent=1))
"""),
    code("""
!python make_figures.py
from IPython.display import Image, display
for f in sorted(glob.glob(os.environ['FIG_DIR'] + '/*.png')):
    print(f); display(Image(f, width=900))
os.makedirs(f'{WORK}/bundle', exist_ok=True)
for d_ in ['Results', 'figures']:
    shutil.copytree(f'{WORK}/{d_}', f'{WORK}/bundle/{d_}', dirs_exist_ok=True)
shutil.make_archive(f'{WORK}/results_bundle', 'zip', f'{WORK}/bundle')
print('Download: results_bundle.zip')
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", os.path.abspath(OUT))
