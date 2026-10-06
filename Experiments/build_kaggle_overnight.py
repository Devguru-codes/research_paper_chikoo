"""Build Kaggle/sapota_overnight.ipynb: runs the experiments that are still missing for the paper
(1) CSV multi-task regression + classification, (2) pre-specified MT-ST+ improvements, with their statistics.
Inputs from the private dataset: sapota_spectra/{spectra.npz, unlabelled_pixels.npz}, extras/{dataset.csv, main.json}."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Kaggle", "sapota_overnight.ipynb")


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)}


def wf(name):
    return code(f"%%writefile {name}\n" + open(os.path.join(HERE, name), encoding="utf-8").read())


import base64, zlib
emb = lambda f: base64.b64encode(zlib.compress(open(os.path.join(HERE, "..", f), "rb").read(), 9)).decode()
EMB = code("EMB_MAIN = '" + emb("Results/main.json") + "'" + chr(10) + "EMB_CSV = '" + emb("Dataset/dataset.csv") + "'")

cells = [
    EMB,
    md("# Sapota overnight run: CSV multi-task experiment + MT-ST+ (outputs: Results/*.json, results_bundle.zip)"),
    code("""
import glob, os, shutil, subprocess, sys
WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
INP = '/kaggle/input' if os.path.isdir('/kaggle/input') else os.getcwd()
os.chdir(WORK)
os.makedirs('data', exist_ok=True); os.makedirs('Results', exist_ok=True)
os.environ.update(DATA_DIR=f'{WORK}/data', RESULTS_DIR=f'{WORK}/Results')
def find(name):
    f = glob.glob(f'{INP}/**/{name}', recursive=True); assert f, name; return f[0]
for n in ['spectra.npz', 'unlabelled_pixels.npz', 'extraction_log.json']:
    shutil.copy(find(n), 'data/')
import base64, zlib
open('Results/main.json', 'wb').write(zlib.decompress(base64.b64decode(EMB_MAIN)))
open('dataset.csv', 'wb').write(zlib.decompress(base64.b64decode(EMB_CSV)))
os.environ['CSV_PATH'] = f'{WORK}/dataset.csv'
import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
"""),
    wf("models.py"), wf("run_experiments.py"), wf("run_csv_multitask.py"),
    code("!python run_csv_multitask.py main 2>&1 | grep -v Warning"),
    code("!python run_csv_multitask.py stats"),
    code("!python run_experiments.py improve"),
    code("!python run_experiments.py improve_stats"),
    code("""
os.makedirs('bundle', exist_ok=True)
shutil.copytree('Results', 'bundle/Results', dirs_exist_ok=True)
shutil.make_archive('results_bundle', 'zip', 'bundle')
print(sorted(os.listdir('Results')))
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", os.path.abspath(OUT))
