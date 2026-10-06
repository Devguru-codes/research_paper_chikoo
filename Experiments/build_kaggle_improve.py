"""Build Kaggle/sapota_improve.ipynb: the pre-specified MT-ST+ experiment on EXACTLY the local outer folds
(Results/splits_image.json is embedded and passed via SPLITS_JSON, because scikit-learn versions shuffle folds
differently). References (plain MT-ST, SVM, class-balanced SVM) are trained in the same run, so all tests are paired."""
import base64
import json
import os
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Kaggle", "sapota_improve.ipynb")


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


def wf(name):
    return code(f"%%writefile {name}\n" + open(os.path.join(HERE, name), encoding="utf-8").read())


emb = base64.b64encode(zlib.compress(open(os.path.join(HERE, "..", "Results", "splits_image.json"), "rb").read(), 9)).decode()
cells = [
    code("EMB_SPLITS = '" + emb + "'"),
    code("""
import glob, os, shutil, sys, base64, zlib
WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
INP = '/kaggle/input' if os.path.isdir('/kaggle/input') else os.getcwd()
os.chdir(WORK); os.makedirs('data', exist_ok=True); os.makedirs('Results', exist_ok=True)
for n in ['spectra.npz', 'unlabelled_pixels.npz', 'extraction_log.json']:
    shutil.copy(glob.glob(f'{INP}/**/{n}', recursive=True)[0], 'data/')
open('Results/splits_image.json', 'wb').write(zlib.decompress(base64.b64decode(EMB_SPLITS)))
os.environ.update(DATA_DIR=f'{WORK}/data', RESULTS_DIR=f'{WORK}/Results', SPLITS_JSON=f'{WORK}/Results/splits_image.json')
import torch, sklearn; print(torch.__version__, sklearn.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
"""),
    wf("models.py"), wf("run_experiments.py"),
    code("!python run_experiments.py improve"),
    code("!python run_experiments.py improve_stats"),
    code("""
os.makedirs('bundle', exist_ok=True); shutil.copytree('Results', 'bundle/Results', dirs_exist_ok=True)
shutil.make_archive('results_bundle', 'zip', 'bundle'); print(sorted(os.listdir('Results')))
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", os.path.abspath(OUT))
