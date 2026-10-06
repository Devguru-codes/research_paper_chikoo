"""Build Kaggle/sapota_compute.ipynb: repeats `run_experiments.py compute` N_RUNS times on a Kaggle GPU machine and
writes Results/compute.json with the per-metric MEDIAN over runs (same format as a single run) plus the min/max range
and the raw runs, so the paper's computational-cost table is measured under controlled, reproducible conditions."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Kaggle", "sapota_compute.ipynb")


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": t.strip("\n").splitlines(keepends=True)}


def wf(name):
    return code(f"%%writefile {name}\n" + open(os.path.join(HERE, name), encoding="utf-8").read())


cells = [
    code("""
import glob, os, shutil, subprocess, sys, json
WORK = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.getcwd()
INP = '/kaggle/input' if os.path.isdir('/kaggle/input') else os.getcwd()
os.chdir(WORK); os.makedirs('data', exist_ok=True); os.makedirs('Results', exist_ok=True)
os.environ.update(DATA_DIR=f'{WORK}/data', RESULTS_DIR=f'{WORK}/Results')
shutil.copy(glob.glob(f'{INP}/**/spectra.npz', recursive=True)[0], 'data/')
print(subprocess.run(['nvidia-smi', '--query-gpu=name,clocks.max.sm', '--format=csv'], capture_output=True, text=True).stdout)
print(subprocess.run(['lscpu'], capture_output=True, text=True).stdout[:600])
"""),
    wf("models.py"), wf("run_experiments.py"),
    code("""
N_RUNS = 3
for i in range(N_RUNS):
    r = subprocess.run([sys.executable, 'run_experiments.py', 'compute'], capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-800:])
    shutil.move('Results/compute.json', f'Results/compute_run{i}.json')
"""),
    code("""
import numpy as np
runs = [json.load(open(f'Results/compute_run{i}.json')) for i in range(N_RUNS)]
def agg(vals):
    if isinstance(vals[0], dict):
        return {k: agg([v[k] for v in vals]) for k in vals[0]}
    if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
        return float(np.median(vals))
    return vals[0]
out = {m: agg([r[m] for r in runs]) for m in runs[0] if m != 'hardware'}
out['hardware'] = runs[0]['hardware']
out['range'] = {m: {d: [min(r[m]['latency'][d]['median_ms'] for r in runs), max(r[m]['latency'][d]['median_ms'] for r in runs)]
                    for d in runs[0][m]['latency']} for m in out if m not in ('hardware',)}
out['n_runs'] = N_RUNS
json.dump(out, open('Results/compute.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
os.makedirs('bundle', exist_ok=True); shutil.copytree('Results', 'bundle/Results', dirs_exist_ok=True)
shutil.make_archive('compute_bundle', 'zip', 'bundle')
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", os.path.abspath(OUT))
