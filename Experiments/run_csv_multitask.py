"""Multi-task experiment on the tabular data set Dataset/dataset.csv (sugar + carbohydrate regression and ripeness
classification), using the same preprocessing, training loop and statistics as run_experiments.py.

IMPORTANT (stated in the manuscript): the sugar/carbohydrate values in dataset.csv are model-generated (LLM) estimates,
not laboratory measurements, and its 'Category' label is a tertile of the sugar value. Results on this table therefore
characterise the learning behaviour of the models on proxy targets, not chemical accuracy.

Structure: 520 rows = 52 source records x 10 augmented copies (Sample_ID '<base>_aug_<k>'); 145 bands.
Grouping unit for every split: the source record (base id + Original_Category), so copies never cross folds.

Usage (from Submission_Package/):  python Experiments/run_csv_multitask.py main   |   stats
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.svm import SVR
from xgboost import XGBRegressor

import run_experiments as R
from models import MT_ST, Baseline1DCNN, BaselineViT1D

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.environ.get("CSV_PATH", os.path.join(HERE, "..", "Dataset", "dataset.csv"))
NB = 145
TARGETS = ["sugar", "carbohydrate"]


def load_csv():
    d = pd.read_csv(CSV)
    base = d["Sample_ID"].str.replace(r"_aug_\d+", "", regex=True)
    X = d[[f"Band_{i}" for i in range(1, NB + 1)]].to_numpy(float)
    return dict(Xm=R.preprocess(X), y=np.array([R.STAGES.index(c) for c in d["Category"]], dtype=np.int64),
                yr=d[["Sugar_Percentage", "Carbohydrate_Percentage"]].to_numpy(float),
                g=(base + "|" + d["Original_Category"]).to_numpy())


def splits(D):
    out = []
    for r in range(R.N_REPEATS):
        cv = StratifiedGroupKFold(R.N_FOLDS, shuffle=True, random_state=r)
        for k, (tr, te) in enumerate(cv.split(D["Xm"], D["y"], D["g"])):
            out.append((r, k, tr, te))
    return out


REGRESSORS = {
    "PLS regression": (PLSRegression(scale=False), {"n_components": [2, 4, 6, 8, 10, 15]}),
    "Ridge regression": (Ridge(), {"alpha": [0.1, 1, 10, 100]}),
    "SVR (RBF)": (SVR(), {"C": [1, 10, 100], "gamma": ["scale", 1e-3], "epsilon": [0.1]}),
    "Random forest": (RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=0),
                      {"max_depth": [None, 8], "max_features": ["sqrt", 0.3]}),
    "XGBoost": (XGBRegressor(learning_rate=0.1, subsample=0.8, n_jobs=4, random_state=0, verbosity=0),
                {"n_estimators": [100, 300], "max_depth": [2, 4]}),
}
DEEP = {  # all trained with the same multi-task loss (lambda_cls = 0.5) unless stated otherwise
    "1D-CNN (multi-task)": (lambda: Baseline1DCNN(NB), dict(lam_cls=0.5)),
    "1D-ViT (multi-task)": (lambda: BaselineViT1D(NB), dict(lam_cls=0.5)),
    "MT-ST (multi-task, lambda_cls=0.5)": (lambda: MT_ST(NB), dict(lam_cls=0.5)),
}
MTST_VARIANTS = {  # fixed lr 1e-3 for the task-weighting study
    "MT-ST regression only (lambda_cls=0)": dict(lam_cls=0.0),
    "MT-ST lambda_cls=0.25": dict(lam_cls=0.25),
    "MT-ST lambda_cls=0.5 (fixed lr)": dict(lam_cls=0.5),
    "MT-ST lambda_cls=1": dict(lam_cls=1.0),
    "MT-ST lambda_cls=2": dict(lam_cls=2.0),
    "MT-ST classification only": dict(cls_only=True),
}


def select_lr_mt(make, X, y, g, yr_s, seed, **kw):
    """Inner grouped hold-out; criterion = validation MSE of the standardised regression targets."""
    tr, va = next(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(X, y, g))
    sc = []
    for lr in R.LR_GRID:
        m = R.train_deep(make, X[tr], y[tr], yr_s[tr], lr=lr, seed=seed, **kw)
        sc.append(np.mean((R.predict_deep(m, X[va])[0] - yr_s[va]) ** 2))
    return R.LR_GRID[int(np.argmin(sc))]


def run_main():
    D = load_csv()
    S = splits(D)
    n = len(D["y"])
    names = [*REGRESSORS, *DEEP, *MTST_VARIANTS]
    reg = {m: np.full((R.N_REPEATS, n, 2), np.nan) for m in names}
    cls = {m: np.full((R.N_REPEATS, n), -1) for m in [*DEEP, *MTST_VARIANTS, "SVM (RBF) classifier", "Majority class"]}
    folds = {m: [] for m in set(names) | set(cls)}
    chosen = {m: [] for m in [*REGRESSORS, *DEEP]}
    for r, k, tr, te in S:
        t0 = time.time()
        sc = R.Scaler().fit(D["Xm"][tr])
        Xa, Xb = sc(D["Xm"][tr]), sc(D["Xm"][te])
        ytr, gtr, yr_tr = D["y"][tr], D["g"][tr], D["yr"][tr]
        rm, rs = yr_tr.mean(0), yr_tr.std(0)
        yr_s = (yr_tr - rm) / rs
        inner = GroupKFold(3)
        for name, (est, grid) in REGRESSORS.items():
            bp = []
            for j in range(2):
                gs = GridSearchCV(clone(est), grid, scoring="neg_root_mean_squared_error", cv=inner, n_jobs=1)
                gs.fit(Xa, yr_tr[:, j], groups=gtr)
                reg[name][r, te, j] = np.ravel(gs.predict(Xb))
                bp.append(gs.best_params_)
            chosen[name].append(bp)
        est, _ = R.fit_classical("SVM (RBF)", Xa, ytr, gtr)
        cls["SVM (RBF) classifier"][r, te] = est.predict(Xb)
        cls["Majority class"][r, te] = np.bincount(ytr).argmax()
        seed = 1000 * r + k
        for name, (make, kw) in DEEP.items():
            lr = select_lr_mt(make, Xa, ytr, gtr, yr_s, seed, **kw)
            m = R.train_deep(make, Xa, ytr, yr_s, lr=lr, seed=seed, **kw)
            p_reg, p_cls = R.predict_deep(m, Xb)
            reg[name][r, te] = p_reg * rs + rm
            cls[name][r, te] = p_cls.argmax(1)
            chosen[name].append({"lr": lr})
        for name, kw in MTST_VARIANTS.items():
            if kw.get("cls_only"):
                m = R.train_deep(lambda: MT_ST(NB), Xa, ytr, None, lr=1e-3, seed=seed)
            else:
                m = R.train_deep(lambda: MT_ST(NB), Xa, ytr, yr_s, lr=1e-3, seed=seed, **kw)
            p_reg, p_cls = R.predict_deep(m, Xb)
            if not kw.get("cls_only"):
                reg[name][r, te] = p_reg * rs + rm
            if kw.get("lam_cls", 1) > 0:
                cls[name][r, te] = p_cls.argmax(1)
        for name in folds:
            row = dict(r=r, k=k, n_test=len(te), n_test_records=len(set(D["g"][te])))
            if name in reg and not np.isnan(reg[name][r, te]).any():
                for j, t in enumerate(TARGETS):
                    row[t] = R.reg_metrics(D["yr"][te, j], reg[name][r, te, j])
            if name in cls and (cls[name][r, te] >= 0).all():
                row["cls"] = R.cls_metrics(D["y"][te], cls[name][r, te])
            folds[name].append(row)
        msg = "  ".join(f"{m.split(' (')[0][:14]}={folds[m][-1]['sugar']['rmse']:.2f}" for m in
                        ["PLS regression", "SVR (RBF)", "MT-ST (multi-task, lambda_cls=0.5)", "MT-ST regression only (lambda_cls=0)"])
        print(f"repeat {r} fold {k}: sugar RMSE {msg} ({time.time() - t0:.0f}s)", flush=True)

    summary = {}
    for name in sorted(set(names) | set(cls)):
        s = {}
        if name in reg and not np.isnan(reg[name]).any():
            for j, t in enumerate(TARGETS):
                reps = [R.reg_metrics(D["yr"][:, j], reg[name][rr, :, j]) for rr in range(R.N_REPEATS)]
                s[t] = {mt: [float(np.mean([x[mt] for x in reps])), float(np.std([x[mt] for x in reps], ddof=1))] for mt in reps[0]}
        if name in cls and (cls[name] >= 0).all():
            reps = [R.cls_metrics(D["y"], cls[name][rr]) for rr in range(R.N_REPEATS)]
            s["cls"] = {mt: [float(np.mean([x[mt] for x in reps])), float(np.std([x[mt] for x in reps], ddof=1))] for mt in reps[0]}
            s["cls"]["confusion_mean"] = np.mean([confusion_matrix(D["y"], cls[name][rr], labels=[0, 1, 2]) for rr in range(R.N_REPEATS)], 0)
        summary[name] = s
        print(name, {t: round(v["rmse"][0], 3) if t != "cls" else round(v["acc"][0], 3) for t, v in s.items()}, flush=True)
    R.save(dict(protocol=dict(n_rows=n, n_records=int(len(set(D["g"]))), bands=NB, folds=R.N_FOLDS, repeats=R.N_REPEATS,
                              target_stats={t: dict(min=float(D["yr"][:, j].min()), max=float(D["yr"][:, j].max()),
                                                    mean=float(D["yr"][:, j].mean()), sd=float(D["yr"][:, j].std(ddof=1)))
                                            for j, t in enumerate(TARGETS)},
                              class_counts=np.bincount(D["y"]).tolist(),
                              test_rows_per_fold=[f["n_test"] for f in folds["PLS regression"]],
                              test_records_per_fold=[f["n_test_records"] for f in folds["PLS regression"]]),
                summary=summary, per_fold=folds, reg_preds=reg, cls_preds=cls, chosen_params=chosen,
                yr=D["yr"], y=D["y"], groups=D["g"]), "csv_main.json")


def run_stats(B=2000):
    M = R.J_load("csv_main.json")
    yr, g = np.array(M["yr"]), np.array(M["groups"])
    rec = np.unique(g)
    idx = [np.where(g == x)[0] for x in rec]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(rec), len(rec))]) for _ in range(B)]
    ci = {}
    for name, P in M["reg_preds"].items():
        P = np.array(P, dtype=float)
        if np.isnan(P).any():
            continue
        ci[name] = {}
        for j, t in enumerate(TARGETS):
            rm = [np.mean([np.sqrt(np.mean((yr[b, j] - P[rr, b, j]) ** 2)) for rr in range(len(P))]) for b in boots]
            r2 = [np.mean([1 - np.sum((yr[b, j] - P[rr, b, j]) ** 2) / np.sum((yr[b, j] - yr[b, j].mean()) ** 2)
                           for rr in range(len(P))]) for b in boots]
            ci[name][t] = dict(rmse=np.percentile(rm, [2.5, 97.5]), r2=np.percentile(r2, [2.5, 97.5]))
    ref = "MT-ST (multi-task, lambda_cls=0.5)"
    tests = {}
    for t in TARGETS:
        a = np.array([f[t]["rmse"] for f in M["per_fold"][ref]])
        tests[t] = {}
        for name, fl in M["per_fold"].items():
            if name == ref or t not in fl[0]:
                continue
            d = np.array([f[t]["rmse"] for f in fl]) - a  # positive = MT-ST has lower RMSE
            J_ = len(d)
            tt = d.mean() / np.sqrt((1 / J_ + 0.25) * d.var(ddof=1)) if d.var() > 0 else 0.0
            tests[t][name] = dict(mean_rmse_diff=float(d.mean()), t=float(tt), p_t=float(2 * stats.t.sf(abs(tt), J_ - 1)),
                                  p_wilcoxon=float(stats.wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d) else 1.0)
        for key in ["p_t", "p_wilcoxon"]:
            names = sorted(tests[t], key=lambda n_: tests[t][n_][key])
            run = 0
            for i, n_ in enumerate(names):
                run = max(run, min(1.0, (len(names) - i) * tests[t][n_][key]))
                tests[t][n_][key + "_holm"] = run
    R.save(dict(bootstrap_ci_95=ci, tests_vs_mtst=tests, n_boot=B), "csv_stats.json")
    print(json.dumps(tests, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["main", "stats"])
    a = ap.parse_args()
    dict(main=run_main, stats=run_stats)[a.task]()
