"""Reproducible evaluation of MT-ST and baselines on the real Pika NIR-320 sapota spectra.

Usage (from Submission_Package/):
  python Experiments/extract_spectra.py
  python Experiments/run_experiments.py main        # all models, repeated grouped CV, + XAI per fold
  python Experiments/run_experiments.py ablation    # MT-ST architectural ablations
  python Experiments/run_experiments.py augment     # noise-augmentation sensitivity
  python Experiments/run_experiments.py bandsel     # IG/attention stability + band-selection test (after main)
  python Experiments/run_experiments.py compute     # params, FLOPs, latency, VRAM
  python Experiments/run_experiments.py stats       # CIs + significance tests (after main)
Optional: --labels FILE.csv with columns fruit_id,sugar,carbohydrate (measured reference values) enables the
regression heads (multi-task loss) and the regression metrics; without it only ripeness is modelled.

Protocol
- Unit of analysis: one fruit view (mean ROI spectrum). Grouping unit: physical fruit, so every view and every
  pixel patch of a fruit is always on the same side of a split.
- Outer CV: StratifiedGroupKFold(5) repeated R times with different shuffles (identical splits for all models).
- Preprocessing (all models): SNV -> Savitzky-Golay 1st derivative (window 15, order 2) per spectrum, then per-band
  z-scoring fitted on the training fold only. First 160 bands (903-1689 nm) so that 160 = 32 patches x 5 bands.
- Training rows: the view-mean spectra plus N_PATCH random 5x5-pixel patch means per view (training folds only);
  test rows: view-mean spectra only.
- Hyper-parameters: classical models by grid search with inner StratifiedGroupKFold(3); deep models choose the
  learning rate from {1e-3, 3e-4} on an inner grouped hold-out, then are retrained on the whole training fold.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

from models import MT_ST, Baseline1DCNN, BaselineViT1D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "Results"))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
STAGES = ["Unripe", "Ripe", "Overripe"]
N_BANDS, N_FOLDS, N_PATCH = 160, 5, 10
N_REPEATS = int(os.environ.get("N_REPEATS", 5))
EPOCHS, BATCH, LR_GRID = int(os.environ.get("EPOCHS", 60)), 32, (1e-3, 3e-4)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- data
def preprocess(X):
    X = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-8)
    return savgol_filter(X, window_length=15, polyorder=2, deriv=1, axis=-1)


def load(labels_csv=None):
    d = np.load(os.path.join(DATA, "spectra.npz"))
    keep = d["stage"] != "Unlabelled"
    D = dict(wl=d["wl"][:N_BANDS], Xm=preprocess(d["X_mean"][keep][:, :N_BANDS]),
             Xp=preprocess(d["X_patch"][keep][:, :N_PATCH, :N_BANDS]), raw=d["X_mean"][keep][:, :N_BANDS],
             y=np.array([STAGES.index(s) for s in d["stage"][keep]], dtype=np.int64), g=d["fruit_id"][keep],
             cube=d["cube"][keep], yr=None)
    if labels_csv:
        import pandas as pd
        lab = pd.read_csv(labels_csv).set_index("fruit_id")
        D["yr"] = lab.loc[D["g"], ["sugar", "carbohydrate"]].to_numpy(float)
    return D


def splits(D, repeats=N_REPEATS):
    """Outer folds. If SPLITS_JSON is set, the folds are read from that file (written by `save_splits`) so that runs on
    machines with a different scikit-learn version use exactly the same folds as the reference run."""
    sj = os.environ.get("SPLITS_JSON")
    if sj and os.path.exists(sj):
        return [(f["r"], f["k"], np.array(f["tr"]), np.array(f["te"])) for f in json.load(open(sj)) if f["r"] < repeats]
    out = []
    for r in range(repeats):
        cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=r)
        for k, (tr, te) in enumerate(cv.split(D["Xm"], D["y"], D["g"])):
            out.append((r, k, tr, te))
    return out


def train_rows(D, idx):
    """View means + patch means of the given views; returns X, y, groups, yr."""
    X = np.concatenate([D["Xm"][idx], D["Xp"][idx].reshape(-1, N_BANDS)])
    rep = np.concatenate([idx, np.repeat(idx, N_PATCH)])
    return X, D["y"][rep], D["g"][rep], (None if D["yr"] is None else D["yr"][rep])


class Scaler:
    def fit(self, X):
        self.m, self.s = X.mean(0), X.std(0) + 1e-8
        return self

    def __call__(self, X):
        return (X - self.m) / self.s


# ----------------------------------------------------------------------------- classical models
class PLSDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.pls_ = PLSRegression(self.n_components, scale=False).fit(X, np.eye(len(self.classes_))[y])
        return self

    def predict(self, X):
        return self.classes_[self.pls_.predict(X).argmax(1)]


CLASSICAL = {
    "PLS-DA": (PLSDA(), {"n_components": [2, 3, 5, 8, 10]}),
    "Logistic regression": (LogisticRegression(max_iter=5000), {"C": [0.01, 0.1, 1, 10]}),
    "SVM (RBF)": (SVC(), {"C": [1, 10, 100], "gamma": ["scale", 1e-3]}),
    "Random forest": (RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=0),
                      {"max_depth": [None, 8], "max_features": ["sqrt", 0.3]}),
    "XGBoost": (XGBClassifier(learning_rate=0.1, subsample=0.8, n_jobs=4, random_state=0, verbosity=0),
                {"n_estimators": [100, 300], "max_depth": [2, 4]}),
}


def fit_classical(name, X, y, g):
    est, grid = CLASSICAL[name]
    gs = GridSearchCV(est, grid, scoring="balanced_accuracy", cv=StratifiedGroupKFold(3, shuffle=True, random_state=0),
                      n_jobs=1, refit=True)
    gs.fit(X, y, groups=g)
    return gs.best_estimator_, gs.best_params_


# ----------------------------------------------------------------------------- deep models
DEEP = {
    "1D-CNN": lambda: Baseline1DCNN(N_BANDS),
    "1D-ViT": lambda: BaselineViT1D(N_BANDS),
    "MT-ST": lambda: MT_ST(N_BANDS),
}


def seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)
    torch.cuda.manual_seed_all(s)


def train_deep(make, X, y, yr=None, lr=1e-3, seed=0, noise=0.05, p_noise=0.5, lam_cls=0.5, epochs=EPOCHS,
               class_weight=None, init_state=None):
    """AdamW (wd 1e-2) + cosine schedule + grad-clip 1.0. Loss = MSE(sugar)+MSE(carb) [if labels] + lam*CE.
    Without reference labels the loss is CE only. Noise: x + N(0, noise^2) on z-scored inputs with prob p_noise.
    class_weight: optional per-class CE weights; init_state: optional pretrained state_dict (strict=False)."""
    seed_all(seed)
    m = make().to(DEV)
    if init_state is not None:
        m.load_state_dict(init_state, strict=False)
    cw = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = torch.tensor(X, dtype=torch.float32, device=DEV), torch.tensor(y, dtype=torch.long, device=DEV)
    yrt = None if yr is None else torch.tensor(yr, dtype=torch.float32, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), device=DEV, generator=gen)
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 2:
                continue
            xb = Xt[b]
            if noise > 0 and torch.rand(1, generator=gen, device=DEV).item() < p_noise:
                xb = xb + noise * torch.randn(xb.shape, device=DEV, generator=gen)
            reg, cls = m(xb)
            loss = (lam_cls if yrt is not None else 1.0) * F.cross_entropy(cls, yt[b], weight=cw)
            if yrt is not None:
                loss = loss + F.mse_loss(reg, yrt[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sch.step()
    return m.eval()


@torch.no_grad()
def predict_deep(m, X):
    reg, cls = m(torch.tensor(X, dtype=torch.float32, device=DEV))
    return reg.cpu().numpy(), cls.softmax(1).cpu().numpy()


def select_lr(make, X, y, g, yr, seed, **kw):
    tr, va = next(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(X, y, g))
    scores = []
    for lr in LR_GRID:
        m = train_deep(make, X[tr], y[tr], None if yr is None else yr[tr], lr=lr, seed=seed, **kw)
        scores.append(balanced_accuracy_score(y[va], predict_deep(m, X[va])[1].argmax(1)))
    return LR_GRID[int(np.argmax(scores))]


# ----------------------------------------------------------------------------- XAI
def integrated_gradients(m, X, target, baseline, steps=64):
    m.eval()
    x = torch.tensor(X, dtype=torch.float32, device=DEV)
    b = torch.tensor(baseline, dtype=torch.float32, device=DEV).expand_as(x)
    t = torch.tensor(target, dtype=torch.long, device=DEV)
    total = torch.zeros_like(x)
    for a in torch.linspace(0, 1, steps + 1, device=DEV)[1:]:
        xi = (b + a * (x - b)).requires_grad_(True)
        logit = m(xi)[1].gather(1, t[:, None]).sum()
        total += torch.autograd.grad(logit, xi)[0]
    return ((x - b) * total / steps).detach().cpu().numpy()


def xai_profiles(m, Xtr, ytr, sc):
    """Per-band importance on the training fold's view means: |IG| with two baselines, and attention rollout."""
    zero_mean = np.zeros((1, N_BANDS))                 # = training-set mean spectrum (z-space origin)
    flat = sc(np.zeros((1, N_BANDS)))                  # featureless spectrum (zero derivative)
    ig_mean = np.abs(integrated_gradients(m, Xtr, ytr, zero_mean)).mean(0)
    ig_flat = np.abs(integrated_gradients(m, Xtr, ytr, flat)).mean(0)
    roll = m.attention_rollout(torch.tensor(Xtr, dtype=torch.float32, device=DEV)).mean(0).cpu().numpy()
    att = np.repeat(roll, m.patch_size)[:N_BANDS]
    return ig_mean, ig_flat, att


# ----------------------------------------------------------------------------- metrics
def cls_metrics(y, p):
    return dict(acc=accuracy_score(y, p), bacc=balanced_accuracy_score(y, p), f1=f1_score(y, p, average="macro"),
                kappa=cohen_kappa_score(y, p))


def reg_metrics(t, p):
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    return dict(rmse=rmse, mae=float(np.mean(np.abs(t - p))), r2=float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)),
                rpd=float(t.std(ddof=1) / rmse))


def save(obj, name):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))


# ----------------------------------------------------------------------------- experiments
def run_main(D):
    S = splits(D)
    n = len(D["y"])
    out = {m: dict(pred=np.full((N_REPEATS, n), -1), folds=[], params=[]) for m in
           ["Majority class", *CLASSICAL, *DEEP]}
    if D["yr"] is not None:
        for m in out:
            out[m]["reg"] = np.full((N_REPEATS, n, 2), np.nan)
    xai = []
    for r, k, tr, te in S:
        t0 = time.time()
        Xtr, ytr, gtr, yrtr = train_rows(D, tr)
        sc = Scaler().fit(Xtr)
        Xtr_s, Xte_s = sc(Xtr), sc(D["Xm"][te])
        yr_sc = None
        if yrtr is not None:
            rm, rs = yrtr.mean(0), yrtr.std(0)
            yr_sc = (yrtr - rm) / rs
        # majority class
        maj = np.bincount(ytr).argmax()
        out["Majority class"]["pred"][r, te] = maj
        for name in CLASSICAL:
            est, bp = fit_classical(name, Xtr_s, ytr, gtr)
            out[name]["pred"][r, te] = est.predict(Xte_s)
            out[name]["params"].append(bp)
        for name, make in DEEP.items():
            seed = 1000 * r + k
            lr = select_lr(make, Xtr_s, ytr, gtr, yr_sc, seed)
            m = train_deep(make, Xtr_s, ytr, yr_sc, lr=lr, seed=seed)
            reg, prob = predict_deep(m, Xte_s)
            out[name]["pred"][r, te] = prob.argmax(1)
            out[name]["params"].append({"lr": lr})
            if yr_sc is not None:
                out[name]["reg"][r, te] = reg * rs + rm
            if name == "MT-ST":
                iv = np.arange(len(tr))  # first len(tr) rows of Xtr are the training view means
                ig_m, ig_f, att = xai_profiles(m, Xtr_s[iv], ytr[iv], sc)
                xai.append(dict(r=r, k=k, ig_mean=ig_m, ig_flat=ig_f, att=att))
        for name in out:
            yt, yp = D["y"][te], out[name]["pred"][r, te]
            out[name]["folds"].append(dict(r=r, k=k, n_test=len(te), n_train_views=len(tr), **cls_metrics(yt, yp)))
        print(f"repeat {r} fold {k}: n_tr={len(tr)} n_te={len(te)}  " +
              "  ".join(f"{m}={out[m]['folds'][-1]['acc']:.2f}" for m in out) + f"  ({time.time() - t0:.0f}s)", flush=True)
    summary = {}
    for name, o in out.items():
        reps = [cls_metrics(D["y"], o["pred"][r]) for r in range(N_REPEATS)]
        summary[name] = {mt: [float(np.mean([x[mt] for x in reps])), float(np.std([x[mt] for x in reps], ddof=1))]
                         for mt in reps[0]}
        summary[name]["confusion_mean"] = np.mean([confusion_matrix(D["y"], o["pred"][r], labels=[0, 1, 2])
                                                   for r in range(N_REPEATS)], 0)
        if "reg" in o and not np.isnan(o["reg"]).all():
            for j, tgt in enumerate(["sugar", "carbohydrate"]):
                rr = [reg_metrics(D["yr"][:, j], o["reg"][r, :, j]) for r in range(N_REPEATS)]
                summary[name][tgt] = {mt: [float(np.mean([x[mt] for x in rr])), float(np.std([x[mt] for x in rr], ddof=1))]
                                      for mt in rr[0]}
    save(dict(protocol=dict(n_views=int(len(D["y"])), n_fruits=int(len(set(D["g"]))),
                            views_per_stage=np.bincount(D["y"]).tolist(),
                            fruits_per_stage=[int(len(set(D["g"][D["y"] == c]))) for c in range(3)],
                            folds=N_FOLDS, repeats=N_REPEATS, n_patch=N_PATCH, epochs=EPOCHS, batch=BATCH,
                            bands=N_BANDS, wl_first=float(D["wl"][0]), wl_last=float(D["wl"][-1]),
                            test_views_per_fold=[f["n_test"] for f in out["MT-ST"]["folds"]]),
              summary=summary, per_fold={k: v["folds"] for k, v in out.items()},
              preds={k: v["pred"] for k, v in out.items()}, chosen_params={k: v["params"] for k, v in out.items()},
              y=D["y"], groups=D["g"]), "main.json")
    save(dict(wl=D["wl"], profiles=xai), "xai_profiles.json")


def run_variants(D, variants, fname, repeats=N_REPEATS):
    S = splits(D, repeats)
    res = {}
    for vname, (make, kw) in variants.items():
        folds, pred = [], np.full((repeats, len(D["y"])), -1)
        for r, k, tr, te in S:
            Xtr, ytr, _, yrtr = train_rows(D, tr)
            if kw.get("no_patches"):
                Xtr, ytr = D["Xm"][tr], D["y"][tr]
            sc = Scaler().fit(Xtr)
            tkw = {a: b for a, b in kw.items() if a != "no_patches"}
            m = train_deep(make, sc(Xtr), ytr, None, lr=1e-3, seed=1000 * r + k, **tkw)
            p = predict_deep(m, sc(D["Xm"][te]))[1].argmax(1)
            pred[r, te] = p
            folds.append(dict(r=r, k=k, **cls_metrics(D["y"][te], p)))
            print(f"  {vname} | repeat {r} fold {k}: acc={folds[-1]['acc']:.2f}", flush=True)
        reps = [cls_metrics(D["y"], pred[r]) for r in range(repeats)]
        res[vname] = dict(summary={mt: [float(np.mean([x[mt] for x in reps])), float(np.std([x[mt] for x in reps], ddof=1))]
                                   for mt in reps[0]}, folds=folds)
        print(vname, {a: round(b[0], 3) for a, b in res[vname]["summary"].items()}, flush=True)
    save(res, fname)


def run_ablation(D):
    V = {
        "Base MT-ST (4 layers, 8 heads, dropout 0.1, sinusoidal PE)": (lambda: MT_ST(N_BANDS), {}),
        "w/o positional encoding": (lambda: MT_ST(N_BANDS, use_pe=False), {}),
        "w/o dropout": (lambda: MT_ST(N_BANDS, dropout=0.0), {}),
        "2 encoder layers": (lambda: MT_ST(N_BANDS, num_layers=2), {}),
        "4 attention heads": (lambda: MT_ST(N_BANDS, num_heads=4), {}),
        "16 attention heads": (lambda: MT_ST(N_BANDS, num_heads=16), {}),
        "Patch size 10 (16 tokens)": (lambda: MT_ST(N_BANDS, patch_size=10), {}),
        "Patch size 1 (160 tokens)": (lambda: MT_ST(N_BANDS, patch_size=1), {}),
    }
    run_variants(D, V, "ablation.json")


def run_augment(D):
    mk = lambda: MT_ST(N_BANDS)
    V = {f"sigma={s:g}, p=0.5": (mk, dict(noise=s, p_noise=0.5)) for s in [0.0, 0.01, 0.025, 0.05, 0.1, 0.2]}
    V["sigma=0.05, p=1.0"] = (mk, dict(noise=0.05, p_noise=1.0))
    V["sigma=0.05, p=0.5, no pixel-patch spectra"] = (mk, dict(noise=0.05, p_noise=0.5, no_patches=True))
    run_variants(D, V, "augment.json")


def run_bandsel(D, k_top=16, n_rand=20):
    """Stability of attributions across folds and a retraining test: SVM on the top-k bands ranked on the TRAINING
    fold vs. k random bands, evaluated on the held-out fold."""
    P = json.load(open(os.path.join(RES, "xai_profiles.json")))
    prof = {p_["r"] * 100 + p_["k"]: p_ for p_ in P["profiles"]}
    S = splits(D)

    def stability(key):
        A = np.array([np.array(p_[key]) for p_ in P["profiles"]])
        rho = [stats.spearmanr(A[i], A[j])[0] for i in range(len(A)) for j in range(i + 1, len(A))]
        tops = [set(np.argsort(-a)[:k_top]) for a in A]
        jac = [len(tops[i] & tops[j]) / len(tops[i] | tops[j]) for i in range(len(A)) for j in range(i + 1, len(A))]
        return dict(spearman=[float(np.mean(rho)), float(np.std(rho))], jaccard_top_k=[float(np.mean(jac)), float(np.std(jac))])

    stab = {key: stability(key) for key in ["ig_mean", "ig_flat", "att"]}
    agree = {f"{a} vs {b}": float(np.mean([stats.spearmanr(p_[a], p_[b])[0] for p_ in P["profiles"]]))
             for a, b in [("ig_mean", "ig_flat"), ("ig_mean", "att"), ("ig_flat", "att")]}
    rng = np.random.default_rng(0)
    rows = {"All 160 bands": [], "Top-k IG bands": [], "Top-k attention bands": [], "Random k bands": []}
    for r, k, tr, te in S:
        Xtr, ytr, _, _ = train_rows(D, tr)
        sc = Scaler().fit(Xtr)
        Xa, Xb = sc(Xtr), sc(D["Xm"][te])
        pp = prof[r * 100 + k]
        sets = {"All 160 bands": np.arange(N_BANDS), "Top-k IG bands": np.argsort(-np.array(pp["ig_mean"]))[:k_top],
                "Top-k attention bands": np.argsort(-np.array(pp["att"]))[:k_top]}
        for nm, idx in sets.items():
            rows[nm].append(accuracy_score(D["y"][te], SVC(C=10).fit(Xa[:, idx], ytr).predict(Xb[:, idx])))
        rows["Random k bands"].append(np.mean([accuracy_score(D["y"][te], SVC(C=10).fit(Xa[:, i], ytr).predict(Xb[:, i]))
                                               for i in [rng.choice(N_BANDS, k_top, replace=False) for _ in range(n_rand)]]))
    test = {nm: [float(np.mean(v)), float(np.std(v, ddof=1))] for nm, v in rows.items()}
    d = np.array(rows["Top-k IG bands"]) - np.array(rows["Random k bands"])
    save(dict(k_top=k_top, stability=stab, agreement=agree, band_selection_acc=test,
              ig_vs_random_wilcoxon_p=float(stats.wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d) else 1.0,
              top_bands_ig_nm=sorted({round(float(P["wl"][i])) for p_ in P["profiles"]
                                      for i in np.argsort(-np.array(p_["ig_mean"]))[:k_top]})), "bandsel.json")
    print(json.dumps(dict(stab=stab, agree=agree, test=test), indent=1))


def run_stats(D, B=2000):
    M = json.load(open(os.path.join(RES, "main.json")))
    y, g = np.array(M["y"]), np.array(M["groups"])
    fr = np.unique(g)
    idx_by_fruit = [np.where(g == f)[0] for f in fr]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx_by_fruit[i] for i in rng.integers(0, len(fr), len(fr))]) for _ in range(B)]
    ci = {}
    for name, P in M["preds"].items():
        P = np.array(P)
        acc = [np.mean([accuracy_score(y[b], P[r, b]) for r in range(len(P))]) for b in boots]
        f1 = [np.mean([f1_score(y[b], P[r, b], average="macro") for r in range(len(P))]) for b in boots]
        ci[name] = dict(acc=np.percentile(acc, [2.5, 97.5]), f1=np.percentile(f1, [2.5, 97.5]))
    # corrected resampled t-test (Nadeau & Bengio, 2003) and Wilcoxon on paired per-fold accuracy, Holm-adjusted
    ref = np.array([f["acc"] for f in M["per_fold"]["MT-ST"]])
    tests = {}
    for name, fl in M["per_fold"].items():
        if name == "MT-ST":
            continue
        d = ref - np.array([f["acc"] for f in fl])
        J = len(d)
        ratio = np.mean([f["n_test"] / (len(y) - f["n_test"]) for f in fl])
        t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1)) if d.var() > 0 else 0.0
        tests[name] = dict(mean_diff=float(d.mean()), t=float(t), p_t=float(2 * stats.t.sf(abs(t), J - 1)),
                           p_wilcoxon=float(stats.wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d) else 1.0)
    for key in ["p_t", "p_wilcoxon"]:
        names = sorted(tests, key=lambda n: tests[n][key])
        run = 0
        for i, n in enumerate(names):
            run = max(run, min(1.0, (len(names) - i) * tests[n][key]))
            tests[n][key + "_holm"] = run
    save(dict(bootstrap_ci_95=ci, n_boot=B, tests_vs_mtst=tests), "stats.json")
    print(json.dumps(tests, indent=1))


def pretrain_mtst(seed, epochs=int(os.environ.get("PRETRAIN_EPOCHS", 100)), mask_ratio=0.3, batch=128, lr=1e-3):
    """Self-supervised masked-token pretraining of MT-ST on pixel spectra from the UNLABELLED sessions only
    (data/unlabelled_pixels.npz; different fruits from the labelled set). 30 % of the 32 patch tokens are replaced
    by a learned [MASK] embedding and a linear head reconstructs the 5 preprocessed band values of each masked patch."""
    U = np.load(os.path.join(DATA, "unlabelled_pixels.npz"))["X"][:, :N_BANDS]
    U = preprocess(U)
    U = Scaler().fit(U)(U)
    seed_all(seed)
    m = MT_ST(N_BANDS).to(DEV)
    d, P = m.cls_token.shape[-1], m.patch_size
    mask_tok = nn.Parameter(torch.zeros(1, 1, d, device=DEV))
    head = nn.Linear(d, P).to(DEV)
    params = list(m.parameters()) + list(head.parameters()) + [mask_tok]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Ut = torch.tensor(U, dtype=torch.float32, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    N = m.num_patches
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Ut), device=DEV, generator=gen)
        for i in range(0, len(Ut), batch):
            x = Ut[perm[i:i + batch]]
            B = len(x)
            t = m.patch_embed(x.unsqueeze(1)).transpose(1, 2)
            mask = torch.rand(B, N, device=DEV, generator=gen) < mask_ratio
            t = torch.where(mask[..., None], mask_tok.expand(B, N, d), t)
            t = m.pos_encoder(torch.cat((m.cls_token.expand(B, -1, -1), t), dim=1))
            rec = head(m.norm(m.transformer(t))[:, 1:, :])
            target = x[:, :N * P].view(B, N, P)
            loss = F.mse_loss(rec[mask], target[mask])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sch.step()
    return {k: v.detach().clone() for k, v in m.state_dict().items()}, float(loss.item())


def run_improve(D, n_ens=5):
    """Pre-registered MT-ST improvements, all on the same 25 outer folds as the main comparison. The final model
    'MT-ST+' (balanced CE + pretraining + 5-seed ensemble) was fixed BEFORE any of these results were seen; the
    individual components are reported as ablations. A class-balanced SVM is added for fairness. Fixed lr: 1e-3 for
    training from scratch, 3e-4 for fine-tuning a pretrained encoder (both chosen a priori)."""
    S = splits(D)
    n = len(D["y"])
    pre = {}
    for s in range(n_ens):
        t0 = time.time()
        pre[s], l = pretrain_mtst(seed=500 + s)
        print(f"pretrained seed {s}: final masked-reconstruction MSE {l:.3f} ({time.time() - t0:.0f}s)", flush=True)
    V = ["MT-ST + class-balanced loss", "MT-ST + self-supervised pretraining", "MT-ST + 5-seed ensemble",
         "MT-ST+ (balanced + pretraining + ensemble)", "SVM (RBF), class-balanced", "MT-ST (reference, fixed lr)",
         "SVM (RBF)"]
    pred = {v: np.full((N_REPEATS, n), -1) for v in V}
    folds = {v: [] for v in V}
    for r, k, tr, te in S:
        t0 = time.time()
        Xtr, ytr, gtr, _ = train_rows(D, tr)
        sc = Scaler().fit(Xtr)
        Xa, Xb = sc(Xtr), sc(D["Xm"][te])
        cnt = np.bincount(ytr, minlength=3)
        cw = (len(ytr) / (3 * np.maximum(cnt, 1))).tolist()
        mk = lambda: MT_ST(N_BANDS)
        base_seed = 1000 * r + k

        def ens(balanced, pretrained, members):
            P = 0
            for s in range(members):
                m = train_deep(mk, Xa, ytr, lr=3e-4 if pretrained else 1e-3, seed=base_seed + 100 * s,
                               class_weight=cw if balanced else None, init_state=pre[s] if pretrained else None)
                P = P + predict_deep(m, Xb)[1]
            return P.argmax(1)

        out = {V[0]: ens(True, False, 1), V[1]: ens(False, True, 1), V[2]: ens(False, False, n_ens),
               V[3]: ens(True, True, n_ens)}
        est, _ = GridSearchCV(SVC(class_weight="balanced"), CLASSICAL["SVM (RBF)"][1], scoring="balanced_accuracy",
                              cv=StratifiedGroupKFold(3, shuffle=True, random_state=0)).fit(Xa, ytr, groups=gtr), None
        out[V[4]] = est.best_estimator_.predict(Xb)
        out[V[5]] = ens(False, False, 1)
        out[V[6]] = fit_classical("SVM (RBF)", Xa, ytr, gtr)[0].predict(Xb)
        for v, p in out.items():
            pred[v][r, te] = p
            folds[v].append(dict(r=r, k=k, n_test=len(te), **cls_metrics(D["y"][te], p)))
        print(f"repeat {r} fold {k}: " + "  ".join(f"{v.split(' (')[0]}={folds[v][-1]['acc']:.2f}" for v in V)
              + f" ({time.time() - t0:.0f}s)", flush=True)
    summ = {}
    for v in V:
        reps = [cls_metrics(D["y"], pred[v][r]) for r in range(N_REPEATS)]
        summ[v] = {mt: [float(np.mean([x[mt] for x in reps])), float(np.std([x[mt] for x in reps], ddof=1))] for mt in reps[0]}
        summ[v]["confusion_mean"] = np.mean([confusion_matrix(D["y"], pred[v][r], labels=[0, 1, 2]) for r in range(N_REPEATS)], 0)
        print(v, {a: round(b[0], 3) for a, b in summ[v].items() if a != "confusion_mean"}, flush=True)
    save(dict(summary=summ, per_fold=folds, preds=pred, y=D["y"], groups=D["g"], n_ensemble=n_ens), "improve.json")


def run_improve_stats(D, B=2000):
    """CIs and paired corrected tests for MT-ST+ against the references trained in the SAME run (identical folds)."""
    I = J_load("improve.json")
    y, g = np.array(I["y"]), np.array(I["groups"])
    fr = np.unique(g)
    idx = [np.where(g == f)[0] for f in fr]
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[i] for i in rng.integers(0, len(fr), len(fr))]) for _ in range(B)]
    preds = dict(I["preds"])
    ci = {nm: np.percentile([np.mean([accuracy_score(y[b], np.array(P)[r, b]) for r in range(len(P))]) for b in boots],
                            [2.5, 97.5]) for nm, P in preds.items()}
    pf = dict(I["per_fold"])
    ref = np.array([f["acc"] for f in pf["MT-ST+ (balanced + pretraining + ensemble)"]])
    tests = {}
    for nm in ["SVM (RBF)", "SVM (RBF), class-balanced", "MT-ST (reference, fixed lr)", "MT-ST + class-balanced loss",
               "MT-ST + self-supervised pretraining", "MT-ST + 5-seed ensemble"]:
        d = ref - np.array([f["acc"] for f in pf[nm]])
        J_ = len(d)
        t = d.mean() / np.sqrt((1 / J_ + 0.25) * d.var(ddof=1)) if d.var() > 0 else 0.0
        tests[nm] = dict(mean_diff=float(d.mean()), t=float(t), p_t=float(2 * stats.t.sf(abs(t), J_ - 1)),
                         p_wilcoxon=float(stats.wilcoxon(d, zero_method="zsplit").pvalue) if np.any(d) else 1.0,
                         folds_mtstplus_better=int((d > 0).sum()), folds_tied=int((d == 0).sum()), folds_worse=int((d < 0).sum()))
    save(dict(bootstrap_ci_95=ci, tests_mtstplus_vs=tests), "improve_stats.json")
    print(json.dumps(tests, indent=1))


def J_load(name):
    return json.load(open(os.path.join(RES, name)))


def run_compute(D):
    from torch.utils.flop_counter import FlopCounterMode
    res = {}
    x1 = torch.randn(1, N_BANDS)
    for name, make in DEEP.items():
        m = make().eval()
        params = sum(p.numel() for p in m.parameters())
        with FlopCounterMode(display=False) as fc:
            m(x1)
        lat = {}
        for dev, threads in [("cuda", None), ("cpu", 1), ("cpu", 4)]:
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            if threads:
                torch.set_num_threads(threads)
            mm, xx = m.to(dev), x1.to(dev)
            ts = []
            with torch.no_grad():
                for i in range(1100):
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    t = time.perf_counter()
                    mm(xx)
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    if i >= 100:
                        ts.append((time.perf_counter() - t) * 1e3)
            lat[f"{dev}{'' if threads is None else f'_{threads}thr'}"] = dict(median_ms=float(np.median(ts)),
                                                                               p95_ms=float(np.percentile(ts, 95)))
        torch.set_num_threads(os.cpu_count())
        # training cost on the real training-fold size (one epoch) and peak GPU memory
        tr = splits(D, 1)[0][2]
        Xtr, ytr, _, _ = train_rows(D, tr)
        train_deep(make, Scaler().fit(Xtr)(Xtr), ytr, epochs=1)  # warm-up (CUDA context, cuDNN autotuning)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        train_deep(make, Scaler().fit(Xtr)(Xtr), ytr, epochs=5)
        torch.cuda.synchronize()
        res[name] = dict(params=params, mflops_per_spectrum=fc.get_total_flops() / 1e6, latency=lat,
                         sec_per_epoch=(time.perf_counter() - t) / 5, train_rows=len(Xtr),
                         peak_vram_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
        print(name, res[name], flush=True)
    import platform
    import subprocess
    cpu = platform.processor()
    try:
        if os.name == "nt":
            cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True).stdout.strip() or cpu
        elif os.path.exists("/proc/cpuinfo"):
            cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        pass
    res["hardware"] = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           cpu=cpu, torch=torch.__version__, cpu_threads=os.cpu_count())
    save(res, "compute.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["main", "ablation", "augment", "bandsel", "stats", "compute", "improve",
                                     "improve_stats"])
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    D = load(a.labels)
    print(f"{len(D['y'])} views, {len(set(D['g']))} fruits, stages={np.bincount(D['y'])}, device={DEV}", flush=True)
    dict(main=run_main, ablation=run_ablation, augment=run_augment, bandsel=run_bandsel, stats=run_stats,
         compute=run_compute, improve=run_improve, improve_stats=run_improve_stats)[a.task](D)
