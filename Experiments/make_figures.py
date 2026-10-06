"""Regenerate every data figure of the revised manuscript from Results/*.json and data/spectra.npz.
All figures: 300 dpi, real wavelength axis (nm), consistent stage names (Unripe, Ripe, Overripe)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "Results"))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
OUT = os.environ.get("FIG_DIR", os.path.join(HERE, "..", "Images", "revised"))
STAGES = ["Unripe", "Ripe", "Overripe"]
COL = {"Unripe": "#2e7d32", "Ripe": "#f9a825", "Overripe": "#8e2c1c"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 300, "font.family": "serif"})
# Approximate band centres from the NIR literature (Workman & Weyer 2012; Golic et al. 2003); used as guides only.
BANDS = [(970, "O–H 2nd overtone (~970 nm)"), (1200, "C–H 2nd overtone (~1200 nm)"), (1450, "O–H 1st overtone (~1450 nm)")]


def load(name):
    return json.load(open(os.path.join(RES, name)))


def savefig(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def fig_spectra():
    from scipy.signal import savgol_filter
    d = np.load(os.path.join(DATA, "spectra.npz"))
    wl, X, st = d["wl"], d["X_mean"], d["stage"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.05))
    for s in STAGES:
        R = X[st == s]
        P = savgol_filter((R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True), 15, 2, deriv=1, axis=1)
        for a, Y in [(ax[0], R), (ax[1], P)]:
            m, sd = Y.mean(0), Y.std(0)
            a.plot(wl, m, color=COL[s], lw=1.2, label=f"{s} (n = {len(R)} views)")
            a.fill_between(wl, m - sd, m + sd, color=COL[s], alpha=0.18, lw=0)
    for a in ax:
        for c, _ in BANDS:
            a.axvline(c, color="0.5", ls=":", lw=0.8)
        a.set_xlabel("Wavelength (nm)")
        a.set_xlim(wl[0], wl[-1])
        a.grid(alpha=0.25, lw=0.5)
    ax[0].set_ylabel("Reflectance (–)")
    ax[1].set_ylabel("SNV + SG 1st derivative (a.u.)")
    ax[0].set_title("(a) Mean ROI reflectance ± 1 SD")
    ax[1].set_title("(b) After preprocessing ± 1 SD")
    ax[0].legend(loc="lower left", frameon=False)
    for c, lab in BANDS:
        ax[0].text(c + 6, ax[0].get_ylim()[1] * 0.97, lab.split(" (")[0], rotation=90, va="top", fontsize=6, color="0.35")
    fig.tight_layout()
    savefig(fig, "fig_mean_spectra.png")


def fig_cv_and_confusion():
    M = load("main.json")
    order = ["Majority class", "PLS-DA", "Logistic regression", "SVM (RBF)", "Random forest", "XGBoost", "1D-CNN",
             "1D-ViT", "MT-ST"]
    fig, ax = plt.subplots(figsize=(6.6, 1.7))
    data = [[100 * f["acc"] for f in M["per_fold"][m]] for m in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, medianprops=dict(color="k", lw=1))
    for b, m in zip(bp["boxes"], order):
        b.set_facecolor("#9ecae1" if m != "MT-ST" else "#fdae6b")
    for i, dd in enumerate(data):
        ax.scatter(np.random.default_rng(i).normal(i + 1, 0.06, len(dd)), dd, s=4, color="0.3", zorder=3)
    ax.set_xticks(range(1, len(order) + 1), [o.replace(" (RBF)", "\n(RBF)").replace("Logistic regression", "Logistic\nregression").replace("Random forest", "Random\nforest").replace("Majority class", "Majority\nclass") for o in order])
    ax.set_ylabel("Fold accuracy (%)")
    P = M["protocol"]
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    savefig(fig, "fig_cv_accuracy.png")

    S = M["summary"]
    best = max([m for m in order if m not in ("MT-ST", "Majority class")], key=lambda m: S[m]["acc"][0])
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.35))
    for a, m in zip(ax, ["MT-ST", best]):
        C = np.array(S[m]["confusion_mean"])
        a.imshow(C / C.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{C[i, j]:.1f}", ha="center", va="center", color="w" if C[i, j] / C[i].sum() > 0.6 else "k")
        a.set_xticks(range(3), STAGES)
        a.set_yticks(range(3), [f"{s}\n(n={int(round(C[i].sum()))})" for i, s in enumerate(STAGES)])
        a.set_xlabel("Predicted stage")
        a.set_ylabel("True stage")
        a.set_title(f"{m}: acc. {100 * S[m]['acc'][0]:.1f} ± {100 * S[m]['acc'][1]:.1f}%")
    fig.tight_layout()
    savefig(fig, "fig_confusion.png")


def fig_augment():
    A = load("augment.json")
    sig = [(float(k.split("=")[1].split(",")[0]), v) for k, v in A.items() if k.endswith("p=0.5")]
    sig.sort()
    x = np.array([s for s, _ in sig])
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for key, lab, c in [("acc", "Accuracy", "#3182bd"), ("f1", "Macro-F1", "#e6550d")]:
        m = np.array([100 * v["summary"][key][0] for _, v in sig])
        s = np.array([100 * v["summary"][key][1] for _, v in sig])
        ax.errorbar(range(len(x)), m, yerr=s, marker="o", ms=3, capsize=2, lw=1, color=c, label=lab)
    ax.set_xticks(range(len(x)), [f"{v:g}" for v in x])
    ax.set_xlabel(r"Noise SD $\sigma$ (z-scored units; $p$ = 0.5)")
    ax.set_ylabel("MT-ST score (%)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "fig_augment.png")


def fig_xai():
    P = load("xai_profiles.json")
    wl = np.array(P["wl"])
    fig, ax = plt.subplots(figsize=(6.6, 1.85))
    for key, lab, c in [("ig_mean", "|IG|, mean-spectrum baseline", "#e6550d"),
                        ("ig_flat", "|IG|, featureless baseline", "#756bb1"),
                        ("att", "Attention rollout (CLS→patch)", "#3182bd")]:
        A = np.array([p[key] for p in P["profiles"]])
        A = A / A.max(1, keepdims=True)
        m, s = A.mean(0), A.std(0)
        ax.plot(wl, m, color=c, lw=1.1, label=lab)
        ax.fill_between(wl, m - s, m + s, color=c, alpha=0.18, lw=0)
    for cc, _ in BANDS:
        ax.axvline(cc, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalised importance (–)")
    ax.set_xlim(wl[0], wl[-1])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    savefig(fig, "fig_xai.png")


def fig_regression():
    """Out-of-fold predictions of multi-task MT-ST on the tabular data set (first repetition) for both targets."""
    C = load("csv_main.json")
    name = "MT-ST (multi-task, lambda_cls=0.5)"
    Y, P = np.array(C["yr"]), np.array(C["reg_preds"][name], dtype=float)[0]
    S = C["summary"][name]
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 2.25))
    for a, j, t, lab in [(ax[0], 0, "sugar", "Sugar (%)"), (ax[1], 1, "carbohydrate", "Carbohydrate (%)")]:
        a.scatter(Y[:, j], P[:, j], s=6, alpha=0.5, color="#3182bd", edgecolor="none")
        lo, hi = min(Y[:, j].min(), P[:, j].min()), max(Y[:, j].max(), P[:, j].max())
        a.plot([lo, hi], [lo, hi], "--", color="#e6550d", lw=1, label="$y=x$")
        a.set_xlabel(f"Reference {lab.lower()}")
        a.set_ylabel(f"Predicted {lab.lower()}")
        a.set_title(f"RMSE {S[t]['rmse'][0]:.2f}, $R^2$ = {S[t]['r2'][0]:.2f}")
        a.grid(alpha=0.25, lw=0.5)
        a.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig_regression.png")


if __name__ == "__main__":
    fig_spectra()
    for f in (fig_cv_and_confusion, fig_augment, fig_xai, fig_regression):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, "-", e)
    print("figures written to", os.path.abspath(OUT))
