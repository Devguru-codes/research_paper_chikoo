"""Extract per-fruit reflectance spectra from the raw Pika NIR-320 hypercubes.

Each ENVI cube (spectral library applies the 'reflectance scale factor' on load,
so values are reflectance in [0, ~1]) shows a tray of 2-6 fruits on a dark
background. Fruits are segmented as connected components of a thresholded
~1100 nm band image, eroded to remove edge/curvature pixels.

Outputs (Experiments/data/spectra.npz):
  wl            wavelengths (nm) kept for modelling (900-1700 nm)
  X_mean        mean ROI spectrum per fruit-view          (n_views, n_bands)
  X_patch       random 5x5 patch means per fruit-view     (n_views, n_patch, n_bands)
  cube, view, tray, stage, fruit_id, date, n_pix          per-row metadata
Ripeness stage is taken from the acquisition folder name (Unripe/Ripe/Overripe);
cubes outside those folders are kept with stage='Unlabelled'.
"""
import glob
import json
import os
import re

import numpy as np
from scipy import ndimage
from spectral import envi

ROOT = os.environ.get("CUBE_ROOT", os.path.join(os.path.dirname(__file__), "..", "Dataset", "CHIKU-IMAGES", "CHIKU"))
OUT = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
WL_MIN, WL_MAX = 900.0, 1700.0
N_PATCH, PATCH = 20, 5
MIN_PIX = 400
rng = np.random.default_rng(0)


def tray_and_view(name):
    # A, AF, AS -> tray A, views top/flipped/side; C1F-2 -> tray C1, view F, crop 2
    m = re.match(r"^([A-Z]\d?)(F|S)?(?:-(\d))?$", name)
    return m.group(1), (m.group(2) or "T"), m.group(3)


def segment(cube):
    img = cube[:, :, 40:50].mean(2)
    thr = 0.3 * np.percentile(img, 99.5)
    mask = ndimage.binary_opening(img > thr, iterations=2)
    lab, n = ndimage.label(mask)
    comps = []
    for k in range(1, n + 1):
        m = ndimage.binary_erosion(lab == k, iterations=5)
        if m.sum() >= MIN_PIX:
            cy, cx = ndimage.center_of_mass(m)
            comps.append((cy, cx, m))
    # reading order: rows (by y, tolerance ~ 1/6 image height), then x
    tol = img.shape[0] / 6
    comps.sort(key=lambda c: (round(c[0] / tol), c[1]))
    return comps


def patches(cube, m):
    ys, xs = np.nonzero(ndimage.binary_erosion(m, iterations=PATCH // 2))
    if len(ys) == 0:
        ys, xs = np.nonzero(m)
    idx = rng.choice(len(ys), size=N_PATCH, replace=len(ys) < N_PATCH)
    h = PATCH // 2
    return np.stack([cube[ys[i] - h:ys[i] + h + 1, xs[i] - h:xs[i] + h + 1].reshape(-1, cube.shape[2]).mean(0)
                     for i in idx])


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, Xm, Xp, skipped = [], [], [], []
    wl_keep = None
    for hdr in sorted(glob.glob(os.path.join(ROOT, "**", "*.hdr"), recursive=True)):
        f = hdr[:-4]
        rel = os.path.relpath(f, ROOT).replace(os.sep, "/")
        if not os.path.exists(f):
            skipped.append(rel)
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        tray, view, crop = tray_and_view(name)
        if crop is not None:
            # C1-1..4 / C2F-1..4 are single-fruit crops of whole-tray cubes (e.g. New folder/C1.bil);
            # skip them to avoid counting the same fruit twice.
            skipped.append(rel + " (crop duplicate)")
            continue
        img = envi.open(hdr, f)
        cube = np.asarray(img.load(), dtype=np.float32)
        wl = np.array([float(w) for w in img.metadata["wavelength"]])
        keep = (wl >= WL_MIN) & (wl <= WL_MAX)
        wl_keep = wl[keep] if wl_keep is None else wl_keep
        parts = rel.split("/")
        date = parts[0] if len(parts) > 1 else "undated"
        stage = parts[1] if len(parts) > 2 and parts[1] in ("Unripe", "Ripe", "Overripe") else "Unlabelled"
        group = "/".join(parts[:-1])  # folder (date[/stage]) + tray letter identifies a physical tray
        for pos, (_, _, m) in enumerate(segment(cube)):
            Xm.append(cube[m][:, keep].mean(0))
            Xp.append(patches(cube, m)[:, keep])
            rows.append(dict(cube=rel, date=date, stage=stage, tray=f"{group}/{tray}", view=view,
                             pos=pos, fruit_id=f"{group}/{tray}#{pos}", n_pix=int(m.sum())))
    np.savez_compressed(os.path.join(OUT, "spectra.npz"), wl=wl_keep, X_mean=np.array(Xm), X_patch=np.array(Xp),
                        **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    with open(os.path.join(OUT, "extraction_log.json"), "w") as fh:
        json.dump(dict(n_fruit_views=len(rows), n_bands=int(len(wl_keep)), wl_first=float(wl_keep[0]),
                       wl_last=float(wl_keep[-1]), skipped=skipped, rows=rows), fh, indent=1)
    print(f"{len(rows)} fruit-views, {len(wl_keep)} bands {wl_keep[0]:.1f}-{wl_keep[-1]:.1f} nm; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
