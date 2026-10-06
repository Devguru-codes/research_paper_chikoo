"""Pixel-patch spectra from the UNLABELLED acquisition sessions (15/23/31 March 2022 and 'New folder'), used only
for self-supervised pretraining of MT-ST. These fruits are different from the labelled 8 April 2022 fruits, so no
labelled/test fruit is seen during pretraining. Uses its own RNG so spectra.npz is unchanged.
Output: data/unlabelled_pixels.npz with X (n, 160 raw reflectance bands), view (source fruit view)."""
import os

import numpy as np
from scipy import ndimage
from spectral import envi

import extract_spectra as E

N_PER_VIEW, PATCH = 300, 3
OUT = E.OUT


def main():
    rng = np.random.default_rng(12345)
    d = np.load(os.path.join(OUT, "spectra.npz"))
    cubes = sorted(set(d["cube"][d["stage"] == "Unlabelled"]))
    X, src = [], []
    for rel in cubes:
        f = os.path.join(E.ROOT, rel)
        img = envi.open(f + ".hdr", f)
        cube = np.asarray(img.load(), dtype=np.float32)
        wl = np.array([float(w) for w in img.metadata["wavelength"]])
        keep = np.where((wl >= E.WL_MIN) & (wl <= E.WL_MAX))[0][:160]
        for pos, (_, _, m) in enumerate(E.segment(cube)):
            ys, xs = np.nonzero(ndimage.binary_erosion(m, iterations=1))
            for i in rng.choice(len(ys), size=N_PER_VIEW, replace=len(ys) < N_PER_VIEW):
                X.append(cube[ys[i] - 1:ys[i] + 2, xs[i] - 1:xs[i] + 2][:, :, keep].reshape(-1, len(keep)).mean(0))
                src.append(f"{rel}#{pos}")
    np.savez_compressed(os.path.join(OUT, "unlabelled_pixels.npz"), X=np.array(X), view=np.array(src))
    print(f"{len(X)} unlabelled 3x3-pixel spectra from {len(set(src))} fruit views in {len(cubes)} cubes")


if __name__ == "__main__":
    main()
