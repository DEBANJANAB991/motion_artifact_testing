import os
import numpy as np
from pathlib import Path
#from config import CLEAN_SINOGRAM_ROOT, ARTIFACT_ROOT, CLEAN_SINOGRAM_2D, ARTIFACT_SINOGRAM_2D
from config import TEST_CLEAN_SINOGRAM, TEST_ARTIFACT_ROOT, CLEAN_SINOGRAM_2D_TEST, ARTIFACT_SINOGRAM_2D_TEST
OUT_CLEAN =CLEAN_SINOGRAM_2D_TEST
OUT_ART   =ARTIFACT_SINOGRAM_2D_TEST
OUT_CLEAN.mkdir(exist_ok=True)
OUT_ART.mkdir(exist_ok=True)

clean_files = sorted([f for f in Path(TEST_CLEAN_SINOGRAM).glob("*.npy")])
artifact_files = sorted([f for f in Path(TEST_ARTIFACT_ROOT).glob("*.npy")])

# match pairs by filename stem
pairs = {}
for c in clean_files:
    stem = c.stem
    a = Path(TEST_ARTIFACT_ROOT) / f"{stem}_artifact.npy"
    if a.exists():
        pairs[stem] = (c, a)

print("Number of paired sinograms found:", len(pairs))

for stem, (clean_path, art_path) in pairs.items():
    clean = np.load(clean_path)        # shape (views, u, v)
    art   = np.load(art_path)

    assert clean.shape == art.shape
    num_views, H, W = clean.shape

    for v in range(num_views):
        clean_slice = clean[v]
        art_slice   = art[v]

        np.save(OUT_CLEAN / f"{stem}_view{v:03d}.npy", clean_slice)
        np.save(OUT_ART   / f"{stem}_view{v:03d}.npy", art_slice)

print("Done. All 2D slices saved.")
