#!/usr/bin/env python3
import h5py, numpy as np
from pathlib import Path
from config import SINOGRAM_ROOT

OUT_H5_PATH = Path("/home/vault/iwi5/iwi5293h/clean_sinograms.h5")

with h5py.File(OUT_H5_PATH, "w", libver="latest") as f:
    for npy in sorted(SINOGRAM_ROOT.rglob("*.npy")):
        rel = npy.relative_to(SINOGRAM_ROOT).with_suffix("")   # e.g. patientX/studyY/sliceZ
        grp = f.require_group(str(rel.parent))             # creates /patientX/studyY
        grp.create_dataset(rel.name,
                           data=np.load(npy),
                           compression="lzf")
    f.flush()

print(f"✅ Bundled clean sinograms into {OUT_H5_PATH}"
