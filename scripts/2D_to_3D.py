#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import re
from collections import defaultdict

from config import PREDICTED_SINOGRAM_2D_TEST, MERGED_SINOGRAM_3D_TEST

IN_DIR  = PREDICTED_SINOGRAM_2D_TEST
OUT_DIR = MERGED_SINOGRAM_3D_TEST
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Example filename:
# CQ500CT215 CQ500CT215_view034.npy
pattern = re.compile(r"(CQ500CT\d+)\s+\1_view(\d+)\.npy")

groups = defaultdict(list)

for f in IN_DIR.glob("*.npy"):
    m = pattern.match(f.name)
    if m:
        patient_id = m.group(1)
        view_idx   = int(m.group(2))
        groups[patient_id].append((view_idx, f))
    else:
        print(f"Skipped (name mismatch): {f.name}")

print(f"Found {len(groups)} patients")

for patient_id, views in groups.items():
    views = sorted(views, key=lambda x: x[0])  # sort by view index
    sino_stack = [np.load(f) for _, f in views]

    sino_3d = np.stack(sino_stack, axis=0)  # (views, U, V)

    np.save(OUT_DIR / f"{patient_id}.npy", sino_3d)
    print(f"{patient_id}: saved sinogram with shape {sino_3d.shape}")

print("✅ Done: merged 2D views → 3D cone-beam sinograms")
