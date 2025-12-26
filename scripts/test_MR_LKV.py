#!/usr/bin/env python3
"""
Test MR_LKV on ALL 2D sinograms (inference only).

Input:
  ARTIFACT_SINOGRAM_2D_TEST/*.npy

Output:
  PREDICTED_SINOGRAM_2D_TEST/*.npy
(same filenames preserved)
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from MR_LKV_refactorv2 import MR_LKV
from config import (
    ARTIFACT_SINOGRAM_2D_TEST,
    PREDICTED_SINOGRAM_2D_TEST,
    CKPT_DIR,
)

# -------------------------
# Configuration
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(CKPT_DIR) / "mr_lkv" / "best_model.pth"

IN_DIR  = Path(ARTIFACT_SINOGRAM_2D_TEST)
OUT_DIR = Path(PREDICTED_SINOGRAM_2D_TEST)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load model
# -------------------------
model = MR_LKV(
    in_channels=1,
    base_channels=24,          # MUST match training
    depths=[2, 2, 3, 2],
    kernels=[35, 55, 75, 95],
    norm_type="batch",
    use_decoder=True,
    final_activation=None,
).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()

print(f"Loaded MR_LKV from {MODEL_PATH}")
print(f"Running inference on device: {DEVICE}")

# -------------------------
# Inference loop
# -------------------------
files = sorted(IN_DIR.glob("*.npy"))
print(f"Found {len(files)} sinograms to process")

with torch.no_grad():
    for f in tqdm(files, desc="MR_LKV inference"):
        sino = np.load(f).astype(np.float32)

        # normalize per image
        s_min, s_max = sino.min(), sino.max()
        sino_n = (sino - s_min) / (s_max - s_min + 1e-12)

        # to tensor (1,1,H,W)
        x = torch.from_numpy(sino_n).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # predict
        y = model(x)

        # back to numpy
        y = y.squeeze().cpu().numpy()

        # de-normalize (optional but recommended)
        y = y * (s_max - s_min) + s_min

        # save
        np.save(OUT_DIR / f.name, y)

print("\n✅ MR_LKV inference completed.")
print(f"Predicted sinograms saved to:\n  {OUT_DIR}")
