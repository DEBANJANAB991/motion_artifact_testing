#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

from diffct.differentiable import ConeBackprojectorFunction
from config import (
    CLEAN_SINOGRAM_2D_TEST,
    ARTIFACT_SINOGRAM_2D_TEST,
    PREDICTED_SINOGRAM_2D_TEST,
)

# ------------------------------
# Geometry (MUST match projection)
# ------------------------------
DET_U = 800
DET_V = 800
DU = DV = 1.0
SID = 530
SDD = 1095
NUM_VIEWS = 540

device = "cuda"

angles = torch.linspace(0, 2 * math.pi, NUM_VIEWS, device=device)

# ------------------------------
# Utilities
# ------------------------------
def load_patient_views(folder, patient_id):
    files = sorted([p for p in Path(folder).glob(f"{patient_id}*")])
    assert len(files) == NUM_VIEWS, f"{patient_id}: expected 540 views, got {len(files)}"
    sino = np.stack([np.load(f) for f in files], axis=0)
    return torch.tensor(sino, dtype=torch.float32, device=device)

def reconstruct(sino_3d):
    return ConeBackprojectorFunction.apply(
        sino_3d,
        angles,
        DET_U, DET_V,
        DU, DV,
        SDD, SID,
        1.0
    ).detach().cpu().numpy()

# ------------------------------
# MAIN
# ------------------------------
def main():
    clean_root = Path(CLEAN_SINOGRAM_2D_TEST)
    patients = sorted({p.name.split()[0] for p in clean_root.glob("*.npy")})[:3]

    out_dir = Path("results/recon_preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid in patients:
        print(f"Reconstructing {pid}")

        sino_clean = load_patient_views(CLEAN_SINOGRAM_2D_TEST, pid)
        sino_art   = load_patient_views(ARTIFACT_SINOGRAM_2D_TEST, pid)
        sino_pred  = load_patient_views(PREDICTED_SINOGRAM_2D_TEST, pid)

        ct_clean = reconstruct(sino_clean)
        ct_art   = reconstruct(sino_art)
        ct_pred  = reconstruct(sino_pred)

        z = ct_clean.shape[0] // 2

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(ct_art[z], cmap="gray")
        axs[0].set_title("Artifact")

        axs[1].imshow(ct_pred[z], cmap="gray")
        axs[1].set_title("MR_LKV")

        axs[2].imshow(ct_clean[z], cmap="gray")
        axs[2].set_title("Clean")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"{pid}_reconstruction.png", dpi=200)
        plt.close()

    print("Done.")

if __name__ == "__main__":
    main()
