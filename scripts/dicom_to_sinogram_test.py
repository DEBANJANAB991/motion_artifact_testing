#!/usr/bin/env python3
import os
import math
import numpy as np
import pydicom
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from diffct.differentiable import ConeProjectorFunction
from config import DATASET_PATH, CLEAN_SINOGRAM_ROOT, TEST_CLEAN_SINOGRAM


# -----------------------------------------------------------
# 1. LOAD DICOM SERIES (NO HU CONVERSION)
# -----------------------------------------------------------
def load_dicom_series(folder):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".dcm")
    ]

    # Safe sorting
    def sort_key(p):
        try:
            return pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber
        except:
            return 0

    files_sorted = sorted(files, key=sort_key)

    slices = []
    first_ds = None

    for f in files_sorted:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)

        if first_ds is None:
            first_ds = ds

        slices.append(arr)

    volume = np.stack(slices, axis=0)  # (Z, Y, X)
    return volume, first_ds


# -----------------------------------------------------------
# 3. VISUALIZE SINOGRAM
# -----------------------------------------------------------
def save_sino_preview(sino, out_png):

    num_views, U, V = sino.shape
    mid_u = U // 2
    mid_v = V // 2

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(sino[:, :, mid_v].T, cmap='gray', aspect='auto')
    axs[0].set_title("Central detector-row")

    axs[1].imshow(sino[:, mid_u, :].T, cmap='gray', aspect='auto')
    axs[1].set_title("Central detector-column")

    axs[2].imshow(sino[num_views//2], cmap='gray')
    axs[2].set_title("One projection")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()



# -----------------------------------------------------------
# NEW: 4. SELECT THE CORRECT CT SERIES
# -----------------------------------------------------------
def select_ct_series(patient_dir):
    """
    Choose exactly one series per patient:
      1) Prefer CT PLAIN THIN
      2) Else CT Plain
      3) Else skip
    """

    thin = []
    plain = []

    for root, _, files in os.walk(patient_dir):
        if not any(f.lower().endswith(".dcm") for f in files):
            continue

        folder = Path(root).name.lower()

        if "ct plain thin" in folder or "ct_plain_thin" in folder or "plain thin" in folder:
            thin.append(root)

        elif "ct plain" in folder or "ct_plain" in folder:
            plain.append(root)

    if len(thin) > 0:
        return sorted(thin)[0]
    if len(plain) > 0:
        return sorted(plain)[0]
    return None



def main():

    ROOT = Path(DATASET_PATH)
    OUT_DIR = Path(TEST_CLEAN_SINOGRAM)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Collect already-processed patients from CLEAN_SINOGRAM_ROOT
    # -----------------------------------------------------------
    existing_patients = set(
        p.stem for p in Path(CLEAN_SINOGRAM_ROOT).glob("*.npy")
    )
    print(f"Found {len(existing_patients)} existing sinograms. Skipping those patients.")

    # Geometry (UNCHANGED)
    det_u = 800
    det_v = 800
    du = dv = 1.0
    sid = 530
    sdd = 1095
    num_views = 540

    angles = torch.linspace(0, 2 * math.pi, num_views, device="cuda")

    # Find patient folders
    patients = sorted([
        p for p in ROOT.rglob("*")
        if p.is_dir() and p.name.startswith("CQ500CT")
    ])

    print(f"Found {len(patients)} total patients.")

    for p in tqdm(patients, desc="Processing unseen patients"):

        # -------------------------------------------------------
        # SKIP patients already used for training
        # -------------------------------------------------------
        if p.name in existing_patients:
            continue

        # -------------------------------------------------------
        # Select correct CT series (UNCHANGED)
        # -------------------------------------------------------
        series = select_ct_series(str(p))
        if series is None:
            print(f"Skipping {p.name}: No CT PLAIN THIN or CT Plain found.")
            continue

        # -------------------------------------------------------
        # Load DICOM volume (UNCHANGED)
        # -------------------------------------------------------
        vol_np, first_ds = load_dicom_series(series)

        vol_np = vol_np.astype(np.float32)
        vol_np = (vol_np - vol_np.min()) / (vol_np.max() - vol_np.min())

        Nz, Ny, Nx = vol_np.shape

        px, py = map(float, first_ds.PixelSpacing)
        th = float(getattr(first_ds, "SliceThickness", min(px, py)))
        voxel_spacing = min(px, py, th)

        vol_torch = torch.tensor(
            vol_np, dtype=torch.float32, device="cuda"
        ).contiguous()

        # -------------------------------------------------------
        # Forward projection (UNCHANGED)
        # -------------------------------------------------------
        sino = ConeProjectorFunction.apply(
            vol_torch,
            angles,
            det_u, det_v,
            du, dv,
            sdd, sid,
            voxel_spacing
        )

        sino_np = sino.detach().cpu().numpy()

        # -------------------------------------------------------
        # Save new unseen sinograms
        # -------------------------------------------------------
        np.save(OUT_DIR / f"{p.name}.npy", sino_np)
        save_sino_preview(sino_np, OUT_DIR / f"{p.name}.png")

    print("\nDONE — unseen sinograms generated & saved.")

# -----------------------------------------------------------
if __name__ == "__main__":
    main()
