#!/usr/bin/env python3
import os
import pydicom
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from diffct.differentiable import FanProjectorFunction
from config import DATASET_PATH, CLEAN_SINOGRAM_ROOT, MAX_SAMPLES

# ---------------------------------------------------------
# FIXED SYNTHETIC SCANNER GEOMETRY (MUST MATCH ROUND TRIP)
# ---------------------------------------------------------
SID = 560.0
SDD = 1100.0
DET_SPACING = 1.2
DET_COUNT = 736
VOXEL_SPACING = 1.0   # MUST remain 1.0
MU_WATER = 0.019
# ---------------------------------------------------------

# ---------------------------------------------------------
# Load DICOM → Extract HU only (not pixel spacing)
# ---------------------------------------------------------
def dicom_to_hu(path: Path):
    ds = pydicom.dcmread(str(path), force=True)

    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = img * slope + inter

    hu = np.clip(hu, -1024, 3071)
    return hu, ds


# ---------------------------------------------------------
# HU → Sinogram (matching reconstruction geometry)
# ---------------------------------------------------------
@torch.no_grad()
def hu_to_sinogram_fanbeam(
    hu,
    angles,
):
    device = angles.device

    # convert HU → μ
    mu_img = MU_WATER * (1.0 + (hu / 1000.0))
    phantom = torch.from_numpy(mu_img.astype(np.float32)).to(device)

    sino = FanProjectorFunction.apply(
        phantom,
        angles,
        int(DET_COUNT),
        float(DET_SPACING),
        float(SDD),
        float(SID),
        float(VOXEL_SPACING)  # fixed voxel spacing
    )

    if sino.ndim == 3:
        sino = sino[:, sino.shape[1] // 2, :].contiguous()

    return sino.cpu().numpy()


# ---------------------------------------------------------
# Convert dataset
# ---------------------------------------------------------
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    angles = torch.linspace(0, 2*np.pi, 720, device=device)[:-1]
    angles = angles.to(device)

    CLEAN_SINOGRAM_ROOT.mkdir(parents=True, exist_ok=True)
    processed = 0

    for root, _, files in os.walk(DATASET_PATH):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue

        out_dir = CLEAN_SINOGRAM_ROOT / Path(root).relative_to(DATASET_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)

        for filename in tqdm(dcm_files, desc=f"Converting ({processed}/{MAX_SAMPLES})"):

            if processed >= MAX_SAMPLES:
                break

            dcm_path = Path(root) / filename
            out_path = out_dir / (filename.replace(".dcm", ".npy"))

            if out_path.exists():
                continue

            hu, ds = dicom_to_hu(dcm_path)

            # generate sinogram
            sino = hu_to_sinogram_fanbeam(hu, angles).astype(np.float32)

            # save
            np.save(out_path, sino)
            processed += 1

    print(f"\n✅ Wrote {processed} sinograms to {CLEAN_SINOGRAM_ROOT}")


if __name__ == "__main__":
    main()
