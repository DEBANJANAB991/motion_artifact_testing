#!/usr/bin/env python3
import os
import pydicom
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from diffct.differentiable import FanProjectorFunction
from config import DATASET_PATH, CLEAN_SINOGRAM_ROOT, MAX_SAMPLES, DET_COUNT_FACTOR

# Physical constant (same as round-trip)
MU_WATER = 0.019  # 1/mm

# ---------------------------------------------------
# Load DICOM and extract HU + geometry
# ---------------------------------------------------
def dicom_to_hu_and_geometry(path: Path):
    ds = pydicom.dcmread(str(path), force=True)

    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = img * slope + inter
   # hu = np.clip(hu, -1000, 2000)
    hu = np.clip(hu, -1024, 3071)  # full CT dynamic range

    px = ds.get("PixelSpacing", None)
    if px is None:
        raise ValueError("Missing PixelSpacing")

    # IMPORTANT — maintain orientation consistency
    pixel_spacing_H = float(px[0])  # row spacing
    pixel_spacing_W = float(px[1])  # column spacing

    # Header geometry or fallback defaults
    SDD = float(getattr(ds, "DistanceSourceToDetector", 1085.6))  # mm
    SAD = float(getattr(ds, "DistanceSourceToPatient", 595.0))   # mm


    return hu.astype(np.float32), pixel_spacing_H, pixel_spacing_W, SDD, SAD


# ---------------------------------------------------
# Fan-beam forward projection (same as round-trip)
# ---------------------------------------------------
@torch.no_grad()
def hu_to_sinogram_fanbeam(hu, angles, pixel_spacing_H, pixel_spacing_W, SDD, SAD):
    H, W = hu.shape
    device = angles.device

    # SAME detector count logic as round-trip
    n_detectors = max(int(W * DET_COUNT_FACTOR), W)

    # SAME detector spacing as round-trip
    det_spacing = pixel_spacing_W  # use W spacing

    # HU → mu (same formula)
    mu_img = MU_WATER * (1.0 + (hu / 1000.0))

    phantom = torch.from_numpy(mu_img.astype(np.float32)).to(device)

    # FanProjectorFunction signature is identical to round-trip
    sino = FanProjectorFunction.apply(
        phantom,
        angles,
        int(n_detectors),
        float(det_spacing),
        float(SDD),
        float(SAD),
        float(pixel_spacing_H),  # row spacing here
    )

    # Return raw sinogram (no filtering here)
    return sino.cpu().numpy()


# ---------------------------------------------------
# Main processing loop
# ---------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SAME angle sampling as round-trip
    angles = torch.linspace(0, 2*np.pi, 360, device=device)[:-1]

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

            # Load HU + geometry
            hu, pixel_spacing_H, pixel_spacing_W, SDD, SAD = dicom_to_hu_and_geometry(dcm_path)

            # Fan-beam sinogram
            sino = hu_to_sinogram_fanbeam(
                hu,
                angles,
                pixel_spacing_H,
                pixel_spacing_W,
                SDD,
                SAD
            )

            np.save(out_path, sino)
            processed += 1

    print(f"\n✅ Wrote {processed} sinograms to {CLEAN_SINOGRAM_ROOT}")


if __name__ == "__main__":
    main()
