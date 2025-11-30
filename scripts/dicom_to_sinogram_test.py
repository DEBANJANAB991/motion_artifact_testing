import os
import pydicom
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from pydicom.uid import ExplicitVRLittleEndian
from diffct.differentiable import FanProjectorFunction
from config import DATASET_PATH, CLEAN_SINOGRAM_ROOT, SUBSET_ROOT, MAX_SAMPLES_TEST, DET_COUNT_FACTOR


# ------------------------------------------
# Load DICOM and convert to HU
# ------------------------------------------
def dicom_to_hu_and_geometry(path: Path):
    ds = pydicom.dcmread(str(path), force=True)

    # -------------------------------
    # FIX 1: Ensure Transfer Syntax exists
    # -------------------------------
    if "TransferSyntaxUID" not in ds.file_meta:
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # -------------------------------
    # FIX 2: Only process CT slices
    # -------------------------------
    if getattr(ds, "Modality", None) != "CT":
        raise ValueError(f"Not a CT slice: {path}")

    # -------------------------------
    # Pixel data decoding
    # -------------------------------
    try:
        img = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Cannot decode pixel data: {path}\n{e}")

    # HU conversion
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = img * slope + inter

    # Geometry
    if "PixelSpacing" not in ds:
        raise ValueError("Missing PixelSpacing")

    pixel_size = float(ds.PixelSpacing[0])  # mm

    SDD = float(getattr(ds, "DistanceSourceToDetector", 946.746))
    SAD = float(getattr(ds, "DistanceSourceToPatient", 538.52))

    return hu.astype(np.float32), pixel_size, SDD, SAD


# ------------------------------------------
# HU -> sinogram
# ------------------------------------------
@torch.no_grad()
def hu_to_sinogram(hu: np.ndarray, angles: torch.Tensor, pixel_size, SDD, SAD):
    H, W = hu.shape
    device = angles.device

    n_detectors = max(int(W * DET_COUNT_FACTOR), W)
    det_spacing = pixel_size

    phantom = torch.from_numpy(hu).to(device=device, dtype=torch.float32)

    sino = FanProjectorFunction.apply(
        phantom,
        angles,
        int(n_detectors),
        float(det_spacing),
        float(SDD),
        float(SAD),
        float(pixel_size),
    )

    return sino.cpu().numpy()


# ------------------------------------------
# Main: create test subset
# ------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    angles = torch.linspace(0, 2*np.pi, 360, device=device)[:-1]

    SUBSET_ROOT.mkdir(parents=True, exist_ok=True)

    processed = 0

    for root, _, files in os.walk(DATASET_PATH):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue

        out_dir = SUBSET_ROOT / Path(root).relative_to(DATASET_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)

        for filename in tqdm(dcm_files, desc=f"Creating Test Subset ({processed}/{MAX_SAMPLES_TEST})"):
            if processed >= MAX_SAMPLES_TEST:
                break

            dcm_path = Path(root) / filename

            subset_out_path = out_dir / (filename.replace(".dcm", ".npy"))
            train_out_path = CLEAN_SINOGRAM_ROOT / Path(root).relative_to(DATASET_PATH) / (filename.replace(".dcm", ".npy"))

            # Skip if already in training sinograms
            if train_out_path.exists() or subset_out_path.exists():
                continue

            # Safely load HU
            try:
                hu, pixel_size, SDD, SAD = dicom_to_hu_and_geometry(dcm_path)
            except Exception as e:
                print(f"[SKIP] {dcm_path}: {e}")
                continue

            # Safely convert to sinogram
            try:
                sino = hu_to_sinogram(hu, angles, pixel_size, SDD, SAD)
            except Exception as e:
                print(f"[SKIP] Sinogram failure at {dcm_path}: {e}")
                continue

            np.save(subset_out_path, sino)
            processed += 1

    print(f"\n✅ Wrote {processed} test sinograms to {SUBSET_ROOT}")


if __name__ == "__main__":
    main()
