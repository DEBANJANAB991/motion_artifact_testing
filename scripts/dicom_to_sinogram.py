import os
import pydicom
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from diffct.differentiable import FanProjectorFunction
from config import DATASET_PATH, CLEAN_SINOGRAM_ROOT, MAX_SAMPLES, DET_COUNT_FACTOR

# ------------------------------------------
# Load DICOM and convert to HU
# ------------------------------------------
def dicom_to_hu_and_geometry(path: Path):
    ds = pydicom.dcmread(str(path), force=True)

    img = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = img * slope + inter

    px = ds.get("PixelSpacing", None)
    if px is None:
        raise ValueError("Missing PixelSpacing")
    pixel_size = float(px[0])  # mm

    SDD = float(getattr(ds, "DistanceSourceToDetector", 946.746))  # mm
    SAD = float(getattr(ds, "DistanceSourceToPatient", 538.52))   # mm

    return hu.astype(np.float32), pixel_size, SDD, SAD


# ------------------------------------------
# Create sinogram using SAME parameters as working round-trip
# ------------------------------------------
@torch.no_grad()
def hu_to_sinogram(hu: np.ndarray, angles: torch.Tensor, pixel_size, SDD, SAD):
    H, W = hu.shape
    device = angles.device

    # detector count same as in working code
    n_detectors = max(int(W * DET_COUNT_FACTOR), W)
    det_spacing = pixel_size  # mm, consistent with round-trip

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
# Main processing loop
# ------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            hu, pixel_size, SDD, SAD = dicom_to_hu_and_geometry(dcm_path)

            # Generate sinogram
            sino = hu_to_sinogram(hu, angles, pixel_size, SDD, SAD)

            np.save(out_path, sino)
            processed += 1

    print(f"\n✅ Wrote {processed} sinograms to {CLEAN_SINOGRAM_ROOT}")


if __name__ == "__main__":
    main()
