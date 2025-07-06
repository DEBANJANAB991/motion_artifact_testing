#!/usr/bin/env python3
"""
Batch-convert every DICOM slice under config.DATASET_PATH → fan-beam sinogram (NumPy .npy).

• Keeps the original folder structure
• Uses GPU FanProjectorFunction from DiffCT            (CUDA kernel)
• Output shape: (n_views, n_detectors)
"""

import os, sys, json
from pathlib import Path
import numpy as np
import pydicom
import torch
from tqdm import tqdm


from config import (DATASET_PATH, SINOGRAM_ROOT,
                    N_VIEWS, N_DET, DET_SPACING,
                    SRC_ISO_PIXELS, SRC_DET_PIXELS, STEP_SIZE)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffct.differentiable import FanProjectorFunction



def dicom_to_hu(path: Path) -> np.ndarray:
    try:
        ds = pydicom.dcmread(str(path), force=True)
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope",  1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        return img * slope + inter              # Hounsfield Units
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return None

@torch.no_grad()
def slice_to_sino(img: np.ndarray, angles: torch.Tensor) -> np.ndarray:
    """Forward-project one 2-D slice → sinogram (GPU)."""
    dev = angles.device
    phantom = torch.from_numpy(img).to(dev)
    sino = FanProjectorFunction.apply(
        phantom,                     # H×W tensor on GPU
        angles,
        N_DET, DET_SPACING,
        STEP_SIZE,
        SRC_DET_PIXELS, SRC_ISO_PIXELS
    )
    return sino.cpu().numpy()        # (views, det)


def main() -> None:
    device = torch.device("cuda")
    angles = torch.linspace(0, 2*np.pi, N_VIEWS, device=device, dtype=torch.float32)

    for root, _, files in os.walk(DATASET_PATH):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if not dcm_files:        # skip non-DICOM dirs
            continue

        out_dir = SINOGRAM_ROOT / Path(root).relative_to(DATASET_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in tqdm(dcm_files, desc=str(Path(root).relative_to(DATASET_PATH))):
            src = Path(root) / f
            dst = out_dir / (f.replace(".dcm", ".npy"))
            if dst.exists():
                continue

            img = dicom_to_hu(src)
            if img is None:
                continue
            sino = slice_to_sino(img, angles)
            np.save(dst, sino)

    print("✅  All sinograms saved to", SINOGRAM_ROOT)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not available - abort.")
    main()
