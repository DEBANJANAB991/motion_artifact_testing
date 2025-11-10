import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/hpc/iwi5/iwi5293h/Debanjana_Master_Thesis/diffct")
import torch


from diffct.differentiable import FanBackprojectorFunction
# ==============================
# Geometry settings (must match projection)
# ==============================
N_VIEWS = 360
DET_SPACING = 1.0
H, W = 256, 256
SRC_DET_PIXELS = 1000.0
SRC_ISO_PIXELS = 500.0
VOXEL_SPACING = 1.0

# ==============================
# Paths
# ==============================
repo_root = Path(__file__).resolve().parents[0]
plots_root = repo_root / "results" / "plots"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



#backprojector = FanBackprojectorFunction.apply(sino, angles, DET_SPACING, H,W,SRC_DET_PIXELS, SRC_ISO_PIXELS, VOXEL_SPACING)
# ==============================
# Function to reconstruct one model
# ==============================
def reconstruct_model(model_path: Path):
    sino_dir = model_path / "recon_sino"
    if not sino_dir.exists():
        print(f" Skipping {model_path.name}: no recon_sino folder found.")
        return

    output_dir = model_path / "reconstructed_ct"
    output_dir.mkdir(parents=True, exist_ok=True)

    sino_files = sorted(sino_dir.glob("*.png"))
    if not sino_files:
        print(f" No sinograms found in {sino_dir}.")
        return

    print(f"\n Reconstructing {len(sino_files)} sinograms for model: {model_path.name}")
    angles = torch.linspace(0, 2*np.pi, N_VIEWS,
                             device=device, dtype=torch.float32)
    for i, sino_path in enumerate(sino_files, 1):
        try:
            # Load PNG sinogram
            sino_img = cv2.imread(str(sino_path), cv2.IMREAD_UNCHANGED)
            if sino_img is None:
                raise ValueError(f"Failed to load image: {sino_path}")
            # Convert to grayscale if needed
            if sino_img.ndim == 3:
                sino_img = cv2.cvtColor(sino_img, cv2.COLOR_BGR2GRAY)
            sino = sino_img.astype(np.float32) / 255.0
            # Ensure 2D shape
            if sino.ndim != 2:
                raise ValueError(f"Unexpected sinogram shape {sino.shape} in {sino_path.name}")
            # Convert to tensor and move to device
            sino_t = torch.tensor(sino, dtype=torch.float32, device=device)
            print(f"\n--- Debug info for {sino_path.name} ---")
            print(f"sino type: {type(sino)}")
            print(f"sino shape: {sino.shape}")
            print(f"sino min/max: {sino.min():.3f}/{sino.max():.3f}")

            # Backprojection using FanBackprojectorFunction
            recon = FanBackprojectorFunction.apply(sino_t,angles,DET_SPACING,H,W,SRC_DET_PIXELS,SRC_ISO_PIXELS,VOXEL_SPACING)
            # Move to CPU and save image
            recon = recon.squeeze(0).cpu().numpy()

            # Save reconstructed image
            output_path = output_dir / f"{sino_path.stem}_recon.png"
            plt.imsave(output_path, recon, cmap="gray")

            print(f"[{i}/{len(sino_files)}] {model_path.name}: {sino_path.name} → {output_path.name}")

        except Exception as e:
            print(f" Error reconstructing {sino_path.name}: {e}")

    print(f" Done: Reconstructed CTs saved in {output_dir}\n")



# ==============================
# Main
# ==============================
if __name__ == "__main__":
    model_dirs = [d for d in plots_root.iterdir() if d.is_dir()]

    if not model_dirs:
        print("No model directories found inside results/plots/")
        exit()

    print(f"Found {len(model_dirs)} models: {[d.name for d in model_dirs]}")

    for model_dir in model_dirs:
        reconstruct_model(model_dir)

    print("All models processed successfully!")
