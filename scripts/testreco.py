#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------
# Path to the sinogram directory
# ---------------------------------------------------------
from config import CLEAN_SINOGRAM_ROOT

# ---------------------------------------------------------
# Save sinogram PNGs in same directory as this script
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()

def show_and_save_sinogram(sino, out_file):
    plt.figure(figsize=(7, 5))
    plt.imshow(sino, cmap='gray', aspect='auto')
    plt.colorbar(label="Line Integral")
    plt.title("Sinogram Preview")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def main():
    # Collect all .npy sinograms
    sinograms = sorted(CLEAN_SINOGRAM_ROOT.rglob("*.npy"))

    if len(sinograms) == 0:
        print("❌ No .npy sinograms found!")
        return

    print(f"Found {len(sinograms)} sinograms. Saving first 5 PNGs...")

    for i, path in enumerate(sinograms[:5]):
        sino = np.load(path)
        sino = np.squeeze(sino)  # Remove extra dimension if present

        out_file = SCRIPT_DIR / f"sinogram_{i}.png"
        show_and_save_sinogram(sino, out_file)

        print(f"Saved: {out_file}")

    print("\n✅ Finished! PNGs saved next to this script.")

if __name__ == "__main__":
    main()
