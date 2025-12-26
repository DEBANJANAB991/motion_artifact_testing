#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import fft, ifft, fftfreq
from config import MERGED_SINOGRAM_3D_TEST, TEST_ARTIFACT_ROOT
# ---------------------------------------------------------
# Ramp filter (1D, applied along detector axis)
# ---------------------------------------------------------
def ramp_filter_1d(sino_row):
    n = sino_row.shape[0]
    freqs = fftfreq(n)
    ramp = np.abs(freqs)

    sino_fft = fft(sino_row)
    filtered = np.real(ifft(sino_fft * ramp))
    return filtered


# ---------------------------------------------------------
# Visualize + SAVE one sinogram
# ---------------------------------------------------------
def save_sinogram_preview(sino_np, save_path, title_prefix="clean"):
    """
    sino_np shape: (num_views, det_u, det_v)
    """

    num_views, det_u, det_v = sino_np.shape
    mid_u = det_u // 2
    mid_v = det_v // 2
    mid_view = num_views // 2

    central_row = sino_np[:, :, mid_v]   # (views, u)
    central_col = sino_np[:, mid_u, :]   # (views, v)
    one_proj = sino_np[mid_view]

    # Ramp-filtered central row
    filtered_row = np.zeros_like(central_row)
    for i in range(num_views):
        filtered_row[i] = ramp_filter_1d(central_row[i])

    fig, axs = plt.subplots(1, 4, figsize=(20, 6))

    axs[0].imshow(central_row.T, cmap="gray", aspect="auto")
    axs[0].set_title(f"Central detector-row\n({title_prefix})")
    axs[0].set_xlabel("Projection index")
    axs[0].set_ylabel("Detector u")

    axs[1].imshow(central_col.T, cmap="gray", aspect="auto")
    axs[1].set_title(f"Central detector-column\n({title_prefix})")
    axs[1].set_xlabel("Projection index")
    axs[1].set_ylabel("Detector v")

    axs[2].imshow(one_proj, cmap="gray")
    axs[2].set_title(f"One mid-angle projection\n({title_prefix})")
    axs[2].set_xlabel("Detector u")
    axs[2].set_ylabel("Detector v")

    axs[3].imshow(filtered_row.T, cmap="gray", aspect="auto")
    axs[3].set_title("Filtered sinogram (ramp)\ncentral detector-row")
    axs[3].set_xlabel("Projection index")
    axs[3].set_ylabel("Detector u")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved → {save_path}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    CLEAN_DIR = MERGED_SINOGRAM_3D_TEST
    ART_DIR   = TEST_ARTIFACT_ROOT

    # Create output folder next to script
    SCRIPT_DIR = Path(__file__).resolve().parent
    OUT_DIR = SCRIPT_DIR / "sino_preview"
    OUT_DIR.mkdir(exist_ok=True)

    clean_files = sorted(CLEAN_DIR.glob("*.npy"))[:10]
    art_files   = sorted(ART_DIR.glob("*.npy"))[:10]

    for cfile, afile in zip(clean_files, art_files):
        base = cfile.stem

        clean_sino = np.load(cfile)
        art_sino   = np.load(afile)

        save_sinogram_preview(
            clean_sino,
            OUT_DIR / f"{base}_clean.png",
            title_prefix="clean"
        )

        save_sinogram_preview(
            art_sino,
            OUT_DIR / f"{base}_artifact.png",
            title_prefix="motion artifact"
        )


if __name__ == "__main__":
    main()
