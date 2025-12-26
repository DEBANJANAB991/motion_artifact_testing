<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import ARTIFACT_SINOGRAM_2D_TEST, PREDICTED_SINOGRAM_2D_TEST, CLEAN_SINOGRAM_2D_TEST
# -------------------------
# CONFIG
# -------------------------
ART_DIR  = ARTIFACT_SINOGRAM_2D_TEST
PRED_DIR = PREDICTED_SINOGRAM_2D_TEST
CLEAN_DIR= CLEAN_SINOGRAM_2D_TEST

OUT_DIR = Path("sino_mr_lkv_preview")
OUT_DIR.mkdir(exist_ok=True)

NUM_SAMPLES = 10   # change if you want more

# -------------------------
# load filenames
# -------------------------
files = sorted(ART_DIR.glob("*.npy"))[:NUM_SAMPLES]
assert len(files) > 0, "No sinograms found"

print(f"Visualising {len(files)} sinograms")

# -------------------------
# helper
# -------------------------
def load_sino(folder, fname):
    return np.load(folder / fname).astype(np.float32)

# -------------------------
# main loop
# -------------------------
for i, f in enumerate(files):
    name = f.name

    sino_art  = load_sino(ART_DIR, name)
    sino_pred = load_sino(PRED_DIR, name)
    sino_gt   = load_sino(CLEAN_DIR, name)

    # squeeze safety
    sino_art  = np.squeeze(sino_art)
    sino_pred = np.squeeze(sino_pred)
    sino_gt   = np.squeeze(sino_gt)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].imshow(sino_art, cmap="gray", aspect="auto")
    axs[0].set_title("Artifacted")

    axs[1].imshow(sino_pred, cmap="gray", aspect="auto")
    axs[1].set_title("MR_LKV Output")

    axs[2].imshow(sino_gt, cmap="gray", aspect="auto")
    axs[2].set_title("Clean (GT)")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / f"compare_{i:03d}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved: {out_path}")

print("DONE.")
=======
import os
import numpy as np
import matplotlib.pyplot as plt
from config import CLEAN_SINOGRAM_ROOT

# -------------------------
# CONFIG
# -------------------------
NUM_FILES = 10
OUTPUT_FOLDER = "sinogram_preview_2d"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SINO_DIR = CLEAN_SINOGRAM_ROOT
OUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_FOLDER)
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------
# 1D Ramp Filter
# -----------------------------------------------------
def ramp_filter_sinogram(sino):
    """
    sino: (views, det_u, det_v)
    Filter is applied along det_u (axis=1) for each v column.
    """

    views, det_u, det_v = sino.shape
    sino_f = np.zeros_like(sino, dtype=np.float32)

    # frequency axis
    freqs = np.fft.fftfreq(det_u)
    ramp = np.abs(freqs)

    for i in range(views):
        # FFT along detector-u
        F = np.fft.fft(sino[i], axis=0)
        F_filtered = F * ramp[:, None]   # apply ramp along u
        sino_f[i] = np.real(np.fft.ifft(F_filtered, axis=0))

    return sino_f


# -----------------------------------------------------
# VISUALIZATION FUNCTION (NOW WITH FILTERED SINO)
# -----------------------------------------------------
def save_sinogram_plots(sino, filename):
    num_views, det_u, det_v = sino.shape
    mid_v = det_v // 2
    mid_u = det_u // 2

    # Compute filtered sinogram
    sino_filtered = ramp_filter_sinogram(sino)

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))

    # 1) Central detector-row sinogram
    axs[0].imshow(sino[:, :, mid_v].T, cmap='gray', aspect='auto')
    axs[0].set_title("Central detector-row\n(clean)")
    axs[0].set_xlabel("Projection index")
    axs[0].set_ylabel("Detector u")

    # 2) Central detector-column sinogram
    axs[1].imshow(sino[:, mid_u, :].T, cmap='gray', aspect='auto')
    axs[1].set_title("Central detector-column\n(clean)")
    axs[1].set_xlabel("Projection index")
    axs[1].set_ylabel("Detector v")

    # 3) One projection (clean view)
    axs[2].imshow(sino[num_views // 2], cmap='gray')
    axs[2].set_title("One mid-angle projection\n(clean)")
    axs[2].set_xlabel("Detector u")
    axs[2].set_ylabel("Detector v")

    # 4) Filtered sinogram row
    axs[3].imshow(sino_filtered[:, :, mid_v].T, cmap='gray', aspect='auto')
    axs[3].set_title("Filtered sinogram (ramp)\ncentral detector-row")
    axs[3].set_xlabel("Projection index")
    axs[3].set_ylabel("Detector u")

    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, f"{filename}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved: {save_path}")


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
def main():
    if not os.path.isdir(SINO_DIR):
        print(f"Sinogram directory not found: {SINO_DIR}")
        return

    files = [f for f in os.listdir(SINO_DIR) if f.endswith(".npy")]
    files = sorted(files)[:NUM_FILES]

    if not files:
        print("No .npy files found.")
        return

    print(f"Processing {len(files)} sinograms...")

    for f in files:
        path = os.path.join(SINO_DIR, f)
        print(f"Loading: {path}")

        sino = np.load(path)
        print(f"Shape: {sino.shape}")

        file_id = os.path.splitext(f)[0]
        save_sinogram_plots(sino, file_id)

    print(f"\nAll plots saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
>>>>>>> 614225be32c37b765549a79cc466658ba9aae45f
