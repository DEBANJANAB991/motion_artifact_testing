import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import ARTIFACT_ROOT_2D, PREDICTED_SINOGRAM_2D_TEST_v2, CLEAN_SINOGRAM_2D
# -------------------------
# CONFIG
# -------------------------
ART_DIR  = ARTIFACT_ROOT_2D
PRED_DIR = PREDICTED_SINOGRAM_2D_TEST_v2
CLEAN_DIR= CLEAN_SINOGRAM_2D

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
  #  sino_pred = load_sino(PRED_DIR, name)
    sino_gt   = load_sino(CLEAN_DIR, name)

    # squeeze safety
    sino_art  = np.squeeze(sino_art)
  #  sino_pred = np.squeeze(sino_pred)
    sino_gt   = np.squeeze(sino_gt)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].imshow(sino_art, cmap="gray", aspect="auto")
    axs[0].set_title("Artifacted")

   # axs[1].imshow(sino_pred, cmap="gray", aspect="auto")
   # axs[1].set_title("MR_LKV Output")

    axs[1].imshow(sino_gt, cmap="gray", aspect="auto")
    axs[1].set_title("Clean (GT)")
    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT_DIR / f"compare_{i:03d}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved: {out_path}")

print("DONE.")
