import os
import numpy as np
import matplotlib.pyplot as plt
import random
from config import ARTIFACT_ROOT
# ----------------------------
# Root folder containing sinogram subfolders
# ----------------------------
sino_root = ARTIFACT_ROOT  # <-- change this to your folder

# Recursively find all .npy files
sino_files = []
for dirpath, _, files in os.walk(sino_root):
    for f in files:
        if f.endswith(".npy"):
            sino_files.append(os.path.join(dirpath, f))

n_files = len(sino_files)
if n_files == 0:
    raise ValueError(f"No .npy sinogram files found in {sino_root}")

# Show at most 3 sinograms
n_show = min(3, n_files)
sample_files = random.sample(sino_files, n_show)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(4*n_show, 4))
for i, fpath in enumerate(sample_files):
    sino = np.load(fpath)
    plt.subplot(1, n_show, i+1)
    plt.imshow(sino, cmap="gray", aspect="auto")
    plt.title(os.path.basename(fpath))
    plt.xlabel("Detector")
    plt.ylabel("View")

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "sample_sinograms.png")
plt.savefig(save_path, dpi=200)
plt.show()

print(f"Saved figure to: {save_path}")