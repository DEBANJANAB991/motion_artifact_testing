#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from skimage.filters import sobel

# ---------------- paths ----------------
plots_root = Path(__file__).resolve().parents[0] / "results" / "plots"
models = [d for d in plots_root.iterdir() if d.is_dir()]

# ---------------- helper ----------------
def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32) / 255.0  # assume images are 0-1 normalized

# ---------------- main ----------------
for model in models:
    gt_dir = model / "ground_truth" / "ct_images"
    art_dir = model / "artifacted" / "ct_images"
    recon_dir = model / "reconstructed" / "ct_images"
    comp_dir = model / "comparison"
    comp_dir.mkdir(exist_ok=True)

    # match files by stem
    gt_files = sorted(gt_dir.glob("*_ct.png"))
    art_files = sorted(art_dir.glob("*_ct.png"))
    recon_files = sorted(recon_dir.glob("*_ct.png"))

    for gt_path, art_path, recon_path in zip(gt_files, art_files, recon_files):
        gt = load_image(gt_path)
        art = load_image(art_path)
        recon = load_image(recon_path)

        # ---------------- Edge-based improvement ----------------
        # Compute edges using Sobel filter
        edges_art = sobel(art)
        edges_recon = sobel(recon)
        edge_diff = edges_art - edges_recon  # positive where edges reduced (streaks removed)
        edge_diff[edge_diff < 0] = 0  # ignore new edges

        # Amplify subtle differences for visibility
        factor = 10.0
        edge_diff_amp = np.clip(edge_diff * factor, 0, 1)

        # Convert reconstructed image to RGB
        recon_rgb = cv2.cvtColor((recon * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Apply heatmap
        heatmap = plt.cm.jet(edge_diff_amp)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        # Blend heatmap with reconstructed image
        alpha = 0.5
        overlay = cv2.addWeighted(heatmap, alpha, recon_rgb, 1 - alpha, 0)

        # ---------------- Plot 4 panels ----------------
        fig = plt.figure(figsize=(24, 5))
        spec = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.05)

        ax0 = fig.add_subplot(spec[0])
        ax1 = fig.add_subplot(spec[1])
        ax2 = fig.add_subplot(spec[2])
        ax3 = fig.add_subplot(spec[3])
        cax = fig.add_subplot(spec[4])

        ax0.imshow(gt, cmap='gray'); ax0.set_title("Ground Truth")
        ax1.imshow(art, cmap='gray'); ax1.set_title("Artifacted")
        ax2.imshow(recon, cmap='gray'); ax2.set_title("Reconstructed")
        im = ax3.imshow(overlay); ax3.set_title("Edge-based Improvement")

        # Colorbar for heatmap (0 = no improvement, 1 = max streak removal)
        norm = Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label("Edge Reduction Intensity", fontsize=10)

        for ax in [ax0, ax1, ax2, ax3]:
            ax.axis('off')

        plt.suptitle(gt_path.stem, fontsize=16)
        out_path = comp_dir / f"{gt_path.stem}_comparison_edge_overlay.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved 4-panel comparison with edge-based overlay: {out_path}")
