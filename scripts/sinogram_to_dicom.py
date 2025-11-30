#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob

from diffct.differentiable import FanBackprojectorFunction
from MR_LKV_refactorv2 import MR_LKV  # your model class
from config import CKPT_DIR, SUBSET_ROOT, ARTIFACT_TEST_ROOT

# -------------------- Paths --------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(CKPT_DIR, "mr_lkv/best_model.pth")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_test")
GT_CT_DIR = os.path.join(OUTPUT_DIR, "ground_truth")
ART_CT_DIR = os.path.join(OUTPUT_DIR, "artifacted")
PRED_CT_DIR = os.path.join(OUTPUT_DIR, "reconstructed")
COMPARE_DIR = os.path.join(OUTPUT_DIR, "comparison")

for d in [GT_CT_DIR, ART_CT_DIR, PRED_CT_DIR, COMPARE_DIR]:
    os.makedirs(d, exist_ok=True)

# -------------------- FBP Geometry --------------------
SDD = 946.746
SAD = 538.52
px_spacing = 1.0
det_spacing = 1.0
H, W = 512, 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
angles = torch.linspace(0, 2*np.pi, 361)[:-1].float().to(device)

# -------------------- Functions --------------------
def ramp_hann_filter(sino_t):
    n_det = sino_t.shape[-1]
    freqs = torch.fft.rfftfreq(n_det, d=1.0, device=sino_t.device)
    ramp = torch.abs(freqs)
    f_max = freqs.max()
    hann = 0.5 * (1 + torch.cos(np.pi * freqs / f_max))
    filt = ramp * hann
    sino_fft = torch.fft.rfft(sino_t, dim=-1) * filt.unsqueeze(0)
    return torch.fft.irfft(sino_fft, n=n_det, dim=-1)

def fbp_reconstruct(sino_np, angles, det_spacing, SDD, SAD, H, W, px_spacing):
    sino_t = torch.from_numpy(sino_np).float().to(angles.device)
    sino_filt = ramp_hann_filter(sino_t)
    with torch.no_grad():
        recon = FanBackprojectorFunction.apply(
            sino_filt, angles, float(det_spacing), int(H), int(W), float(SDD), float(SAD), float(px_spacing)
        )
    return recon.cpu().numpy()

def apply_window(img, window_center=40, window_width=80):
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    img_windowed = np.clip(img, min_val, max_val)
    img_windowed = (img_windowed - min_val) / (max_val - min_val)
    return img_windowed

def save_ct(ct_img, path, window=True):
    if window:
        ct_img = apply_window(ct_img)
    plt.imshow(ct_img, cmap="gray")
    plt.axis("off")
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_three_panel(gt, art, pred, out_path):
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.imshow(apply_window(gt), cmap='gray'); plt.title("Ground Truth"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(apply_window(art), cmap='gray'); plt.title("Artifacted"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(apply_window(pred), cmap='gray'); plt.title("Model Output"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -------------------- Load Model --------------------
model = MR_LKV()
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model.to(device)
model.eval()

# -------------------- Processing --------------------
# Recursively find all .npy files
files_gt = sorted(glob.glob(os.path.join(SUBSET_ROOT, '**', '*.npy'), recursive=True))

for path_gt in files_gt:
    # Compute relative path for nested structure
    rel_path = os.path.relpath(path_gt, SUBSET_ROOT)
    path_art = os.path.join(ARTIFACT_TEST_ROOT, rel_path)
    os.makedirs(os.path.join(GT_CT_DIR, os.path.dirname(rel_path)), exist_ok=True)
    os.makedirs(os.path.join(ART_CT_DIR, os.path.dirname(rel_path)), exist_ok=True)
    os.makedirs(os.path.join(PRED_CT_DIR, os.path.dirname(rel_path)), exist_ok=True)
    os.makedirs(os.path.join(COMPARE_DIR, os.path.dirname(rel_path)), exist_ok=True)

    # Load sinograms
    sino_gt = np.load(path_gt)
    sino_art = np.load(path_art)

    # Model prediction
    sino_art_tensor = torch.from_numpy(sino_art).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        sino_pred_tensor = model(sino_art_tensor)
    sino_pred = sino_pred_tensor.squeeze().cpu().numpy()

    # Reconstruct CT images
    ct_gt = fbp_reconstruct(sino_gt, angles, det_spacing, SDD, SAD, H, W, px_spacing)
    ct_art = fbp_reconstruct(sino_art, angles, det_spacing, SDD, SAD, H, W, px_spacing)
    ct_pred = fbp_reconstruct(sino_pred, angles, det_spacing, SDD, SAD, H, W, px_spacing)

    # Save CT images
    save_ct(ct_gt, os.path.join(GT_CT_DIR, rel_path.replace(".npy","_ct.png")))
    save_ct(ct_art, os.path.join(ART_CT_DIR, rel_path.replace(".npy","_ct.png")))
    save_ct(ct_pred, os.path.join(PRED_CT_DIR, rel_path.replace(".npy","_ct.png")))

    # Save comparison
    compare_path = os.path.join(COMPARE_DIR, rel_path.replace(".npy","_comparison.png"))
    save_three_panel(ct_gt, ct_art, ct_pred, compare_path)

    print("Processed:", rel_path)

print("All CT images and comparisons saved successfully!")
