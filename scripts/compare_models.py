#!/usr/bin/env python3
import sys
from pathlib import Path
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info

# ============================================================
# PATH SETUP
# ============================================================
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# ============================================================
# OUTPUT DIRECTORY (results/ next to this script)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# FINAL TEST METRICS (FROM .out LOGS – MANUAL)
# ============================================================
FINAL_METRICS = {
    "mr_lkv":     {"PSNR": 39.97, "SSIM": 0.9940},
    "unet":       {"PSNR": 37.11, "SSIM": 0.9815},
    "replknet":   {"PSNR": -12.42, "SSIM": 0.9034},
    "swinir":     {"PSNR": 38.98, "SSIM": 0.9813},
    "restormer":  {"PSNR": 11.76, "SSIM": 0.0233},
}

# ============================================================
# INPUT SIZES (MATCH TRAINING)
# ============================================================
MODEL_INPUTS = {
    "mr_lkv":     (1, 512, 512),   # full sinogram
    "unet":       (1, 512, 512),
    "replknet":   (1, 512, 512),
    "swinir":     (1, 128, 128),   # PATCH
    "restormer":  (1, 96, 96),     # PATCH
}

# ============================================================
# IMPORT TRAINING WRAPPERS (IMPORTANT)
# ============================================================
from train_test_split import (
    MR_LKV,
    UNet,
    RepLKNetReg,
    SwinIRWrapper,
    RestormerWrapper,
)

# ============================================================
# MODEL BUILDERS (EXACTLY AS TRAINING)
# ============================================================
MODELS = {
    "mr_lkv": lambda: MR_LKV(
        in_channels=1,
        base_channels=32,
        depths=[2, 2, 3, 2],
        kernels=[35, 55, 75, 95],
        norm_type="batch",
        use_decoder=True,
        final_activation=None,
    ),
    "unet": lambda: UNet(
        in_channels=1,
        base_channels=64,
        levels=4,
        norm_type="batch",
        dropout_bottleneck=0.1,
        final_activation=None,
    ),
    "replknet": lambda: RepLKNetReg(
        large_kernel_sizes=[31, 29, 27, 13],
        layers=[2, 2, 18, 2],
        channels=[64, 128, 256, 512],
        drop_path_rate=0.0,
        small_kernel=5,
        in_channels=1,
    ),
    "swinir": lambda: SwinIRWrapper(
        img_size=128,
        window_size=8,
        in_chans=1,
        out_chans=1,
        depths=[4, 4, 4, 4],
        embed_dim=64,
        num_heads=[2, 2, 2, 2],
        mlp_ratio=4,
        use_checkpoint=False,
    ),
    "restormer": lambda: RestormerWrapper(
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=[2, 2, 2, 4],
        num_refinement_blocks=2,
        heads=[1, 2, 2, 4],
        ffn_expansion_factor=2.0,
        bias=False,
        LayerNorm_type="WithBias",
    ),
}

# ============================================================
# MAIN
# ============================================================
def main():
    results = []

    for name, builder in MODELS.items():
        print(f"Evaluating {name}")
        model = builder().to(DEVICE).eval()

        C, H, W = MODEL_INPUTS[name]
        dummy = torch.randn(1, C, H, W, device=DEVICE)

        # ---------------- Params ----------------
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        # ---------------- FLOPs (MACs) ----------------
        macs, _ = get_model_complexity_info(
            model,
            (C, H, W),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        flops_gmac = macs / 1e9

        # ---------------- Inference Time ----------------
        with torch.no_grad():
            for _ in range(10):  # warm-up
                _ = model(dummy)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            for _ in range(50):
                _ = model(dummy)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        inf_ms = (time.time() - start) / 50 * 1000

        results.append({
            "model": name,
            "PSNR (dB)": FINAL_METRICS[name]["PSNR"],
            "SSIM": FINAL_METRICS[name]["SSIM"],
            "Params (M)": round(params_m, 2),
            "FLOPs (GMac)": round(flops_gmac, 2),
            "Inf (ms)": round(inf_ms, 2),
        })

    # ========================================================
    # SAVE RESULTS
    # ========================================================
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    df.to_json(RESULTS_DIR / "model_comparison.json", orient="records", indent=2)
    print("\nFINAL MODEL COMPARISON\n")
    print(df)

    # ========================================================
    # PLOTS
    # ========================================================
    models = df["model"].tolist()
    x = np.arange(len(models))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    axes[0].bar(x, df["PSNR (dB)"]); axes[0].set_title("PSNR (dB)")
    axes[1].bar(x, df["SSIM"]); axes[1].set_title("SSIM")
    axes[2].bar(x, df["Inf (ms)"]); axes[2].set_title("Inference Time (ms)")
    axes[3].bar(x, df["Params (M)"]); axes[3].set_title("Parameters (M)")
    axes[4].bar(x, df["FLOPs (GMac)"]); axes[4].set_title("FLOPs (×10⁹ MACs)")
    axes[5].axis("off")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison_plots.png", dpi=300)
    plt.show()

    print("Saved:")
    print(" - model_comparison.csv")
    print(" - model_comparison.json")
    print(" - model_comparison_plots.png")

# ============================================================
if __name__ == "__main__":
    main()
