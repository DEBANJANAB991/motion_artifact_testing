#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load CSV 
df = pd.read_csv("results/tables/metrics_from_logs.csv")

# 2. Compute efficiency 
if "psnr_per_Mparam" not in df.columns:
    df["psnr_per_Mparam"] = df["final_test_psnr"] / (df["params"] / 1e6)

# 3. Helper to plot & save
def bar_plot(col, ylabel, fname):
    plt.figure()
    plt.bar(df["model"], df[col])
    plt.title(f"{ylabel} per Model")
    plt.ylabel(ylabel)
    plt.yscale('log')

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"results/figures/{fname}")

# 4. Make plots
bar_plot("final_test_loss", "MSE Loss",            "loss_comparison.png")
bar_plot("final_test_psnr", "PSNR (dB)",           "psnr_comparison.png")
bar_plot("final_test_ssim", "SSIM",                "ssim_comparison.png")
bar_plot("params",             "Parameter Count",   "params_comparison.png")
bar_plot("psnr_per_Mparam",    "PSNR per MParam",   "efficiency_psnr.png")

print("Saved comparison charts to results/figures/")
