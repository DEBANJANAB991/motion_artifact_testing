import pandas as pd
import matplotlib.pyplot as plt

# Replace these placeholder values with your actual metrics
results = {
    'MR_LKV':   {'Loss': 0.000196, 'PSNR': 37.09, 'SSIM': 0.9827},
    'U-Net':    {'Loss': 0.000800, 'PSNR': 30.12, 'SSIM': 0.9500}, 
    'RepLKNet': {'Loss': 0.003978, 'PSNR': 24.01, 'SSIM': 0.8167},
}

# Create DataFrame
df = pd.DataFrame(results).T

# Print the comparison table
print("Model Comparison Overview:")
print(df)

# Plot PSNR comparison
plt.figure()
df['PSNR'].plot(kind='bar')
plt.title('Model Comparison: PSNR')
plt.ylabel('PSNR (dB)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("/results/figures/model_comparison_psnr.png")
plt.show()

# Plot SSIM comparison
plt.figure()
df['SSIM'].plot(kind='bar')
plt.title('Model Comparison: SSIM')
plt.ylabel('SSIM')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("/results/figures/model_comparison_ssim.png")
plt.show()
