import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path


from train_test_split import (
    SinogramDataset,
    MR_LKV,
    TRAIN_FRAC,
    VAL_FRAC,
    CLEAN_SINOGRAM_ROOT,
    ARTIFACT_ROOT,
    CKPT_DIR,
    BATCH_SIZE,
)
from train_test_split import psnr, ssim 

def evaluate():
    print("Starting evaluation…", flush=True)

   
    dataset = SinogramDataset(Path(CLEAN_SINOGRAM_ROOT),
                              Path(ARTIFACT_ROOT))
    total   = len(dataset)
    n_train = int(TRAIN_FRAC * total)
    n_val   = int(VAL_FRAC   * total)
    n_test  = total - n_train - n_val

    _, _, test_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    test_loader = DataLoader(test_ds,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)

    # Loads the best model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MR_LKV().to(device)
    model.load_state_dict(torch.load(Path(CKPT_DIR) / "best_model.pth",
                                     map_location=device))
    model.eval()

    # Metrics accumulators
    total_loss = total_psnr = total_ssim = 0.0

    with torch.no_grad():
        for art, clean in test_loader:
            art, clean = art.to(device), clean.to(device)
            pred = model(art)
            if pred.shape != clean.shape:
                pred = F.interpolate(pred,
                                     size=clean.shape[-2:],
                                     mode='bilinear',
                                     align_corners=False)

            loss = F.mse_loss(pred, clean, reduction='mean')
            total_loss += loss.item() * art.size(0)
            total_psnr += psnr(pred, clean).item() * art.size(0)
            total_ssim += ssim(pred, clean).item() * art.size(0)

    avg_loss = total_loss / n_test
    avg_psnr = total_psnr / n_test
    avg_ssim = total_ssim / n_test

    print(f"── Test Results ─────────", flush=True)
    print(f"Loss: {avg_loss:.6f}",    flush=True)
    print(f"PSNR: {avg_psnr:.2f} dB",  flush=True)
    print(f"SSIM: {avg_ssim:.4f}",    flush=True)
    print(f"──────────────────────────", flush=True)



if __name__ == "__main__":
    evaluate()
