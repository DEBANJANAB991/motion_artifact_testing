#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import torch.nn.functional as F
# network definition
from model_mr_lkv import MR_LKV
# user config
from config import (
    CLEAN_SINOGRAM_ROOT,
    ARTIFACT_ROOT,
    CKPT_DIR,
    BATCH_SIZE,
    LR,
    EPOCHS,
    SAVE_INTERVAL,
)

torch.manual_seed(42)



# ---- PSNR (same as before) ----
def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target, reduction='mean')
    return 10 * torch.log10(max_val**2 / mse)

# ---- Simple SSIM implementation ----
def gaussian(window_size: int, sigma: float):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size//2
    g = torch.exp(-(coords**2) / (2*sigma**2))
    return g / g.sum()

def create_window(window_size: int, channel: int):
    _1d = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2d = _1d @ _1d.t()
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(pred: torch.Tensor, target: torch.Tensor, 
         window_size: int = 11, K=(0.01, 0.03), L: float = 1.0):
    # pred & target: (N, C, H, W), in [0, L]
    C1 = (K[0]*L)**2
    C2 = (K[1]*L)**2
    channel = pred.size(1)
    window = create_window(window_size, channel).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2

    sigma1_sq = F.conv2d(pred*pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target*target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(pred*target, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# --- Split fractions: 80% train, 10% val, 10% test ---
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
TEST_FRAC  = 1.0 - TRAIN_FRAC - VAL_FRAC  # = 0.1


class SinogramDataset(Dataset):
    # (unchanged from your original)
    def __init__(self, clean_root: Path, art_root: Path):
        self.clean_root = Path(clean_root)
        self.art_root   = Path(art_root)
        self.clean_paths = sorted(self.clean_root.rglob("*.npy"))

        self.art_paths = [
            self.art_root / p.relative_to(self.clean_root)
            for p in self.clean_paths
        ]
        if not all(p.exists() for p in self.art_paths):
            art_map = {p.name: p for p in self.art_root.rglob("*.npy")}
            self.art_paths = [art_map.get(p.name) for p in self.clean_paths]
            missing = [p for p in self.art_paths if p is None]
            if missing:
                raise RuntimeError(f"Missing artifact files for: {missing}")

        if not self.clean_paths:
            raise RuntimeError(f"No .npy files found under {clean_root}")

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_arr = np.load(self.clean_paths[idx]).astype(np.float32)
        art_arr   = np.load(self.art_paths[idx]).astype(np.float32)
        clean_arr = (clean_arr - clean_arr.min()) / (clean_arr.max() - clean_arr.min() + 1e-8)
        art_arr   = (art_arr   - art_arr.min())   / (art_arr.max()   - art_arr.min()   + 1e-8)
        clean = torch.from_numpy(clean_arr)[None]
        art   = torch.from_numpy(art_arr)[None]
        return art, clean


def parse_args():
    p = argparse.ArgumentParser(description="Train MR-LKV for artifact reduction")
    p.add_argument("--clean-root",   type=Path,
                   default=Path(CLEAN_SINOGRAM_ROOT))
    p.add_argument("--art-root",     type=Path,
                   default=Path(ARTIFACT_ROOT))
    p.add_argument("--ckpt-dir",     type=Path,
                   default=Path(CKPT_DIR))
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--epochs",       type=int, default=EPOCHS)
    p.add_argument("--save-interval",type=int, default=SAVE_INTERVAL)
    return p.parse_args()


def main():
    print("🚀 Starting training…", flush=True)
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) prepare dataset and splits
    dataset = SinogramDataset(args.clean_root, args.art_root)
    total   = len(dataset)
    n_train = int(TRAIN_FRAC * total)
    n_val   = int(VAL_FRAC   * total)
    n_test  = total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

   # train_loader = DataLoader(train_ds, batch_size=args.batch_size,
    #                          shuffle=True,  num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42), num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # 2) model, optimizer with weight decay, LR scheduler, loss
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = MR_LKV().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5           # <-- weight decay for regularization
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,                 # halve LR on plateau
        patience=5
    )
    criterion = nn.MSELoss(reduction='mean')

    best_val      = float('inf')
    epochs_no_imp = 0
    EARLY_STOP_PATIENCE = 10      # stop after 10 bad epochs

    # 3) training loop with scheduler & early stopping
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train = 0.0
        for art, clean in train_loader:
            art, clean = art.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(art)
            if pred.shape[-2:] != clean.shape[-2:]:
                pred = F.interpolate(
                    pred,
                    size=clean.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()
            running_train += loss.item() * art.size(0)

        train_loss = running_train / n_train

        model.eval()
        running_val = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        with torch.no_grad():
            for art, clean in val_loader:
                art, clean = art.to(device), clean.to(device)
                pred = model(art)
                if pred.shape[-2:] != clean.shape[-2:]:
                    pred = F.interpolate(
                        pred,
                        size=clean.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                running_val += criterion(pred, clean).item() * art.size(0)
                running_psnr += psnr(pred, clean).item() * art.size(0)
                running_ssim += ssim(pred, clean).item() * art.size(0)
        val_loss = running_val / n_val
        val_psnr = running_psnr / n_val
        val_ssim = running_ssim / n_val
        print(f"Epoch {epoch}/{args.epochs} — "       f"train: {train_loss:.6f}, val: {val_loss:.6f} | "       f"PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}",       flush=True)

        # step the LR scheduler on validation loss
        scheduler.step(val_loss)
        # --- manual LR logging ---
        current_lr = scheduler.get_last_lr()[0]
        print(f"⇢ learning rate is now {current_lr:.6e}")

        # early stopping & best‐val checkpoint
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_imp = 0
            torch.save(model.state_dict(), args.ckpt_dir / "best_model.pth")
            print(f"🔖 New best model saved (val {best_val:.6f})")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= EARLY_STOP_PATIENCE:
                print(f"✋ Early stopping at epoch {epoch} (no improvement in {EARLY_STOP_PATIENCE} epochs)")
                break

        # periodic checkpointing
        if epoch % args.save_interval == 0:
            pt = args.ckpt_dir / f"epoch{epoch}.pth"
            torch.save(model.state_dict(), pt)
            print(f"🔖 Saved checkpoint: {pt}")

    # 4) final test evaluation
    model.eval()
    running_test = 0.0
    with torch.no_grad():
        for art, clean in test_loader:
            art, clean = art.to(device), clean.to(device)
            running_test += criterion(model(art), clean).item() * art.size(0)

    test_loss = running_test / max(n_test, 1)
    print(f"Final Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()
