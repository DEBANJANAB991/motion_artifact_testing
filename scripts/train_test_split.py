#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid, save_image
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset

import numpy as np


repo_root = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(repo_root))

sys.path.insert(0, str(repo_root / "Restormer"))

sys.path.insert(0, str(repo_root / "swin2sr"))
import importlib
print("TRY:", repo_root)
print("SwinIR:", importlib.util.find_spec("SwinIR") is not None)
print("basicsr:", importlib.util.find_spec("basicsr") is not None)

from MR_LKV_refactorv2 import MR_LKV      # MR-LKV implementation
from UNet import UNet                     # U-Net baseline
from replknet import RepLKNet             # Original RepLKNet backbone
from models.network_swin2sr import Swin2SR


from SwinIR.models.network_swinir import SwinIR
# Restormer import (repo: swz30/Restormer)
try:
    from basicsr.models.archs.restormer_arch import Restormer as RestormerNet
except ModuleNotFoundError:
   
    archs_dir = repo_root / "Restormer" / "basicsr" / "models" / "archs"
    sys.path.insert(0, str(archs_dir))
    from restormer_arch import Restormer as RestormerNet

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

# ---------------- Metrics ---------------- #

def psnr(pred, target, max_val: float = 1.0):
    mse = F.mse_loss(pred, target, reduction='mean')
    return 10 * torch.log10(max_val**2 / mse)

def gaussian(window_size: int, sigma: float):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()

def create_window(window_size: int, channel: int):
    _1d = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2d = _1d @ _1d.t()
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, K=(0.01, 0.03), L: float = 1.0):
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    channel = pred.size(1)
    window = create_window(window_size, channel).to(pred.device)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
#defining L1 +SSIM Loss
class L1SSIMLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return (1 - self.alpha) * self.l1(pred, target) + self.alpha * (1 - ssim(pred, target))

#code to save reconstructed sinograms
def save_recon_sinos(recon_batch, epoch, recon_dir, max_imgs=3):
    """
    Save only reconstructed sinograms.
    """
    for i in range(min(max_imgs, recon_batch.size(0))):
        save_path = recon_dir / f"epoch{epoch:03d}_sino{i}.png"
        save_image(recon_batch[i], save_path)


# ---------------- Data ---------------- #

TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
TEST_FRAC  = 1.0 - TRAIN_FRAC - VAL_FRAC

class SinogramDataset(Dataset):
    def __init__(self, clean_root: Path, art_root: Path, patch: int = 256):
        self.clean_root = Path(clean_root)
        self.art_root   = Path(art_root)
        self.clean_paths = sorted(self.clean_root.rglob("*.npy"))
        self.art_paths = [self.art_root / p.relative_to(self.clean_root) for p in self.clean_paths]
        self.patch = int(patch)
        if not all(p.exists() for p in self.art_paths):
            art_map = {p.name: p for p in self.art_root.rglob("*.npy")}
            self.art_paths = [art_map.get(p.name) for p in self.clean_paths]
            missing = [p for p in self.art_paths if p is None]
            if missing:
                raise RuntimeError(f"Missing artifact files for: {missing}")
        if not self.clean_paths:
            raise RuntimeError(f"No .npy files found under {self.clean_root}")

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_arr = np.load(self.clean_paths[idx]).astype(np.float32)
        art_arr   = np.load(self.art_paths[idx]).astype(np.float32)
        # normalization
        clean_arr = (clean_arr - clean_arr.min()) / (clean_arr.max() - clean_arr.min() + 1e-8)
        art_arr   = (art_arr   - art_arr.min())   / (art_arr.max()   - art_arr.min()   + 1e-8)
        # random crop to patch×patch
        H, W = clean_arr.shape
        th = tw = int(self.patch)
        i = np.random.randint(0, H - th + 1)
        j = np.random.randint(0, W - tw + 1)
        clean_crop = clean_arr[i:i+th, j:j+tw]
        art_crop   = art_arr  [i:i+th, j:j+tw]
        # to torch tensors, adding channel dimension
        clean = torch.from_numpy(clean_crop)[None]
        art   = torch.from_numpy(art_crop  )[None]
        return art, clean

# ---------------- RepLKNet Regression Wrapper ---------------- #

class RepLKNetReg(nn.Module):
    def __init__(
        self,
        large_kernel_sizes,
        layers,
        channels,
        drop_path_rate,
        small_kernel,
        dw_ratio=1,
        ffn_ratio=4,
        in_channels=1,
        use_checkpoint=False,
        small_kernel_merged=False,
    ):
        super().__init__()
        self.backbone = RepLKNet(
            large_kernel_sizes=large_kernel_sizes,
            layers=layers,
            channels=channels,
            drop_path_rate=drop_path_rate,
            small_kernel=small_kernel,
            dw_ratio=dw_ratio,
            ffn_ratio=ffn_ratio,
            in_channels=in_channels,
            num_classes=None,
            out_indices=[len(layers)-1],  # returns only the last feature map
            use_checkpoint=use_checkpoint,
            small_kernel_merged=small_kernel_merged,
            use_sync_bn=False,
            norm_intermediate_features=False
        )
        # drops classifier head if present
        if hasattr(self.backbone, 'head'):
            del self.backbone.head
        if hasattr(self.backbone, 'avgpool'):
            del self.backbone.avgpool
        # add regression decoder
        self.decoder = nn.Conv2d(channels[-1], 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        feats = self.backbone.forward_features(x)
        # unwraps single-element list if needed
        if isinstance(feats, list):
            feats = feats[0]
        out = self.decoder(feats)
        return F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

class SwinIRWrapper(nn.Module):
    def __init__(self, img_size=512, window_size=8, in_chans=1, out_chans=1,*, use_checkpoint: bool = False, **kwargs):
        super().__init__()
        self.net = SwinIR(
            img_size=img_size,
            window_size=window_size,
            in_chans=in_chans,
            out_chans=out_chans,
            img_range=1.0,
            upsampler='none',
            use_checkpoint=use_checkpoint,
            **kwargs
        )
    def forward(self, x):
        return self.net(x)

class RestormerWrapper(nn.Module):
    """
    Thin wrapper around swz30/Restormer for 1→1 denoising/artifact removal.
    Uses a lighter config so it fits easily on 10GB GPUs. Tweak if you want.
    """
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=48,
                 num_blocks=[2, 2, 2, 2],            
                 num_refinement_blocks=2,            
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False):
        super().__init__()
        self.net = RestormerNet(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task
        )

    def forward(self, x):
        return self.net(x)

class Swin2SRWrapper(nn.Module):
    def __init__(
        self,
        in_ch=1,
        embed_dim=96,
        depths=(6,6,6,6),
        num_heads=(6,6,6,6),
        window_size=8,
        upscale=1,
        upsampler='',
        img_range=1.0,
        img_size=(96,96),
    ):
        super().__init__()
        self.net = Swin2SR(
            img_size=img_size,        # must be a multiple of window_size
            patch_size=1,             # your patch size
            in_chans=in_ch,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            upscale=upscale,
            img_range=img_range,
            upsampler=upsampler,
            resi_connection='1conv',
        )

    def forward(self, x):
        # ensure input is padded internally if needed
        x = self.net.check_image_size(x)
        return self.net(x)


# ---------------- Args ---------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Train artifact-reduction models")
    p.add_argument("--model", choices=["mr_lkv", "unet", "replk","swinir","restormer","swin2sr"], default="mr_lkv",
                   help="Which architecture to train")
    p.add_argument("--clean-root",   type=Path, default=Path(CLEAN_SINOGRAM_ROOT))
    p.add_argument("--art-root",     type=Path, default=Path(ARTIFACT_ROOT))
    p.add_argument("--ckpt-dir",     type=Path, default=Path(CKPT_DIR))
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--epochs",       type=int, default=EPOCHS)
    p.add_argument("--save-interval",type=int, default=SAVE_INTERVAL)
    p.add_argument("--base-ch",      type=int, default=32, help="Base channels for MR_LKV")
    p.add_argument("--norm", choices=["batch","instance","group", "layer","none"], default="batch")
    p.add_argument("--no-decoder", action="store_true",
                   help="Disable decoder in MR_LKV")
    p.add_argument("--replk-kernels",   nargs=4, type=int, default=[31,29,27,13],
                   help="Large-kernel sizes per RepLKNet stage")
    p.add_argument("--replk-layers",    nargs=4, type=int, default=[2,2,18,2],
                   help="Number of blocks per RepLKNet stage")
    p.add_argument("--replk-channels",  nargs=4, type=int, default=[64,128,256,512],
                   help="Channel dims per RepLKNet stage")
    p.add_argument("--replk-small",     type=int, default=5,
                   help="Small-kernel size for reparam conv")
    p.add_argument("--replk-drop-path", type=float, default=0.0,
                   help="Drop path rate for RepLKNet")
    p.add_argument("--patch", type=int, default=96, help="random crop size")
    return p.parse_args()

# ---------------- Checkpoint folder mapping ---------------- #

_FOLDER_MAP = {
    "mr_lkv": "mr_lkv",
    "unet": "unet",
    "replk": "replknet",   
    "swinir": "swinir",
    "restormer": "restormer",
    "swin2sr": "swin2sr",
}

def _model_dir(ckpt_root: Path, model_name: str) -> Path:
    key = model_name.lower()
    folder = _FOLDER_MAP.get(key, key)
    return Path(ckpt_root) / folder

# ---------------- Train & Eval ---------------- #

def main():
    

    args = parse_args()
    print(f"Starting training with model={args.model}…", flush=True)
    results_dir = Path(__file__).resolve().parent / "results" / "plots"/args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = results_dir/"recon_sino"
    recon_dir.mkdir(parents=True, exist_ok=True)
    model_ckpt_dir = _model_dir(args.ckpt_dir, args.model)
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"→ Checkpoints will be saved to: {model_ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preparing data splits
    dataset = SinogramDataset(args.clean_root, args.art_root, patch=args.patch)
    total   = len(dataset)
    n_train = int(TRAIN_FRAC * total)
    n_val   = int(VAL_FRAC   * total)
    n_test  = total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # selects model
    if args.model == "unet":
        model = UNet(in_channels=1, base_channels=64, levels=4,
                     norm_type="batch", dropout_bottleneck=0.1,
                     final_activation=None).to(device)
    elif args.model == "mr_lkv":
        model = MR_LKV(
            in_channels=1,
            base_channels=args.base_ch,
            depths=[2,2,3,2],
            kernels=[35,55,75,95],
            norm_type=args.norm,
            use_decoder=(not args.no_decoder),
            final_activation=None
        ).to(device)
    elif args.model == "replk":
        model = RepLKNetReg(large_kernel_sizes=args.replk_kernels, layers=args.replk_layers,
                            channels=args.replk_channels, drop_path_rate=args.replk_drop_path,
                            small_kernel=args.replk_small, in_channels=1).to(device)
    elif args.model == "swinir":
        model = SwinIRWrapper(img_size=512, window_size=8, in_chans=1, out_chans=1,
                              depths=[4,4,4,4], embed_dim=64, num_heads=[2,2,2,2],
                              mlp_ratio=2, use_checkpoint=True).to(device)
    elif args.model == "restormer":
        model = RestormerWrapper(
            inp_channels=1,
            out_channels=1,
            dim=48,
            num_blocks=[2,2,2,4],          
            num_refinement_blocks=2,
            heads=[1,2,2,4],
            ffn_expansion_factor=2.0,
            bias=False,
            LayerNorm_type='WithBias'
        ).to(device)
    elif args.model == "swin2sr":
        model = Swin2SRWrapper(
            in_ch=1,
            embed_dim=64,
            depths=(4,4,4,4),
            num_heads=(4,4,4,4),
            window_size=8,        # divides 96 exactly
            upscale=1,            # for artifact removal
            upsampler='',         # no upsampling
            img_range=1.0,
            img_size=(64,64),     # your dataset’s crop size
        ).to(device)


    else:
        raise ValueError(f"Unknown model {args.model}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scheduler.step(val_loss)
   # criterion = nn.MSELoss(reduction='mean')
    criterion = L1SSIMLoss(alpha=0.7) #changed from 0.84 to 0.7 for more pixel focus


    best_val = float('inf')
    epochs_no_imp = 0
    EARLY_STOP_PATIENCE = 10
    # ---- Logging setup ---- #
    log_path = results_dir  / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_psnr,val_ssim\n")

    train_losses, val_losses, psnr_scores, ssim_scores = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        running_train = 0.0
        for art, clean in train_loader:
            art, clean = art.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(art)
            if pred.shape[-2:] != clean.shape[-2:]:
                pred = F.interpolate(pred, size=clean.shape[-2:],
                                     mode='bilinear', align_corners=False)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()
            running_train += loss.item() * art.size(0)
        train_loss = running_train / n_train

        # validate
        model.eval()
        running_val = running_psnr = running_ssim = 0.0
        with torch.no_grad():
            for art, clean in val_loader:
                art, clean = art.to(device), clean.to(device)
                pred = model(art)
                if pred.shape[-2:] != clean.shape[-2:]:
                    pred = F.interpolate(pred, size=clean.shape[-2:],
                                         mode='bilinear', align_corners=False)
                running_val   += criterion(pred, clean).item() * art.size(0)
                running_psnr  += psnr(pred, clean).item() * art.size(0)
                running_ssim  += ssim(pred, clean).item() * art.size(0)
        val_loss = running_val / n_val
        val_psnr = running_psnr / n_val
        val_ssim = running_ssim / n_val
        print(f"Epoch {epoch}/{args.epochs} — "f"Train: {train_loss:.6f}, Val: {val_loss:.6f} | "f"PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}", flush=True)
        # ---- Save to CSV log ---- #
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_psnr:.4f},{val_ssim:.4f}\n")

        # ---- Store for plotting ---- #
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        psnr_scores.append(val_psnr)
        ssim_scores.append(val_ssim)

        scheduler.step(val_loss)
       

        print(f"⇢ learning rate is now {scheduler.get_last_lr()[0]:.2e}")
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_imp = 0
            torch.save(model.state_dict(), model_ckpt_dir / "best_model.pth")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % args.save_interval == 0:
            ckpt = model_ckpt_dir / f"epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
    # ---- Plot training curves ---- #
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(psnr_scores, label="Val PSNR (dB)", marker='s')
    plt.plot(ssim_scores, label="Val SSIM", marker='s')
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)

    plot_path = results_dir  / "training_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training curves to {plot_path}")

    # test
    model.eval()
    running_test = running_psnr = running_ssim = 0.0
    with torch.no_grad():
        for batch_idx, (art, clean) in enumerate(test_loader):
        #for art, clean in test_loader:
            art, clean = art.to(device), clean.to(device)
            pred = model(art)
            if pred.shape[-2:] != clean.shape[-2:]:
                pred = F.interpolate(pred, size=clean.shape[-2:],
                                     mode='bilinear', align_corners=False)
            # Save only reconstructed sinograms for first batch
            if batch_idx == 0:
                #save_recon_sinos(pred, epoch, recon_dir, max_imgs=3)
                save_image(art[0], recon_dir / f"epoch{epoch:03d}_input.png")
                save_image(pred[0], recon_dir / f"epoch{epoch:03d}_recon.png")
                save_image(clean[0], recon_dir / f"epoch{epoch:03d}_target.png")

                print(f"Saved reconstructed sinograms to {recon_dir}")
            running_test   += criterion(pred, clean).item() * art.size(0)
            running_psnr   += psnr(pred, clean).item() * art.size(0)
            running_ssim   += ssim(pred, clean).item() * art.size(0)
    test_loss = running_test / n_test
    test_psnr = running_psnr / n_test
    test_ssim = running_ssim / n_test
    print(f"Final Test — Loss: {test_loss:.6f}, PSNR: {test_psnr:.2f} dB, SSIM: {test_ssim:.4f}")

if __name__ == "__main__":
    main()