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

# network definition
from model_mr_lkv import MR_LKV


torch.manual_seed(42)

class SinogramDataset(Dataset):
    """
    Paired dataset that recursively finds .npy files in CLEAN_SINOGRAM_ROOT
    and matches them to ARTIFACT_ROOT using relative paths.

    If directory structures diverge, a filename-based lookup is used.
    """
    def __init__(self, clean_root: Path, art_root: Path):
        self.clean_root = Path(clean_root)
        self.art_root   = Path(art_root)
        self.clean_paths = sorted(self.clean_root.rglob("*.npy"))

        # attempt structural mirror
        self.art_paths = [
            self.art_root / p.relative_to(self.clean_root)
            for p in self.clean_paths
        ]
        # fallback to name-based lookup if necessary
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
        # add channel dim
        clean = torch.from_numpy(clean_arr)[None]
        art   = torch.from_numpy(art_arr)[None]
        return art, clean


def parse_args():
    p = argparse.ArgumentParser(description="Train MR-LKV for artifact reduction")
    p.add_argument("--clean-root",   type=Path,
                   default=Path(CLEAN_SINOGRAM_ROOT), help="Clean sinogram .npy root")
    p.add_argument("--art-root",     type=Path,
                   default=Path(ARTIFACT_ROOT), help="Artifact sinogram .npy root")
    p.add_argument("--ckpt-dir",     type=Path,
                   default=Path(CKPT_DIR), help="Directory to save checkpoints")
    p.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--epochs",       type=int, default=EPOCHS)
    p.add_argument("--save-interval",type=int, default=SAVE_INTERVAL)
    return p.parse_args()


def main():
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # prepare dataset and splits
    dataset = SinogramDataset(args.clean_root, args.art_root)
    total = len(dataset)
    n_train = int(0.8 * total)
    n_val   = int(0.1 * total)
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

    # model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MR_LKV().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # training loop
    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        running_train = 0.0
        for art, clean in train_loader:
            art, clean = art.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(art)
            # if shapes don’t match, upsample predictions
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

        # validate
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for art, clean in val_loader:
                art, clean = art.to(device), clean.to(device)
                running_val += criterion(model(art), clean).item() * art.size(0)
        val_loss = running_val / n_val

        print(f"Epoch {epoch}/{args.epochs} — train: {train_loss:.6f}, val: {val_loss:.6f}")

        # save
        if epoch % args.save_interval == 0:
            ckpt_path = args.ckpt_dir / f"epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"🔖 Saved checkpoint: {ckpt_path}")

    # final test
    model.eval()
    running_test = 0.0
    with torch.no_grad():
        for art, clean in test_loader:
            art, clean = art.to(device), clean.to(device)
            running_test += criterion(model(art), clean).item() * art.size(0)
    test_loss = running_test / n_test
    print(f"Final Test Loss: {test_loss:.6f}")

if __name__ == "__main__":
    main()
