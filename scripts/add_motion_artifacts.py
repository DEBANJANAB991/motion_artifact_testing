
#!/usr/bin/env python3
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from config import SINOGRAM_ROOT, ARTIFACT_ROOT, N_VIEWS, DET_SPACING

# Spline-based motion model (Thies et al. 2021)
from scipy.interpolate import CubicSpline

def apply_motion(sino_np: np.ndarray,
                 n_views: int = N_VIEWS,
                 det_spacing: float = DET_SPACING,
                 n_nodes: int = 10,
                 max_trans: float = 10.0) -> np.ndarray:
    # control-point indices
    t_nodes = np.linspace(0, n_views, n_nodes, endpoint=False)

    # random translations at control-points
    tx = np.random.uniform(-max_trans, max_trans, n_nodes)
    ty = np.random.uniform(-max_trans, max_trans, n_nodes)
    # enforce periodic endpoints
    tx[-1] = tx[0]
    ty[-1] = ty[0]

    # build periodic splines
    cs_tx = CubicSpline(t_nodes, tx, bc_type='periodic')
    cs_ty = CubicSpline(t_nodes, ty, bc_type='periodic')

    # evaluate translations at each projection
    t_idx = np.arange(n_views)
    txs = cs_tx(t_idx)
    tys = cs_ty(t_idx)
    angs = np.linspace(0, 2 * np.pi, n_views, endpoint=False)

    n_det = sino_np.shape[1]
    coords = np.arange(n_det)
    out = np.zeros_like(sino_np)

    for i in range(n_views):
        shift_mm = txs[i] * np.cos(angs[i]) + tys[i] * np.sin(angs[i])
        shift_px = shift_mm / det_spacing
        out[i] = np.interp(
            coords,
            coords - shift_px,
            sino_np[i],
            left=0.0,
            right=0.0
        )
    return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_npy = sorted(SINOGRAM_ROOT.rglob("*.npy"))
    for npy_path in tqdm(all_npy, desc="Motion+cleanup"):
        rel = npy_path.relative_to(SINOGRAM_ROOT)
        dst_path = ARTIFACT_ROOT / rel
        dst_path.parent.mkdir(exist_ok=True, parents=True)

        clean = np.load(npy_path)
        art = apply_motion(clean)
        np.save(dst_path, art)

        try:
            npy_path.unlink()  # delete clean on the fly
        except Exception as e:
            print(f"⚠️ Could not delete {npy_path}: {e}")

    print("✅ All sinograms artifacted under", ARTIFACT_ROOT)

if __name__ == "__main__":
    main()

