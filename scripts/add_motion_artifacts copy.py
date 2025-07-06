import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import SINOGRAM_ROOT, ARTIFACT_ROOT, N_VIEWS


def simulate_motion(sinogram: np.ndarray,
                    max_translation: float = 5.0,
                    frequency: float = 3.0) -> np.ndarray:
    """Simulate sinusoidal motion artifacts on a sinogram."""
    n_views, n_det = sinogram.shape
    motion = max_translation * np.sin(2 * np.pi * np.linspace(0, 1, n_views) * frequency)

    distorted = np.zeros_like(sinogram)
    for i in range(n_views):
        shift = int(motion[i])
        distorted[i] = np.roll(sinogram[i], shift)

    return distorted


def main():
    # Ensure the artifact root directory exists
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    # Gather all sinogram files
    sinogram_paths = list(SINOGRAM_ROOT.rglob("*.npy"))
    print(f"🔍 Looking in {SINOGRAM_ROOT!r}")
    print(f"    → found {len(sinogram_paths)} sinogram files")

    # Fail early if no files are found
    if len(sinogram_paths) == 0:
        raise RuntimeError(
            f"No sinogram files found in {SINOGRAM_ROOT!r}."
            " Check that the path is correct and that there are .npy files there."
        )

    # Process each file with a progress bar
    for path in tqdm(sinogram_paths, desc="Processing sinograms"):
        relative_path = path.relative_to(SINOGRAM_ROOT)
        out_path = ARTIFACT_ROOT / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Load, process, and save
        sino = np.load(path)
        sino_artifact = simulate_motion(sino)
        np.save(out_path, sino_artifact)

    print(f"✅ Motion artifacts saved to: {ARTIFACT_ROOT}")


if __name__ == "__main__":
    main()
