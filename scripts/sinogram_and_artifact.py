import numpy as np
import matplotlib.pyplot as plt

# ← adjust these paths to point at one example sinogram pair
clean_path    = "/scratch/iwi5/iwi5293h/sinograms/example_sino.npy"
artifact_path = "/home/vault/iwi5/iwi5293h/.../sinograms_artifact/example_sino.npy"

# load
clean    = np.load(clean_path)
artifact = np.load(artifact_path)

# compute difference
diff = np.abs(clean - artifact)

# plot clean sinogram
plt.imshow(clean)
plt.title("Clean Sinogram")
plt.axis("off")
plt.show()

# plot artifacted sinogram
plt.imshow(artifact)
plt.title("Artifacted Sinogram")
plt.axis("off")
plt.show()

# plot absolute difference
plt.imshow(diff)
plt.title("Absolute Difference")
plt.axis("off")
plt.show()
