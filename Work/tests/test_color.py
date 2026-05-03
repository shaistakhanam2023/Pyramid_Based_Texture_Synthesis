import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.color import PCAColorSpace

def test_pca_reconstruction():
    # Create a random "RGB" image (values between 0 and 1)
    H, W = 64, 64
    image = np.random.rand(H, W, 3).astype(np.float32)
    
    # Initialize PCA transformer
    pca = PCAColorSpace()
    
    # Transform to PCA space
    pca_image = pca.rgb_to_pca(image)
    
    # Reconstruct back to RGB space
    reconstructed = pca.pca_to_rgb(pca_image)
    
    # Check if the reconstruction matches the original image perfectly
    error = np.max(np.abs(image - reconstructed))
    
    assert error < 1e-4, f"PCA inversion failed! Max error: {error}"
    
    print(f"Color PCA Test: PASSED (Max error: {error:.8f})")

if __name__ == "__main__":
    test_pca_reconstruction()