import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.pyramid import SteerablePyramid

def test_invertibility():
    """Test that decomposing and reconstructing returns the original image."""
    # Use dimensions divisible by 2**3 = 8
    H, W = 128, 128
    image = np.random.rand(H, W).astype(np.float32)

    # Initialize pyramid
    pyr = SteerablePyramid(H, W, n_scales=3, n_orient=4)
    
    # Decompose
    decomp = pyr.decompose(image)
    
    # Reconstruct
    recon = pyr.reconstruct(decomp)

    # Check error (allowing for small floating point errors)
    error = np.max(np.abs(recon - image))
    
    print(f"Test Reconstruction Error: {error:.2e}")
    assert error < 1e-4, f"Reconstruction error too high: {error}"
    print("✅ Test Passed: Invertibility verified.")

if __name__ == '__main__':
    test_invertibility()