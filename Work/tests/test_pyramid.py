import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.pyramid import SteerablePyramid

def test_invertibility():
    """Test that decomposing and reconstructing returns the original image."""
    H, W = 128, 128
    image = np.random.rand(H, W).astype(np.float32)

    pyr = SteerablePyramid(H, W, n_scales=3, n_orient=4)
    decomp = pyr.decompose(image)
    recon = pyr.reconstruct(decomp)

    error = np.max(np.abs(recon - image))
    print(f"Test Reconstruction Error: {error:.8f}")
    
    # With the math correctly fixed, floating point error should be well below 1e-4
    assert error < 1e-4, f"Reconstruction failed! Error: {error}"

if __name__ == "__main__":
    test_invertibility()