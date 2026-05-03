import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.synthesis import TextureSynthesizer

def test_synthesis_grayscale():
    # 1. Create a dummy target "texture" (a mathematical grid pattern)
    x = np.linspace(0, 4*np.pi, 64)
    y = np.linspace(0, 4*np.pi, 64)
    X, Y = np.meshgrid(x, y)
    target = (np.sin(X) + np.cos(Y)).astype(np.float32)
    
    # 2. Initialize synthesizer (use a small pyramid for a fast test)
    synthesizer = TextureSynthesizer(n_scales=2, n_orient=4)
    
    # 3. Synthesize a LARGER texture from the small sample
    out_shape = (128, 128)
    
    try:
        # Just 2 iterations to ensure the loop runs without crashing
        result = synthesizer.synthesize_grayscale(target, out_shape, n_iters=2)
    except Exception as e:
        assert False, f"Synthesis loop crashed with error: {e}"
    
    # 4. Validate outputs
    assert result.shape == out_shape, f"Expected shape {out_shape}, got {result.shape}"
    assert not np.isnan(result).any(), "Output contains NaN (Not a Number) values! Math exploded."
    
    print("Grayscale Synthesis Loop Test: PASSED")

if __name__ == "__main__":
    test_synthesis_grayscale()