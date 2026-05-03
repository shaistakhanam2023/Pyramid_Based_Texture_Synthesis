import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI errors
import matplotlib.pyplot as plt
from src.pyramid import SteerablePyramid

# 1. Create a "test image" (White Noise)
size = 256
input_img = np.random.rand(size, size).astype(np.float32)

# 2. Initialize the Pyramid
pyr_engine = SteerablePyramid(size, size, n_scales=4, n_orient=4)

# 3. RUN: Decompose and Reconstruct
coeffs = pyr_engine.decompose(input_img)
output_img = pyr_engine.reconstruct(coeffs)

# 4. VERIFY: The Math Check
mse = np.mean((input_img - output_img)**2)
print(f"--- Phase 1 Results ---")
print(f"Reconstruction MSE: {mse:.2e}")

if mse < 1e-5:
    print("✅ SUCCESS: Pyramid is perfectly invertible.")
else:
    print("❌ FAIL: Check your filter math in filters.py.")

# 5. VISUALIZE (Save to disk instead of showing window)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Input (Noise)")
plt.imshow(input_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(output_img, cmap='gray')
plt.axis('off')

plt.savefig('phase1_result.png')
print("📸 Visualization saved to 'phase1_result.png'")