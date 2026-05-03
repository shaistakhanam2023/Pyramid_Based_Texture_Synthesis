import os
import numpy as np
from PIL import Image

def load_image(filepath: str) -> np.ndarray:
    """
    Loads an image from disk and normalizes it to [0.0, 1.0] float32.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find image at: {filepath}")
        
    img = Image.open(filepath).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0

def save_image(filepath: str, image: np.ndarray):
    """
    Saves a [0.0, 1.0] float32 image array to disk.
    """
    # Clip values to ensure they stay in valid bounds, then convert to 8-bit
    image_clipped = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(image_clipped)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    img.save(filepath)