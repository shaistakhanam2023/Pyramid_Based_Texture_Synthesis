import numpy as np
import math
from typing import Tuple

def _freq_grid(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Generates frequency radius and angle grids for FFT."""
    h, w = shape
    fy = np.fft.fftfreq(h) * 2 * np.pi
    fx = np.fft.fftfreq(w) * 2 * np.pi
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    
    r = np.sqrt(FX**2 + FY**2)
    theta = np.arctan2(FY, FX)
    return r.astype(np.float32), theta.astype(np.float32)

def compute_high_filter(shape: Tuple[int, int], factor: float = 1.0) -> np.ndarray:
    """High-pass filter H_0(r/factor)."""
    r, _ = _freq_grid(shape)
    r_scaled = r / factor
    
    H = np.zeros_like(r_scaled, dtype=np.float32)
    mask_mid = (r_scaled > np.pi/4) & (r_scaled < np.pi/2)
    mask_high = r_scaled >= np.pi/2
    
    H[mask_mid] = np.cos(np.pi/2 * np.log2(2 * r_scaled[mask_mid] / np.pi))
    H[mask_high] = 1.0
    return H

def compute_low_filter(shape: Tuple[int, int], factor: float = 1.0) -> np.ndarray:
    """Low-pass filter L_0(r/factor)."""
    r, _ = _freq_grid(shape)
    r_scaled = r / factor
    
    L = np.zeros_like(r_scaled, dtype=np.float32)
    mask_low = r_scaled <= np.pi/4
    mask_mid = (r_scaled > np.pi/4) & (r_scaled < np.pi/2)
    
    L[mask_low] = 1.0
    L[mask_mid] = np.cos(np.pi/2 * np.log2(4 * r_scaled[mask_mid] / np.pi))
    return L

def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to the circular range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_steerable_filter(shape: Tuple[int, int], q: int, n_orient: int, alpha: float) -> np.ndarray:
    """Steerable bandpass filter B_q."""
    r, theta = _freq_grid(shape)
    H = compute_high_filter(shape, factor=1.0)
    
    delta1 = wrap_angle(theta - np.pi * q / n_orient)
    mask1 = np.abs(delta1) <= np.pi/2
    term1 = (np.cos(delta1) ** (n_orient - 1)) * mask1
    
    delta2 = wrap_angle(theta - np.pi * (q - n_orient) / n_orient)
    mask2 = np.abs(delta2) <= np.pi/2
    term2 = (np.cos(delta2) ** (n_orient - 1)) * mask2
    
    # FIX: Force symmetry so IFFT is strictly real (energy is preserved)
    sign = 1.0 
    
    G = alpha * (term1 + sign * term2)
    B = H * G
    return B.astype(np.float32)

def compute_alpha(K: int) -> float:
    """Normalization constant for steerable filters to guarantee invertibility."""
    num = (2 ** (K - 1)) * math.factorial(K - 1)
    den = math.sqrt(K * math.factorial(2 * K - 2))
    return num / den