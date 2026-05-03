# src/filters.py
import numpy as np
import math
from typing import Tuple

def compute_alpha(n_orient: int) -> float:
    """Compute the constant of reversibility alpha for given number of orientations."""
    log_fact_alpha1 = 0.0
    for k in range(2, n_orient):
        log_fact_alpha1 += math.log(k)
    log_fact_alpha2 = 0.0
    for k in range(2, 2 * n_orient - 1):
        log_fact_alpha2 += math.log(k)
    log_alpha = (n_orient - 1) * math.log(2) + log_fact_alpha1 - 0.5 * (math.log(n_orient) + log_fact_alpha2)
    return math.exp(log_alpha)

def _freq_grid(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return radial and angular frequency grids for given shape."""
    h, w = shape
    fx = np.fft.fftfreq(w) * 2 * np.pi
    fy = np.fft.fftfreq(h) * 2 * np.pi
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    r = np.sqrt(FX**2 + FY**2)
    theta = np.arctan2(FY, FX)
    return r.astype(np.float32), theta.astype(np.float32)

def compute_high_filter(shape: Tuple[int, int], factor: float = 1.0) -> np.ndarray:
    """High-pass filter H(r/factor) in the frequency domain."""
    r, _ = _freq_grid(shape)
    r_scaled = r / factor
    H = np.zeros_like(r_scaled, dtype=np.float32)
    mask_mid = (r_scaled > np.pi/4) & (r_scaled < np.pi/2)
    mask_high = r_scaled >= np.pi/2
    H[mask_mid] = np.cos(np.pi/2 * np.log2(2 * r_scaled[mask_mid] / np.pi))
    H[mask_high] = 1.0
    return H

def compute_low_filter(shape: Tuple[int, int], factor: float = 1.0) -> np.ndarray:
    """Low-pass filter L(r/factor) in the frequency domain."""
    r, _ = _freq_grid(shape)
    r_scaled = r / factor
    L = np.zeros_like(r_scaled, dtype=np.float32)
    L[r_scaled == 0] = 1.0
    mask_low = r_scaled <= np.pi/4
    mask_mid = (r_scaled > np.pi/4) & (r_scaled < np.pi/2)
    L[mask_low] = 1.0
    L[mask_mid] = np.cos(np.pi/2 * np.log2(4 * r_scaled[mask_mid] / np.pi))
    return L

def compute_steerable_filter(shape: Tuple[int, int], q: int,
                              n_orient: int, alpha: float) -> np.ndarray:
    """
    Band-pass filter Bq = H(r) * Gq(theta).
    CORRECTED VERSION with 'sign' logic.
    """
    r, theta = _freq_grid(shape)
    H = compute_high_filter(shape, factor=1)

    # Compute Gq
    delta = theta - np.pi * q / n_orient
    term = np.cos(delta) ** (n_orient - 1)
    mask1 = np.abs(delta) <= np.pi/2

    delta2 = theta - np.pi * (q - n_orient) / n_orient
    term2 = np.cos(delta2) ** (n_orient - 1)
    mask2 = np.abs(delta2) <= np.pi/2

    # Calculate sign: (-1)^(n_orient - 1)
    # If n_orient is even (e.g., 4), sign is -1. If odd, sign is 1.
    sign = 1 if (n_orient % 2 == 1) else -1

    G = alpha * (term * mask1 + sign * term2 * mask2)
    B = H * G
    return B.astype(np.float32)



def downsample_fft(fft_in: np.ndarray) -> np.ndarray:
    """
    Downsample by factor 2.
    Corrected to exactly match C code indexing.
    """
    h, w = fft_in.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("Dimensions must be even for downsampling")
    h2, w2 = h // 2, w // 2
    
    # Logic from C: 
    # if (l < mmy) ll = l else ll = l + my
    # if (k < mmx) kk = k else kk = k + mx
    
    # mmy = ceil(my/2) = ceil((h/2)/2) = ceil(h/4)
    # mmx = ceil(mx/2) = ceil((w/2)/2) = ceil(w/4)
    
    mmy = (h2 + 1) // 2  # ceil(h/4)
    mmx = (w2 + 1) // 2  # ceil(w/4)
    
    # Create output indices (0 to h2-1)
    l_indices = np.arange(h2)
    k_indices = np.arange(w2)
    
    # Map to input indices
    # For l < mmy, we take l (the start)
    # For l >= mmy, we take l + h2 (the end/bottom part)
    ll = np.where(l_indices < mmy, l_indices, l_indices + h2)
    kk = np.where(k_indices < mmx, k_indices, k_indices + w2)
    
    return fft_in[np.ix_(ll, kk)]

def upsample_fft(fft_in: np.ndarray) -> np.ndarray:
    """
    Upsample by factor 2.
    Matches the inverse of the corrected downsample_fft.
    """
    h, w = fft_in.shape
    h2, w2 = 2 * h, 2 * w
    
    # Logic from C (inverse of downsample)
    # mmx = ceil(h/4), mmy = ceil(w/4)
    mmx = (h + 1) // 2
    mmy = (w + 1) // 2
    
    # Input indices (0 to h-1)
    l_indices = np.arange(h)
    k_indices = np.arange(w)
    
    # Map to output indices
    # If l < mmx, output index is l
    # If l >= mmx, output index is l + h
    ll = np.where(l_indices < mmx, l_indices, l_indices + h)
    kk = np.where(k_indices < mmy, k_indices, k_indices + w)
    
    out = np.zeros((h2, w2), dtype=fft_in.dtype)
    out[np.ix_(ll, kk)] = fft_in
    return out