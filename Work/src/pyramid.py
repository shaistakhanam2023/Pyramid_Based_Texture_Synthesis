import numpy as np
from typing import Tuple, List, Any
from src.filters import compute_high_filter, compute_low_filter, compute_steerable_filter, compute_alpha

def do_fft(image: np.ndarray) -> np.ndarray:
    """Forward FFT, scaled by 1/N for invertibility logic."""
    N = image.shape[0] * image.shape[1]
    return np.fft.fft2(image) / N

def do_ifft(spectrum: np.ndarray) -> np.ndarray:
    """Inverse FFT, scaled by N."""
    N = spectrum.shape[0] * spectrum.shape[1]
    return np.fft.ifft2(spectrum) * N

def downsample_fft(fft_in: np.ndarray) -> np.ndarray:
    """Downsample by 2 in frequency domain (extract central low frequencies)."""
    h, w = fft_in.shape
    h2, w2 = h // 2, w // 2
    
    mmy = (h2 + 1) // 2
    mmx = (w2 + 1) // 2
    
    l_indices = np.arange(h2)
    k_indices = np.arange(w2)
    
    ll = np.where(l_indices < mmy, l_indices, l_indices + (h - h2))
    kk = np.where(k_indices < mmx, k_indices, k_indices + (w - w2))
    return fft_in[np.ix_(ll, kk)]

def upsample_fft(fft_in: np.ndarray) -> np.ndarray:
    """Upsample by 2 in frequency domain (zero pad)."""
    h, w = fft_in.shape
    h2, w2 = 2 * h, 2 * w
    
    mmx = (h + 1) // 2
    mmy = (w + 1) // 2
    
    l_indices = np.arange(h)
    k_indices = np.arange(w)
    
    ll = np.where(l_indices < mmx, l_indices, l_indices + h)
    kk = np.where(k_indices < mmy, k_indices, k_indices + w)
    
    out = np.zeros((h2, w2), dtype=fft_in.dtype)
    out[np.ix_(ll, kk)] = fft_in
    return out

class SteerablePyramid:
    def __init__(self, height: int, width: int, n_scales: int = 4, n_orient: int = 4):
        self.height = height
        self.width = width
        self.n_scales = n_scales
        self.n_orient = n_orient
        self.alpha = compute_alpha(n_orient)
        self._build_filters()
        
    def _build_filters(self):
        self.H0 = compute_high_filter((self.height, self.width), factor=2)
        self.L0 = compute_low_filter((self.height, self.width), factor=2)
        
        self.L = [None]
        self.B = [None]
        
        for scale in range(1, self.n_scales + 1):
            h = self.height // (2 ** (scale - 1))
            w = self.width // (2 ** (scale - 1))
            size = (h, w)
            
            L_s = compute_low_filter(size, factor=1)
            self.L.append(L_s)
            
            B_s = []
            for q in range(self.n_orient):
                B_q = compute_steerable_filter(size, q, self.n_orient, self.alpha)
                B_s.append(B_q)
            self.B.append(B_s)
            
    def decompose(self, image: np.ndarray) -> List[Any]:
        U = do_fft(image)
        u_high = do_ifft(U * self.H0).real
        
        v_low_hat = U * self.L0
        v = do_ifft(v_low_hat).real
        
        levels = []
        for scale in range(1, self.n_scales + 1):
            V = do_fft(v)
            
            subbands = []
            for q in range(self.n_orient):
                B_q = self.B[scale][q]
                subband = do_ifft(V * B_q).real
                subbands.append(subband)
            levels.append(subbands)
            
            L_s = self.L[scale]
            v_low_hat = V * L_s
            v_hat_small = downsample_fft(v_low_hat)
            v = do_ifft(v_hat_small).real
            
        return [u_high] + levels + [v]
        
    def reconstruct(self, coeffs: List[Any]) -> np.ndarray:
        u_high = coeffs[0]
        levels = coeffs[1:-1]
        v = coeffs[-1]
        
        U = do_fft(v)
        
        for scale in range(self.n_scales, 0, -1):
            L_s = self.L[scale]
            U_up = upsample_fft(U)
            U_low = U_up * L_s
            
            subbands = levels[scale - 1]
            U_sub_sum = np.zeros_like(U_low)
            
            for q in range(self.n_orient):
                B_q = self.B[scale][q]
                S = do_fft(subbands[q])
                U_sub_sum += S * B_q
                
            U = U_low + U_sub_sum
            
        U_high = do_fft(u_high)
        U_final = U * self.L0 + U_high * self.H0
        return do_ifft(U_final).real