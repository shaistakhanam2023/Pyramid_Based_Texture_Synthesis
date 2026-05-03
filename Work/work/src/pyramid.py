# src/pyramid.py
import numpy as np
from .filters import (compute_alpha, compute_high_filter, compute_low_filter,
                      compute_steerable_filter, downsample_fft, upsample_fft)

# --- Custom FFT Wrappers to match C Code Normalization ---
def do_fft(image: np.ndarray) -> np.ndarray:
    """
    Computes DFT and divides by N (matches C code's do_fft).
    """
    N = image.shape[0] * image.shape[1]
    return np.fft.fft2(image) / N

def do_ifft(spectrum: np.ndarray) -> np.ndarray:
    """
    Computes inverse DFT and multiplies by N (matches C code's do_ifft).
    """
    N = spectrum.shape[0] * spectrum.shape[1]
    # NumPy's ifft2 divides by N, so we must multiply by N to cancel it out
    return np.fft.ifft2(spectrum) * N
# -------------------------------------------------------

class SteerablePyramid:
    """
    Steerable pyramid decomposition and reconstruction.
    """

    def __init__(self, height: int, width: int, n_scales: int = 4, n_orient: int = 4):
        self.height = height
        self.width = width
        self.n_scales = n_scales
        self.n_orient = n_orient
        self.alpha = compute_alpha(n_orient)
        self._build_filters()

    def _build_filters(self) -> None:
        """Precompute all filters needed for analysis and synthesis."""
        # Full-size filters (first level)
        self.H0 = compute_high_filter((self.height, self.width), factor=2)
        self.L0 = compute_low_filter((self.height, self.width), factor=2)

        # Per-scale filters (index 1..n_scales)
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

    def decompose(self, image: np.ndarray) -> dict:
        """
        Decompose an image into a steerable pyramid using C-style normalization.
        """
        image = image.astype(np.float32)
        U = do_fft(image)  # Uses custom wrapper

        # High-pass residual
        high_res = do_ifft(U * self.H0).real

        # Initial low-pass band
        v = do_ifft(U * self.L0).real

        levels = []
        for scale in range(1, self.n_scales + 1):
            V = do_fft(v)

            # Oriented subbands at this scale
            subbands = []
            for q in range(self.n_orient):
                B_q = self.B[scale][q]
                subband = do_ifft(V * B_q).real
                subbands.append(subband)

            # Low-pass + downsample for next scale
            L_s = self.L[scale]
            v_low_hat = V * L_s
            v_hat_small = downsample_fft(v_low_hat)
            v = do_ifft(v_hat_small).real

            levels.append(subbands)

        low_res = v
        return {'high': high_res, 'levels': levels, 'low': low_res}

    def reconstruct(self, pyramid: dict) -> np.ndarray:
        """
        Reconstruct an image using C-style normalization.
        """
        u = pyramid['low'].astype(np.float32)

        # Iterate from coarse to fine scales
        for scale in range(self.n_scales, 0, -1):
            U = do_fft(u)
            U_up = upsample_fft(U)
            L_s = self.L[scale]
            U_low = U_up * L_s
            u = do_ifft(U_low).real

            # Add oriented subbands
            subbands = pyramid['levels'][scale - 1]
            for q in range(self.n_orient):
                subband = subbands[q].astype(np.float32)
                B_q = self.B[scale][q]
                S = do_fft(subband)
                u += do_ifft(S * B_q).real

        # Final low-pass and high-pass additions
        U = do_fft(u)
        u = do_ifft(U * self.L0).real

        high_res = pyramid['high'].astype(np.float32)
        U_high = do_fft(high_res)
        u += do_ifft(U_high * self.H0).real

        return u