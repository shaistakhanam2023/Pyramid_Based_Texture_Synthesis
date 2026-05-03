import numpy as np
from typing import List, Any
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm isn't installed, though it should be from Phase 0
    tqdm = lambda x, **kwargs: x

from src.pyramid import SteerablePyramid
from src.histogram import match_histogram

class TextureSynthesizer:
    def __init__(self, n_scales: int = 4, n_orient: int = 4):
        self.n_scales = n_scales
        self.n_orient = n_orient
        
    def _match_pyramids(self, synth_pyr: List[Any], target_pyr: List[Any]) -> List[Any]:
        """Matches histograms of all subbands from the synthesis pyramid to the target pyramid."""
        matched_pyr = []
        
        # 1. Match high-pass residual
        matched_pyr.append(match_histogram(synth_pyr[0], target_pyr[0]))
        
        # 2. Match oriented subbands at each scale
        for scale in range(1, self.n_scales + 1):
            synth_subbands = synth_pyr[scale]
            target_subbands = target_pyr[scale]
            
            matched_subbands = []
            for q in range(self.n_orient):
                matched = match_histogram(synth_subbands[q], target_subbands[q])
                matched_subbands.append(matched)
            matched_pyr.append(matched_subbands)
            
        # 3. Match low-pass residual
        matched_pyr.append(match_histogram(synth_pyr[-1], target_pyr[-1]))
        
        return matched_pyr

    def synthesize_grayscale(self, target_image: np.ndarray, out_shape: tuple, n_iters: int = 5) -> np.ndarray:
        """
        Synthesizes a new texture that matches the target_image.
        
        Parameters
        ----------
        target_image : np.ndarray
            The original sample texture (2D grayscale).
        out_shape : tuple
            The desired (height, width) of the output texture.
        n_iters : int
            Number of synthesis iterations (default 5 is usually enough).
        """
        H_out, W_out = out_shape
        H_in, W_in = target_image.shape
        
        # Initialize pyramid builders (this pre-computes the FFT filters so we don't recalculate them in the loop)
        target_pyramid_builder = SteerablePyramid(H_in, W_in, self.n_scales, self.n_orient)
        synth_pyramid_builder = SteerablePyramid(H_out, W_out, self.n_scales, self.n_orient)
        
        # Step 1: Decompose target image to get the reference subbands
        target_pyr = target_pyramid_builder.decompose(target_image)
        
        # Step 2: Initialize noise image and match its global histogram to the target
        synth_image = np.random.randn(H_out, W_out).astype(np.float32)
        synth_image = match_histogram(synth_image, target_image)
        
        # Step 3: The Core Iterative Synthesis Loop
        for i in tqdm(range(n_iters), desc="Synthesizing Texture"):
            # a) Decompose current synthesis image
            synth_pyr = synth_pyramid_builder.decompose(synth_image)
            
            # b) Match histograms of all frequency/orientation subbands
            matched_pyr = self._match_pyramids(synth_pyr, target_pyr)
            
            # c) Reconstruct the image from the modified subbands
            synth_image = synth_pyramid_builder.reconstruct(matched_pyr)
            
            # d) Enforce the global pixel histogram again
            synth_image = match_histogram(synth_image, target_image)
            
        return synth_image