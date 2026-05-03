import numpy as np

def match_histogram(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Modifies the source array to exactly match the histogram of the target array.
    This works by rank-matching the pixels.
    
    Parameters
    ----------
    source : np.ndarray
        The array (e.g., synthesized image or subband) to be modified.
    target : np.ndarray
        The reference array (e.g., original texture or subband) providing the target histogram.
        
    Returns
    -------
    np.ndarray
        The modified source array with the same shape as the input source.
    """
    orig_shape = source.shape
    
    # Flatten arrays for 1D statistical ranking
    s_flat = source.ravel()
    t_flat = target.ravel()
    
    # Sort target values to get the target distribution
    t_sorted = np.sort(t_flat)
    
    # Get indices that sort the source
    # This tells us the "rank" of every pixel in the source image
    s_sort_indices = np.argsort(s_flat)
    
    matched = np.empty_like(s_flat)
    
    # Fast path: Sizes match perfectly (e.g., subbands of the same shape)
    if s_flat.size == t_flat.size:
        matched[s_sort_indices] = t_sorted
    else:
        # Sizes differ (e.g., synthesizing a larger texture from a small sample)
        # We interpolate the sorted target CDF to fit the number of pixels in the source
        x_t = np.linspace(0, 1, t_flat.size)
        x_s = np.linspace(0, 1, s_flat.size)
        t_interp = np.interp(x_s, x_t, t_sorted)
        
        matched[s_sort_indices] = t_interp
        
    # Reshape back to original geometry and ensure float32
    return matched.reshape(orig_shape).astype(np.float32)