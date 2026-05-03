import numpy as np

class PCAColorSpace:
    """
    Handles the transformation of an RGB image into Principal Component space
    to decorrelate the color channels for independent synthesis.
    """
    def __init__(self):
        self.mean = None
        self.eigenvectors = None

    def rgb_to_pca(self, image: np.ndarray) -> np.ndarray:
        """
        Transforms an RGB image to its Principal Component space.
        
        Parameters
        ----------
        image : np.ndarray
            Input image of shape (H, W, 3).
            
        Returns
        -------
        np.ndarray
            Decorrelated PCA image of shape (H, W, 3).
        """
        H, W, C = image.shape
        assert C == 3, "Image must have exactly 3 channels (RGB)"
        
        # Flatten to a list of pixels (N x 3)
        pixels = image.reshape(-1, 3)
        
        # Calculate the mean color and center the data
        self.mean = np.mean(pixels, axis=0)
        centered_pixels = pixels - self.mean
        
        # Calculate the covariance matrix
        cov = np.cov(centered_pixels, rowvar=False)
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvectors = eigenvectors[:, idx]
        
        # Project pixels onto the principal components
        pca_pixels = np.dot(centered_pixels, self.eigenvectors)
        
        return pca_pixels.reshape(H, W, C).astype(np.float32)

    def pca_to_rgb(self, pca_image: np.ndarray) -> np.ndarray:
        """
        Transforms a PCA-space image back to standard RGB.
        
        Parameters
        ----------
        pca_image : np.ndarray
            Synthesized image in PCA space of shape (H, W, 3).
            
        Returns
        -------
        np.ndarray
            Reconstructed RGB image of shape (H, W, 3).
        """
        assert self.eigenvectors is not None, "Must call rgb_to_pca first to calculate eigenvectors!"
        H, W, C = pca_image.shape
        
        pca_pixels = pca_image.reshape(-1, 3)
        
        # Project back to RGB space: X = Y * V^T + mean
        # Since our eigenvectors form an orthogonal matrix, its inverse is its transpose.
        rgb_pixels = np.dot(pca_pixels, self.eigenvectors.T) + self.mean
        
        return rgb_pixels.reshape(H, W, C).astype(np.float32)