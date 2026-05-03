import numpy as np
from src.histogram import match_histogram


class TextureSynthesizer:
    def __init__(self, target_image, n_scales=4, n_orient=4, n_iters=5):
        """
        Parameters
        ----------
        target_image : np.ndarray
            Target image in PCA space (or RGB depending on pipeline)
        """
        self.target_image = target_image
        self.n_scales = n_scales
        self.n_orient = n_orient
        self.n_iters = n_iters

        # Initialize synthesized image (noise or copy)
        self.image = self._initialize_image()

    def _initialize_image(self):
        """
        Initialize the synthesized image.
        Usually Gaussian noise with same shape as target.
        """
        return np.random.randn(*self.target_image.shape).astype(np.float32)

    def _update(self, image):
        """
        One iteration update step.
        Replace this with your actual pyramid / filtering logic.
        """
        # 🔧 Placeholder: you likely already have real logic here
        # For now, just return image unchanged
        return image

    def synthesize(self):
        """
        Run texture synthesis and apply histogram matching at the end.
        """
        print("[*] Starting synthesis...")

        # 🔁 Iterative synthesis
        for i in range(self.n_iters):
            print(f"[*] Iteration {i+1}/{self.n_iters}")
            self.image = self._update(self.image)

        print("[*] Applying final histogram matching...")

        # ✅ FINAL STEP: histogram matching (ONLY ONCE)
        self.image = match_histogram(self.image, self.target_image)

        print("[*] Synthesis complete.")

        return self.image