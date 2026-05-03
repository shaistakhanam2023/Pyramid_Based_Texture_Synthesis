import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from src.image_io import load_image, save_image
from src.color import PCAColorSpace
from src.synthesis import TextureSynthesizer


def get_next_filename(input_path, output_dir):
    """
    Generate filename like:
    pink_wall.png → pink_wall_1.png, pink_wall_2.png
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    i = 1
    while True:
        new_filename = f"{base_name}_{i}.png"
        new_path = os.path.join(output_dir, new_filename)

        if not os.path.exists(new_path):
            return new_path

        i += 1


def save_histogram_image(image, output_path, title="Histogram"):
    """
    Save histogram of image (RGB or grayscale)
    """
    plt.figure(figsize=(8, 5))

    if image.ndim == 3:
        colors = ['r', 'g', 'b']
        for i, color in enumerate(colors):
            plt.hist(image[..., i].ravel(), bins=256, color=color, alpha=0.5, label=f'Channel {i}')
    else:
        plt.hist(image.ravel(), bins=256, color='gray', alpha=0.7)

    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


def synthesize_texture(input_path, output_path, out_width, out_height,
                       n_scales=4, n_orient=4, n_iters=5):

    print(f"[*] Loading reference texture from: {input_path}")
    target_image = load_image(input_path)

    # ✅ Save histogram of INPUT image
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    input_hist_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(input_path))[0] + "_input_hist.png"
    )

    save_histogram_image(target_image, input_hist_path, title="Input Image Histogram")

    print(f"[*] Input histogram saved to: {input_hist_path}")

    # 1. PCA transform
    print("[*] Transforming color space (PCA)...")
    pca = PCAColorSpace()
    pca_target = pca.rgb_to_pca(target_image)

    # 2. Initialize synthesizer
    print(f"[*] Initializing Steerable Pyramid (Scales: {n_scales}, Orientations: {n_orient})...")
    synthesizer = TextureSynthesizer(n_scales=n_scales, n_orient=n_orient)

    # 3. Synthesize each channel
    print(f"[*] Starting Synthesis ({n_iters} iterations) for shape ({out_height}, {out_width})...")
    pca_synth = np.zeros((out_height, out_width, 3), dtype=np.float32)

    for c in range(3):
        print(f"\n--- Processing Color Component {c+1}/3 ---")
        pca_synth[..., c] = synthesizer.synthesize_grayscale(
            pca_target[..., c],
            (out_height, out_width),
            n_iters=n_iters
        )

    # 4. Back to RGB
    print("\n[*] Reconstructing RGB color space...")
    synth_image = pca.pca_to_rgb(pca_synth)

    # 5. Save output (input-based naming)
    output_path = get_next_filename(input_path, output_dir)

    save_image(output_path, synth_image)

    print(f"[*] SUCCESS! Synthesized texture saved to: {output_path}")

    # 6. Save histogram of OUTPUT image
    hist_path = output_path.replace(".png", "_hist.png")

    save_histogram_image(synth_image, hist_path, title="Synthesized Histogram")

    print(f"[*] Output histogram saved to: {hist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heeger-Bergen Texture Synthesis")

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to input texture image")

    parser.add_argument("-o", "--output", type=str,
                        default="data/output/output.png",
                        help="Output folder (filename ignored)")

    parser.add_argument("-W", "--width", type=int, default=256,
                        help="Output width")

    parser.add_argument("-H", "--height", type=int, default=256,
                        help="Output height")

    parser.add_argument("-s", "--scales", type=int, default=4,
                        help="Number of pyramid scales")

    parser.add_argument("-r", "--orient", type=int, default=4,
                        help="Number of orientations")

    parser.add_argument("-it", "--iters", type=int, default=5,
                        help="Number of synthesis iterations")

    args = parser.parse_args()

    synthesize_texture(
        args.input,
        args.output,
        args.width,
        args.height,
        args.scales,
        args.orient,
        args.iters
    )