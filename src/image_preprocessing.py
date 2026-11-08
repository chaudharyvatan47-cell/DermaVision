"""Demonstrate basic OpenCV preprocessing on one image per dataset.

Run from any directory with:
    python src/image_preprocessing.py

The script reads only one sample from each dataset, keeps the datasets separate,
and writes new processed sample files under results/preprocessing/. It never
changes the source images and does not train a model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "This script requires OpenCV and NumPy. Install them in the active "
        "Python environment with: python -m pip install opencv-python numpy"
    ) from exc


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TARGET_SIZE = (224, 224)  # OpenCV expects (width, height).


def find_sample(dataset_dir: Path) -> Path | None:
    """Choose the first supported image in a stable recursive sort order."""
    if not dataset_dir.is_dir():
        return None
    candidates = sorted(
        (path for path in dataset_dir.rglob("*")
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )
    return candidates[0] if candidates else None


def load_rgb(image_path: Path) -> np.ndarray | None:
    """Decode with OpenCV, then convert its BGR output to RGB for display."""
    # IMREAD_COLOR requests a consistent 3-channel color image. OpenCV returns
    # those channels in BGR order, so convert explicitly to standard RGB order.
    bgr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr_image is None:
        return None
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Resize an RGB image and scale its uint8 pixel values to float [0, 1]."""
    resized_rgb = cv2.resize(image_rgb, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    # Convert before division so the result is floating point, not rounded ints.
    normalized_rgb = resized_rgb.astype(np.float32) / 255.0
    return normalized_rgb


def process_dataset(name: str, dataset_dir: Path, output_dir: Path) -> dict[str, Any] | None:
    """Read, describe, process, and save one sample without touching its source."""
    sample_path = find_sample(dataset_dir)
    if sample_path is None:
        print(f"{name}: no supported image found in {dataset_dir}; skipped.")
        return None

    original_rgb = load_rgb(sample_path)
    if original_rgb is None:
        print(f"{name}: OpenCV could not read sample image {sample_path}; skipped.")
        return None

    original_height, original_width, original_channels = original_rgb.shape
    normalized_rgb = preprocess(original_rgb)

    # Convert normalized RGB floats back to 8-bit, then RGB to BGR because
    # OpenCV's image writer expects BGR channel order for color images.
    save_rgb = np.clip(normalized_rgb * 255.0, 0, 255).round().astype(np.uint8)
    save_bgr = cv2.cvtColor(save_rgb, cv2.COLOR_RGB2BGR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name.lower()}_sample_224x224.png"
    if not cv2.imwrite(str(output_path), save_bgr):
        print(f"{name}: could not save processed image to {output_path}.")
        output_path = None

    print(f"\n{name}")
    print(f"  Sample: {sample_path}")
    print(f"  Original dimensions (width × height): {original_width} × {original_height}")
    print(f"  Original shape: {original_rgb.shape}; channels: {original_channels} (RGB)")
    print(f"  Processed shape: {normalized_rgb.shape}; channels: 3 (RGB)")
    print(f"  Processed pixel range: {normalized_rgb.min():.6f} to {normalized_rgb.max():.6f}")
    if output_path is not None:
        print(f"  Processed image saved: {output_path}")

    return {
        "name": name,
        "path": sample_path,
        "original": original_rgb,
        "processed": normalized_rgb,
        "output": output_path,
    }


def display_samples(samples: list[dict[str, Any]]) -> None:
    """Show original and processed RGB images in a Matplotlib window."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nMatplotlib is not installed; images were saved, but display was skipped.")
        return

    # Each dataset gets its own row, with original and resized/normalized view.
    figure, axes = plt.subplots(len(samples), 2, figsize=(10, 4.5 * len(samples)), squeeze=False)
    for row, sample in enumerate(samples):
        original = sample["original"]
        processed = sample["processed"]
        axes[row, 0].imshow(original)  # RGB order ensures colors are displayed correctly.
        axes[row, 0].set_title(
            f"{sample['name']} original — {original.shape[1]} × {original.shape[0]} px"
        )
        axes[row, 1].imshow(processed, vmin=0.0, vmax=1.0)
        axes[row, 1].set_title(
            f"{sample['name']} processed — {processed.shape[1]} × {processed.shape[0]} px\n"
            f"shape {processed.shape}, range {processed.min():.3f}–{processed.max():.3f}"
        )
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    figure.tight_layout()
    plt.show()


def main() -> None:
    """Find one sample in each dataset and process them independently."""
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dermnet", type=Path, default=Path("dataset/raw/DermNet"),
                        help="DermNet directory; relative paths are from project root")
    parser.add_argument("--ham10000", type=Path, default=Path("dataset/raw/HAM10000"),
                        help="HAM10000 directory; relative paths are from project root")
    parser.add_argument("--output", type=Path, default=Path("results/preprocessing"),
                        help="Directory for processed sample images")
    args = parser.parse_args()

    # Resolve relative options from the project root so the command works from
    # any current working directory, not only when launched at the repo root.
    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    output_dir = resolve(args.output)
    samples = []
    for name, dataset_dir in (
        ("DermNet", resolve(args.dermnet)),
        ("HAM10000", resolve(args.ham10000)),
    ):
        sample = process_dataset(name, dataset_dir, output_dir)
        if sample is not None:
            samples.append(sample)

    if samples:
        display_samples(samples)
    else:
        print("No sample images could be processed.")


if __name__ == "__main__":
    main()
