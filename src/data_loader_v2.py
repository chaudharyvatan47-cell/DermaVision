"""Low-memory TensorFlow input pipeline for the reviewed DermNet dataset.

Only train and validation folders are read. File paths and integer labels are
held in memory, while image bytes are decoded one batch at a time. No cache is
used; decoding is single-threaded; only one batch is prefetched.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import tensorflow as tf


IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4
SEED = 42
PREFETCH_BATCHES = 1
PARALLEL_CALLS = 1
SUPPORTED_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "derived" / "DermNet_review_v1"
SUMMARY_PATH = PROJECT_ROOT / "results" / "duplicate_review" / "final_dataset_summary.json"


def discover_classes(split_dir: Path) -> list[str]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Required split directory is missing: {split_dir}")
    classes = sorted((p.name for p in split_dir.iterdir() if p.is_dir()), key=str.casefold)
    if not classes:
        raise ValueError(f"No category folders found under {split_dir}")
    root_files = [p.name for p in split_dir.iterdir() if p.is_file()]
    if root_files:
        raise ValueError(f"Files outside category folders found in {split_dir}: {root_files[:5]}")
    return classes


def collect_split(split_name: str, class_names: list[str]) -> tuple[list[str], list[int], Counter[str]]:
    """Collect paths and labels, rejecting unsupported files instead of skipping them."""
    root = DATASET_ROOT / split_name
    if not root.is_dir():
        raise FileNotFoundError(f"Required {split_name} split is missing: {root}")
    found_classes = discover_classes(root)
    if found_classes != class_names:
        raise ValueError(f"{split_name} categories differ from training categories.")
    mapping = {name: index for index, name in enumerate(class_names)}
    paths: list[str] = []
    labels: list[int] = []
    counts: Counter[str] = Counter()
    for class_name in class_names:
        category_files = sorted(
            (p for p in (root / class_name).rglob("*") if p.is_file()),
            key=lambda p: str(p).casefold(),
        )
        unsupported = [p for p in category_files if p.suffix.lower() not in SUPPORTED_EXTENSIONS]
        if unsupported:
            raise ValueError(f"Unsupported files in {root / class_name}: {unsupported[:5]}")
        counts[class_name] = len(category_files)
        paths.extend(str(p) for p in category_files)
        labels.extend([mapping[class_name]] * len(category_files))
    if not paths:
        raise ValueError(f"No images found in {root}")
    if any(counts[name] == 0 for name in class_names):
        raise ValueError(f"A category has no images in {root}: {[n for n in class_names if not counts[n]]}")
    return paths, labels, counts


def _decode_resize_normalize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=23, dtype=tf.float32)
    return image, label


def _make_dataset(paths: list[str], labels: list[int], training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    # Shuffle only path/label pairs, not decoded image tensors, to keep RAM use low.
    if training:
        dataset = dataset.shuffle(min(len(paths), 512), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(_decode_resize_normalize, num_parallel_calls=PARALLEL_CALLS, deterministic=not training)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    return dataset.prefetch(PREFETCH_BATCHES)


def create_dermnet_datasets() -> dict[str, Any]:
    """Create separate train/validation datasets and verify them against the final audit."""
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Reviewed DermNet dataset is missing: {DATASET_ROOT}")
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(f"Final derived dataset audit is missing: {SUMMARY_PATH}")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if Path(summary.get("derived_dataset_directory", "")).resolve() != DATASET_ROOT.resolve():
        raise ValueError("Final audit points to a different dataset path.")

    class_names = discover_classes(DATASET_ROOT / "train")
    if len(class_names) != 23:
        raise ValueError(f"Expected 23 DermNet classes; found {len(class_names)}")
    train_paths, train_labels, train_counts = collect_split("train", class_names)
    validation_paths, validation_labels, validation_counts = collect_split("validation", class_names)
    expected = summary.get("derived_class_counts_by_split", {})
    actual = {"train": dict(train_counts), "validation": dict(validation_counts)}
    for split in actual:
        if expected.get(split) != actual[split]:
            raise ValueError(f"{split} per-class image counts do not match the final dataset audit.")
    totals = summary.get("derived_image_count_by_split", {})
    if len(train_paths) != totals.get("train") or len(validation_paths) != totals.get("validation"):
        raise ValueError("Train/validation totals do not match final_dataset_summary.json.")

    train_total = len(train_paths)
    class_weights = {
        index: train_total / (len(class_names) * train_counts[class_name])
        for index, class_name in enumerate(class_names)
    }
    return {
        "training": _make_dataset(train_paths, train_labels, training=True),
        "validation": _make_dataset(validation_paths, validation_labels, training=False),
        "class_names": class_names,
        "class_to_index": {name: index for index, name in enumerate(class_names)},
        "training_counts": train_counts,
        "validation_counts": validation_counts,
        "class_weights": class_weights,
        "training_image_count": train_total,
        "validation_image_count": len(validation_paths),
        "pipeline": {
            "dataset_path": str(DATASET_ROOT), "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE, "parallel_calls": PARALLEL_CALLS,
            "prefetch_batches": PREFETCH_BATCHES, "cache": False,
            "test_split_loaded": False,
        },
    }


def verify_sample_batches(bundle: dict[str, Any]) -> None:
    """Check output shapes, one-hot encoding, and normalized pixels for each input pipeline."""
    for split_name in ("training", "validation"):
        images, labels = next(iter(bundle[split_name]))
        expected_images = (BATCH_SIZE, 128, 128, 3)
        expected_labels = (BATCH_SIZE, 23)
        if tuple(images.shape) != expected_images or tuple(labels.shape) != expected_labels:
            raise ValueError(f"Unexpected {split_name} batch shapes: {images.shape}, {labels.shape}")
        if float(tf.reduce_min(images)) < 0.0 or float(tf.reduce_max(images)) > 1.0:
            raise ValueError(f"{split_name} batch pixel values fall outside [0, 1].")
        if not bool(tf.reduce_all(tf.reduce_sum(labels, axis=1) == 1.0)):
            raise ValueError(f"{split_name} labels are not one-hot encoded.")


if __name__ == "__main__":
    datasets = create_dermnet_datasets()
    verify_sample_batches(datasets)
    print(f"DermNet train={datasets['training_image_count']:,}; validation={datasets['validation_image_count']:,}; classes={len(datasets['class_names'])}")
    print(f"Class mapping: {datasets['class_to_index']}")
    print(f"Input configuration: {datasets['pipeline']}")
    print("Sample batches, one-hot labels, and pixel normalization: PASS. Test split not loaded.")
