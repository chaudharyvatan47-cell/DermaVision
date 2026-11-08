"""Load separate DermNet and HAM10000 train/validation datasets with Keras.

Run from any directory with:
    python src/data_loader.py

This loader uses only the processed train and validation folders. It never
opens or loads the test split, does not augment images, and does not train or
evaluate a model. Labels are one-hot encoded for CategoricalCrossentropy.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

try:
    import tensorflow as tf
except Exception as exc:
    raise SystemExit(
        "TensorFlow could not be imported. Check the active Python environment "
        "and TensorFlow installation before running this script. "
        f"Import error: {exc}"
    ) from exc


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42

# Match the file formats supported by Keras' directory image loader. If another
# file appears inside a class folder, fail visibly instead of silently skipping it.
SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = PROJECT_ROOT / "dataset" / "processed"
# DermNet now uses the separately reviewed derived copy. HAM10000 remains on
# its existing processed split tree; the datasets stay fully independent.
DATASET_ROOTS = {
    "DermNet": PROJECT_ROOT / "dataset" / "derived" / "DermNet_review_v1",
    "HAM10000": PROCESSED_ROOT / "HAM10000",
}


def list_class_folders(split_directory: Path) -> list[str]:
    """Return a stable class-folder order and reject missing/empty split roots."""
    if not split_directory.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_directory}")
    root_files = [entry for entry in split_directory.iterdir() if entry.is_file()]
    if root_files:
        examples = ", ".join(str(path) for path in root_files[:5])
        raise ValueError(
            f"Files found directly inside the split root {split_directory}; "
            f"Keras would not assign them a class. Examples: {examples}"
        )
    class_names = sorted(
        (entry.name for entry in split_directory.iterdir() if entry.is_dir()),
        key=str.casefold,
    )
    if not class_names:
        raise ValueError(f"No class folders found in {split_directory}")
    return class_names


def count_split_labels(
    split_directory: Path,
    class_names: list[str],
    split_name: str,
) -> Counter[str]:
    """Count a split's folder labels and reject files Keras would skip."""
    counts: Counter[str] = Counter()
    unsupported_files: list[Path] = []
    for class_name in class_names:
        class_directory = split_directory / class_name
        class_files = [path for path in class_directory.rglob("*") if path.is_file()]
        for path in class_files:
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                counts[class_name] += 1
            else:
                unsupported_files.append(path)

    if unsupported_files:
        examples = ", ".join(str(path) for path in unsupported_files[:5])
        raise ValueError(
            f"Unsupported files were found inside {split_name} class folders. "
            f"Keras would ignore them; refusing to continue. Examples: {examples}"
        )
    empty_classes = [name for name in class_names if counts[name] == 0]
    if empty_classes and split_name == "training":
        raise ValueError(f"Training class folders contain no supported images: {empty_classes}")
    return counts


def calculate_class_weights(
    training_counts: Counter[str],
    class_names: list[str],
) -> dict[int, float]:
    """Calculate balanced weights using training labels and no other split."""
    total_training_images = sum(training_counts.values())
    number_of_classes = len(class_names)
    # Balanced weight: total training images / (class count * images in class).
    return {
        class_index: total_training_images / (number_of_classes * training_counts[class_name])
        for class_index, class_name in enumerate(class_names)
    }


def normalize_batch(images: tf.Tensor, one_hot_labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Convert image pixels from 0..255 integers/floats to float32 0..1."""
    images = tf.cast(images, tf.float32) / 255.0
    return images, one_hot_labels


def create_dataset_pair(dataset_name: str, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    """Load one dataset's train and validation splits with identical class indices."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    try:
        dataset_directory = DATASET_ROOTS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected one of {list(DATASET_ROOTS)}") from exc
    training_directory = dataset_directory / "train"
    validation_directory = dataset_directory / "validation"

    # Training folders define the mapping once. Supplying that exact order to
    # validation ensures each one-hot column has the same meaning in both splits.
    class_names = list_class_folders(training_directory)
    validation_class_names = list_class_folders(validation_directory)
    if set(validation_class_names) != set(class_names):
        missing = sorted(set(class_names) - set(validation_class_names), key=str.casefold)
        extra = sorted(set(validation_class_names) - set(class_names), key=str.casefold)
        raise ValueError(
            f"{dataset_name} class-folder mismatch between train and validation. "
            f"Missing in validation: {missing}; unexpected in validation: {extra}"
        )

    # Only the training directories contribute to class counts and weights.
    training_counts = count_split_labels(training_directory, class_names, "training")
    validation_counts = count_split_labels(validation_directory, class_names, "validation")
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    class_weights = calculate_class_weights(training_counts, class_names)

    # labels='inferred' uses the directory name; categorical creates one-hot vectors.
    # Only the train dataset shuffles. Validation keeps a stable order.
    raw_training = tf.keras.utils.image_dataset_from_directory(
        training_directory,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    raw_validation = tf.keras.utils.image_dataset_from_directory(
        validation_directory,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )

    # Fail if Keras' discovered label order differs from the mapping we expose.
    if list(raw_training.class_names) != class_names or list(raw_validation.class_names) != class_names:
        raise ValueError(f"Keras returned an unexpected class order for {dataset_name}.")

    autotune = tf.data.AUTOTUNE
    training_dataset = raw_training.map(normalize_batch, num_parallel_calls=autotune).prefetch(autotune)
    validation_dataset = raw_validation.map(normalize_batch, num_parallel_calls=autotune).prefetch(autotune)

    return {
        "name": dataset_name,
        "batch_size": batch_size,
        "training": training_dataset,
        "validation": validation_dataset,
        "class_names": class_names,
        "class_to_index": class_to_index,
        "training_counts": training_counts,
        "validation_counts": validation_counts,
        "class_weights": class_weights,
        "training_image_count": sum(training_counts.values()),
        # Validation counts are for reporting only and never affect class weights.
        "validation_image_count": sum(validation_counts.values()),
    }


def print_dataset_summary(dataset: dict[str, Any]) -> None:
    """Read one train and validation batch and print the requested checks."""
    train_images, train_labels = next(iter(dataset["training"]))
    validation_images, validation_labels = next(iter(dataset["validation"]))

    expected_image_shape = (BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    expected_label_shape = (BATCH_SIZE, len(dataset["class_names"]))
    for split_name, images, labels in (
        ("training", train_images, train_labels),
        ("validation", validation_images, validation_labels),
    ):
        if tuple(images.shape) != expected_image_shape:
            raise ValueError(
                f"{dataset['name']} {split_name} image batch has shape {tuple(images.shape)}; "
                f"expected {expected_image_shape}."
            )
        if tuple(labels.shape) != expected_label_shape:
            raise ValueError(
                f"{dataset['name']} {split_name} label batch has shape {tuple(labels.shape)}; "
                f"expected {expected_label_shape}."
            )
        one_hot_values = tf.reduce_all((labels == 0.0) | (labels == 1.0))
        one_hot_rows = tf.reduce_all(tf.reduce_sum(labels, axis=1) == 1.0)
        if not bool((one_hot_values & one_hot_rows).numpy()):
            raise ValueError(f"{dataset['name']} {split_name} labels are not one-hot encoded.")
        pixel_min = float(tf.reduce_min(images).numpy())
        pixel_max = float(tf.reduce_max(images).numpy())
        if pixel_min < 0.0 or pixel_max > 1.0:
            raise ValueError(
                f"{dataset['name']} {split_name} pixels are outside [0, 1]: "
                f"{pixel_min} to {pixel_max}."
            )

    print(f"\n{dataset['name']}")
    print(f"  Training images: {dataset['training_image_count']:,}")
    print(f"  Validation images: {dataset['validation_image_count']:,}")
    print(f"  Number of classes: {len(dataset['class_names'])}")
    print(f"  Class-to-index mapping: {dataset['class_to_index']}")
    print(f"  Training image batch shape: {tuple(train_images.shape)}")
    print(f"  Training one-hot label batch shape: {tuple(train_labels.shape)}")
    print(f"  Validation image batch shape: {tuple(validation_images.shape)}")
    print(f"  Validation one-hot label batch shape: {tuple(validation_labels.shape)}")
    print("  Batch shape, one-hot label, and pixel-range checks: PASS")
    print(
        "  Training batch pixel range: "
        f"{float(tf.reduce_min(train_images).numpy()):.6f} to "
        f"{float(tf.reduce_max(train_images).numpy()):.6f}"
    )
    print(
        "  Validation batch pixel range: "
        f"{float(tf.reduce_min(validation_images).numpy()):.6f} to "
        f"{float(tf.reduce_max(validation_images).numpy()):.6f}"
    )
    print("  Class weights (computed from training labels only):")
    for class_name, class_index in dataset["class_to_index"].items():
        count = dataset["training_counts"][class_name]
        weight = dataset["class_weights"][class_index]
        print(f"    {class_name} (index {class_index}, train count {count:,}): {weight:.6f}")


def main() -> None:
    """Create and inspect independent train/validation pipelines."""
    # These calls are deliberately separate: DermNet and HAM10000 never share
    # images, labels, class mappings, or class-weight calculations.
    dermnet = create_dataset_pair("DermNet")
    ham10000 = create_dataset_pair("HAM10000")

    print_dataset_summary(dermnet)
    print_dataset_summary(ham10000)
    print("\nNo test images were loaded. No augmentation, training, or evaluation was run.")


if __name__ == "__main__":
    main()
