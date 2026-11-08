"""Build and summarize separate, untrained DermaVision baseline CNNs.

Run from any directory with:
    python src/cnn_model.py

This script only defines, compiles, summarizes, and saves model architectures.
It does not load datasets, use test data, train, evaluate, or claim accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import tensorflow as tf
except Exception as exc:
    raise SystemExit(
        "TensorFlow could not be imported in the active Python environment. "
        "Check that the environment's Python executable and TensorFlow install work. "
        f"Import error: {exc}"
    ) from exc


# These are separate output configurations: their classes and model files are
# never combined. The values match the two dataset directory/metadata labels.
DATASET_CONFIGS = {
    "DermNet": 23,
    "HAM10000": 7,
}


def build_cnn_model(num_classes: int) -> tf.keras.Model:
    """Create and compile the requested baseline CNN for ``num_classes``.

    Categorical crossentropy expects one-hot encoded target labels if this
    model is trained later. This function does not fit or evaluate the model.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least 2.")

    model = tf.keras.Sequential(
        [
            # Input images are RGB with a fixed 224 by 224 size.
            tf.keras.layers.Input(shape=(224, 224, 3), name="image"),

            # Convolutions learn local visual patterns; ReLU adds non-linearity.
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv_32"),
            # Pooling downsamples feature maps and reduces later computation.
            tf.keras.layers.MaxPooling2D((2, 2), name="pool_1"),

            # Deeper convolution layers learn increasingly complex patterns.
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv_64"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool_2"),

            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", name="conv_128"),
            tf.keras.layers.MaxPooling2D((2, 2), name="pool_3"),

            # Flatten turns the final feature maps into a vector for Dense layers.
            tf.keras.layers.Flatten(name="flatten"),
            # Dense combines learned features; dropout randomly regularizes later training.
            tf.keras.layers.Dense(128, activation="relu", name="dense_128"),
            tf.keras.layers.Dropout(0.5, name="dropout_50_percent"),

            # Softmax returns one normalized score per dataset-specific class.
            tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities"),
        ],
        name=f"dermavision_cnn_{num_classes}_classes",
    )

    # Compile configures how training would work later; compile() does not train.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    """Build, print, and save one independent architecture per dataset."""
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    output_paths = {
        dataset_name: models_dir / f"{dataset_name.lower()}_cnn_architecture.json"
        for dataset_name in DATASET_CONFIGS
    }

    # Avoid overwriting files from earlier work without making that explicit.
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Model architecture file(s) already exist: {names}. "
            "Move or rename them before generating new architecture files."
        )

    models: dict[str, tf.keras.Model] = {}
    architectures: dict[str, str] = {}
    for dataset_name, class_count in DATASET_CONFIGS.items():
        print(f"\n{dataset_name}: {class_count} output classes")
        model = build_cnn_model(class_count)
        model.summary()
        models[dataset_name] = model
        architectures[dataset_name] = model.to_json(indent=2)

    # Save the model structures as JSON. No weights are trained or saved.
    models_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, architecture in architectures.items():
        path = output_paths[dataset_name]
        path.write_text(architecture + "\n", encoding="utf-8")
        print(f"Saved {dataset_name} architecture: {path}")

    print("\nBoth models are untrained. No dataset or test split was loaded.")


if __name__ == "__main__":
    main()
