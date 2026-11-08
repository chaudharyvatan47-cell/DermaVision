"""Memory-efficient 23-class DermNet CNN for 128x128 RGB images."""

from __future__ import annotations

import tensorflow as tf


def build_cnn_model_v2(num_classes: int = 23) -> tf.keras.Model:
    """Build the V2 baseline with GlobalAveragePooling to reduce dense parameters."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least 2.")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(128, 128, 3), name="image"),
            tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv_32"),
            tf.keras.layers.MaxPooling2D(2, name="pool_1"),
            tf.keras.layers.Conv2D(64, 3, activation="relu", name="conv_64"),
            tf.keras.layers.MaxPooling2D(2, name="pool_2"),
            tf.keras.layers.Conv2D(128, 3, activation="relu", name="conv_128"),
            tf.keras.layers.MaxPooling2D(2, name="pool_3"),
            # Average each feature map into one value instead of flattening a
            # large spatial tensor. This greatly reduces Dense-layer memory.
            tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_128"),
            tf.keras.layers.Dropout(0.5, name="dropout_50_percent"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities"),
        ],
        name=f"dermavision_cnn_v2_{num_classes}_classes",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    cnn = build_cnn_model_v2(23)
    cnn.summary()
