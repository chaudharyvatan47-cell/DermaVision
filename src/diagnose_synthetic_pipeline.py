"""Test whether the V2 CNN can learn an easy synthetic 23-class task.

This diagnostic uses generated RGB patterns only. It never reads any dataset,
checkpoint, or test split. Success suggests the model/loss/optimizer loop can
learn a simple mapping; it does not establish that the real dataset pipeline is
correct or that the model generalizes to skin images.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import tensorflow as tf

from cnn_model_v2 import build_cnn_model_v2


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "dermnet_diagnostics" / "synthetic_pipeline_sanity.json"
SEED = 123
CLASSES = 23
SAMPLES_PER_CLASS = 4
IMAGE_SIZE = 128


def main() -> None:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(SEED)

    # Each generated class has a unique RGB color code, repeated in 4 examples.
    # This is deliberately much easier than natural skin photographs.
    colors = np.stack(
        [
            np.linspace(0.05, 0.95, CLASSES, dtype=np.float32),
            np.linspace(0.9, 0.1, CLASSES, dtype=np.float32),
            (np.arange(CLASSES, dtype=np.float32) % 5) / 5.0 + 0.1,
        ],
        axis=1,
    )
    images = np.repeat(colors[:, None, None, :], IMAGE_SIZE, axis=1)
    images = np.repeat(images, IMAGE_SIZE, axis=2)
    images = np.repeat(images, SAMPLES_PER_CLASS, axis=0)
    labels = np.repeat(np.arange(CLASSES, dtype=np.int32), SAMPLES_PER_CLASS)
    one_hot = tf.one_hot(labels, depth=CLASSES)
    dataset = tf.data.Dataset.from_tensor_slices((images, one_hot))
    dataset = dataset.shuffle(len(labels), seed=SEED, reshuffle_each_iteration=True).batch(23)

    model = build_cnn_model_v2(CLASSES)
    before = model.evaluate(dataset, verbose=0, return_dict=True)
    history = model.fit(dataset, epochs=50, verbose=0)
    after = model.evaluate(dataset, verbose=0, return_dict=True)
    report = {
        "data_source": "synthetic generated RGB patterns only",
        "dataset_accessed": False,
        "test_split_accessed": False,
        "checkpoint_loaded_or_modified": False,
        "seed": SEED,
        "classes": CLASSES,
        "samples": int(len(labels)),
        "image_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "initial_loss": float(before["loss"]),
        "initial_accuracy": float(before["accuracy"]),
        "final_loss": float(after["loss"]),
        "final_accuracy": float(after["accuracy"]),
        "epochs": len(history.history["loss"]),
        "result": "PASS" if after["accuracy"] >= 0.90 and after["loss"] < before["loss"] else "FAIL",
        "interpretation": "A pass only checks basic model/optimizer/loss learnability on an artificial task; it is not evidence of skin-image learning or generalization.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
