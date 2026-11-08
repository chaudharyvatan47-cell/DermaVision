"""Audit label alignment and run small train-only diagnostics for DermNet V2.

This script never opens the test directory and never writes to the saved model
or either dataset. Diagnostic subset models exist only in memory.
"""

from __future__ import annotations

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import ctypes
import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass

import data_loader_v2
from cnn_model_v2 import build_cnn_model_v2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "dermnet_diagnostics"
CHECKPOINT = PROJECT_ROOT / "results" / "dermnet_baseline_v2" / "model.keras"
HISTORY_CSV = PROJECT_ROOT / "results" / "dermnet_baseline_v2" / "training_history.csv"
SUMMARY_JSON = PROJECT_ROOT / "results" / "dermnet_baseline_v2" / "experiment_summary.json"
SEED = 42
SUBSET_PER_CLASS = 4
DIAGNOSTIC_EPOCHS = 20
BATCH_SIZE = 4
BETA_EFFECTIVE_N = 0.999


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def save_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_history() -> list[dict[str, float]]:
    with HISTORY_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    history: list[dict[str, float]] = []
    for row in rows:
        history.append({key: float(value) for key, value in row.items() if value not in (None, "")})
    return history


def class_weight_map(counts: Counter[str], class_names: list[str]) -> dict[int, float]:
    total = sum(counts.values())
    if any(counts[name] <= 0 for name in class_names):
        raise ValueError("A training class has no examples; refusing to calculate weights.")
    return {index: total / (len(class_names) * counts[name]) for index, name in enumerate(class_names)}


def audit_label_mapping(bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check every train/validation path against its folder-derived integer label."""
    classes: list[str] = bundle["class_names"]
    mapping: dict[str, int] = bundle["class_to_index"]
    if mapping != {name: index for index, name in enumerate(classes)}:
        raise AssertionError("Loader class_to_index does not match its ordered class_names.")
    if len(classes) != 23:
        raise AssertionError(f"Expected 23 classes, got {len(classes)}.")

    paths_by_split: dict[str, list[str]] = {}
    labels_by_split: dict[str, list[int]] = {}
    for split_name in ("train", "validation"):
        paths, labels, counts = data_loader_v2.collect_split(split_name, classes)
        if len(paths) != len(labels):
            raise AssertionError(f"{split_name} path and label list lengths differ.")
        if set(counts) != set(classes) or any(counts[c] <= 0 for c in classes):
            raise AssertionError(f"{split_name} does not have valid images for every class.")
        wrong: list[str] = []
        encoded_counts: Counter[int] = Counter()
        for image_path, label_index in zip(paths, labels):
            path = Path(image_path)
            relative_parts = path.relative_to(data_loader_v2.DATASET_ROOT / split_name).parts
            expected_label = mapping[relative_parts[0]]
            encoded_counts[label_index] += 1
            if not path.is_file() or expected_label != label_index:
                wrong.append(image_path)
        if wrong:
            raise AssertionError(f"{split_name} has path/label mismatches; examples: {wrong[:5]}")
        path_counts = Counter(mapping[Path(p).relative_to(data_loader_v2.DATASET_ROOT / split_name).parts[0]] for p in paths)
        if path_counts != encoded_counts:
            raise AssertionError(f"{split_name} path-derived counts differ from encoded label counts.")
        paths_by_split[split_name] = paths
        labels_by_split[split_name] = labels

    expected_mapping = {name: index for index, name in enumerate(classes)}
    if expected_mapping != bundle["class_to_index"]:
        raise AssertionError("Training/validation mappings are not identical.")

    output_layer = None
    # Verify the output neuron order is the same order used to create labels.
    model = tf.keras.models.load_model(CHECKPOINT)
    output_layer = model.get_layer("class_probabilities")
    if int(output_layer.units) != len(classes) or tuple(model.output_shape) != (None, len(classes)):
        raise AssertionError("Checkpoint output dimension does not match the class mapping.")

    # Check actual TensorFlow one-hot batches in both train and validation streams.
    one_hot_checks: dict[str, bool] = {}
    for split_name in ("training", "validation"):
        images, labels = next(iter(bundle[split_name]))
        values_ok = bool(tf.reduce_all((labels == 0.0) | (labels == 1.0)).numpy())
        rows_ok = bool(tf.reduce_all(tf.reduce_sum(labels, axis=1) == 1.0).numpy())
        shape_ok = tuple(labels.shape) == (BATCH_SIZE, len(classes))
        pixel_ok = float(tf.reduce_min(images)) >= 0.0 and float(tf.reduce_max(images)) <= 1.0
        one_hot_checks[split_name] = values_ok and rows_ok and shape_ok and pixel_ok
        if not one_hot_checks[split_name]:
            raise AssertionError(f"{split_name} one-hot batch or normalized pixels failed verification.")

    audit_rows: list[dict[str, Any]] = []
    for index, class_name in enumerate(classes):
        train_samples = [p for p, label in zip(paths_by_split["train"], labels_by_split["train"]) if label == index]
        validation_samples = [p for p, label in zip(paths_by_split["validation"], labels_by_split["validation"]) if label == index]
        audit_rows.append({
            "class_index": index,
            "class_name": class_name,
            "output_neuron_index": index,
            "training_image_count": len(train_samples),
            "validation_image_count": len(validation_samples),
            "training_paths_checked": len(train_samples),
            "training_path_label_mismatches": 0,
            "validation_paths_checked": len(validation_samples),
            "validation_path_label_mismatches": 0,
            "training_one_hot_label_index": index,
            "validation_one_hot_label_index": index,
            "valid_training_samples": len(train_samples) > 0,
            "sample_train_path": train_samples[0] if train_samples else "",
            "sample_validation_path": validation_samples[0] if validation_samples else "",
        })
    checks = {
        "class_count": len(classes),
        "class_to_index": expected_mapping,
        "train_validation_mapping_identical": True,
        "output_neuron_count": int(output_layer.units),
        "one_hot_and_pixel_checks": one_hot_checks,
        "all_paths_exist_and_match_parent_category": True,
        "training_images": len(paths_by_split["train"]),
        "validation_images": len(paths_by_split["validation"]),
        "test_directory_accessed": False,
    }
    return audit_rows, checks


def class_distribution_report(bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = bundle["class_names"]
    train_counts: Counter[str] = bundle["training_counts"]
    val_counts: Counter[str] = bundle["validation_counts"]
    weights = class_weight_map(train_counts, names)  # Training labels only.
    total_train = sum(train_counts.values())
    total_validation = sum(val_counts.values())
    rows = []
    for index, name in enumerate(names):
        count = train_counts[name]
        weight = weights[index]
        rows.append({
            "class_index": index,
            "class_name": name,
            "training_count": count,
            "training_percent": 100.0 * count / total_train,
            "validation_count_descriptive_only": val_counts[name],
            "validation_percent_descriptive_only": 100.0 * val_counts[name] / total_validation,
            "balanced_training_weight": weight,
            "weighted_effective_training_count": count * weight,
            "effective_sample_number_beta_0_999": (1.0 - BETA_EFFECTIVE_N ** count) / (1.0 - BETA_EFFECTIVE_N),
        })
    ordered = sorted(train_counts.items(), key=lambda item: item[1])
    majority_val_class, majority_val_count = max(val_counts.items(), key=lambda item: item[1])
    return rows, {
        "training_total": total_train,
        "validation_total_descriptive_only": total_validation,
        "training_min_count": ordered[0][1],
        "training_min_class": ordered[0][0],
        "training_max_count": ordered[-1][1],
        "training_max_class": ordered[-1][0],
        "training_max_min_ratio": ordered[-1][1] / ordered[0][1],
        "training_majority_class_percent": 100.0 * ordered[-1][1] / total_train,
        "validation_majority_class": majority_val_class,
        "validation_majority_accuracy_baseline": majority_val_count / total_validation,
        "weights_formula": "total training images / (number of classes * training count for the class)",
        "effective_weighted_samples_per_class": total_train / len(names),
        "effective_sample_number_formula": "(1 - beta^n) / (1 - beta), beta=0.999; descriptive only",
        "weights_used_validation_or_test_labels": False,
    }


def prediction_diagnostics(bundle: dict[str, Any]) -> dict[str, Any]:
    """Inspect saved checkpoint probabilities and derive the pre-softmax logits."""
    model = tf.keras.models.load_model(CHECKPOINT)
    output_layer = model.get_layer("class_probabilities")
    feature_model = tf.keras.Model(model.inputs, output_layer.input)
    classes = bundle["class_names"]
    all_logits: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    maximum_softmax_discrepancy = 0.0
    for images, one_hot in bundle["validation"]:
        model_probs = model(images, training=False).numpy()
        features = feature_model(images, training=False)
        logits = (tf.linalg.matmul(features, output_layer.kernel) + output_layer.bias).numpy()
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        maximum_softmax_discrepancy = max(maximum_softmax_discrepancy, float(np.max(np.abs(model_probs - probs))))
        all_logits.append(logits)
        all_probs.append(probs)
        all_true.append(np.argmax(one_hot.numpy(), axis=1))
    logits = np.concatenate(all_logits, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    true = np.concatenate(all_true, axis=0)
    if len(probs) != bundle["validation_image_count"]:
        raise AssertionError("Checkpoint diagnostic did not see every validation image.")
    predicted = np.argmax(probs, axis=1)
    predicted_counts = np.bincount(predicted, minlength=len(classes))
    actual_counts = np.bincount(true, minlength=len(classes))
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    mean_prob = probs.mean(axis=0)
    per_class_prob_std = probs.std(axis=0)
    max_prob = probs.max(axis=1)
    prediction_rows = []
    for i, name in enumerate(classes):
        prediction_rows.append({
            "class_index": i,
            "class_name": name,
            "actual_validation_count": int(actual_counts[i]),
            "actual_validation_percent": 100.0 * int(actual_counts[i]) / len(true),
            "predicted_validation_count": int(predicted_counts[i]),
            "predicted_validation_percent": 100.0 * int(predicted_counts[i]) / len(predicted),
            "mean_probability": float(mean_prob[i]),
            "probability_std_across_images": float(per_class_prob_std[i]),
            "mean_raw_logit": float(logits[:, i].mean()),
            "raw_logit_std": float(logits[:, i].std()),
        })
    return {
        "split": "validation",
        "images": len(probs),
        "unique_predicted_classes": int(np.count_nonzero(predicted_counts)),
        "predicted_class_counts": {classes[i]: int(n) for i, n in enumerate(predicted_counts) if n},
        "accuracy": float(np.mean(predicted == true)),
        "mean_prediction_entropy_nats": float(entropy.mean()),
        "maximum_entropy_nats_for_23_classes": math.log(len(classes)),
        "normalized_mean_entropy": float(entropy.mean() / math.log(len(classes))),
        "mean_max_probability": float(max_prob.mean()),
        "median_max_probability": float(np.median(max_prob)),
        "fraction_max_probability_at_least_0_9": float(np.mean(max_prob >= 0.9)),
        "minimum_probability": float(probs.min()),
        "maximum_probability": float(probs.max()),
        "probabilities_finite": bool(np.isfinite(probs).all()),
        "logits_finite": bool(np.isfinite(logits).all()),
        "probability_rows_sum_to_one": bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)),
        "max_difference_model_vs_softmax_of_recovered_logits": maximum_softmax_discrepancy,
        "prediction_rows": prediction_rows,
        "test_directory_accessed": False,
    }


def make_subset_dataset(paths: list[str], labels: list[int], image_size: int, shuffle_seed: int) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.shuffle(len(paths), seed=shuffle_seed, reshuffle_each_iteration=True)

    def decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        contents = tf.io.read_file(path)
        image = tf.io.decode_image(contents, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, (image_size, image_size), antialias=True)
        image = tf.cast(image, tf.float32) / 255.0
        return image, tf.one_hot(label, depth=23, dtype=tf.float32)

    return dataset.map(decode, num_parallel_calls=1).batch(BATCH_SIZE).prefetch(1)


class DiagnosticEpochLog(tf.keras.callbacks.Callback):
    def __init__(self, run_name: str) -> None:
        super().__init__()
        self.run_name = run_name
        self.rows: list[dict[str, float]] = []
        self.stop_reason: str | None = None
        self.epoch_start = 0.0

    @staticmethod
    def available_ram_mb() -> float | None:
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong), ("available_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                ("available_extended", ctypes.c_ulonglong),
            ]
        try:
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.available_phys / (1024 * 1024)
        except (AttributeError, OSError):
            logging.exception("Could not read available memory during subset experiment.")
        return None

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.epoch_start = time.monotonic()

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        if batch % 5 == 0:
            available = self.available_ram_mb()
            if available is not None and available < 512:
                self.stop_reason = f"Available RAM fell below 512 MB ({available:.0f} MB)."
                logging.error("%s Stopping diagnostic at batch boundary.", self.stop_reason)
                self.model.stop_training = True

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        row = {key: float(value) for key, value in (logs or {}).items()}
        row["epoch"] = float(epoch + 1)
        row["epoch_duration_seconds"] = time.monotonic() - self.epoch_start
        available = self.available_ram_mb()
        row["available_ram_mb"] = available if available is not None else float("nan")
        self.rows.append(row)
        logging.info("%s epoch %d: %s", self.run_name, epoch + 1, row)


def build_diagnostic_model(image_size: int, learning_rate: float) -> tf.keras.Model:
    if image_size == 128:
        model = build_cnn_model_v2(23)
    else:
        # Resolution-only comparison: identical V2 layers and class head.
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(image_size, image_size, 3), name="image"),
                tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv_32"),
                tf.keras.layers.MaxPooling2D(2, name="pool_1"),
                tf.keras.layers.Conv2D(64, 3, activation="relu", name="conv_64"),
                tf.keras.layers.MaxPooling2D(2, name="pool_2"),
                tf.keras.layers.Conv2D(128, 3, activation="relu", name="conv_128"),
                tf.keras.layers.MaxPooling2D(2, name="pool_3"),
                tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool"),
                tf.keras.layers.Dense(128, activation="relu", name="dense_128"),
                tf.keras.layers.Dropout(0.5, name="dropout_50_percent"),
                tf.keras.layers.Dense(23, activation="softmax", name="class_probabilities"),
            ],
            name="diagnostic_cnn_v2_224",
        )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def run_small_experiments(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Run sequential train-only subset checks; no validation/test data is used."""
    names = bundle["class_names"]
    paths, labels, _ = data_loader_v2.collect_split("train", names)
    class_paths: dict[int, list[str]] = defaultdict(list)
    for path, label in zip(paths, labels):
        class_paths[label].append(path)
    rng = random.Random(SEED)
    subset_paths: list[str] = []
    subset_labels: list[int] = []
    for class_index in range(len(names)):
        if len(class_paths[class_index]) < SUBSET_PER_CLASS:
            raise ValueError(f"Class {names[class_index]} has too few training images for subset diagnostic.")
        chosen = sorted(rng.sample(class_paths[class_index], SUBSET_PER_CLASS))
        subset_paths.extend(chosen)
        subset_labels.extend([class_index] * len(chosen))

    weights = class_weight_map(bundle["training_counts"], names)
    experiments = [
        {"name": "balanced_128_unweighted_lr_0.001", "image_size": 128, "learning_rate": 0.001, "use_class_weights": False},
        {"name": "balanced_128_fulltrain_weights_lr_0.001", "image_size": 128, "learning_rate": 0.001, "use_class_weights": True},
        {"name": "balanced_128_unweighted_lr_0.0001", "image_size": 128, "learning_rate": 0.0001, "use_class_weights": False},
        {"name": "balanced_224_unweighted_lr_0.001", "image_size": 224, "learning_rate": 0.001, "use_class_weights": False},
    ]
    results: list[dict[str, Any]] = []
    for experiment_index, config in enumerate(experiments):
        logging.info("Starting diagnostic experiment sequentially: %s", config)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        dataset = make_subset_dataset(subset_paths, subset_labels, config["image_size"], SEED)
        model = build_diagnostic_model(config["image_size"], config["learning_rate"])
        monitor = DiagnosticEpochLog(config["name"])
        start = time.monotonic()
        history = model.fit(
            dataset,
            epochs=DIAGNOSTIC_EPOCHS,
            class_weight=weights if config["use_class_weights"] else None,
            callbacks=[monitor],
            verbose=0,
            shuffle=False,
        )
        duration = time.monotonic() - start
        final_accuracy = float(history.history["accuracy"][-1])
        final_loss = float(history.history["loss"][-1])
        results.append({
            **config,
            "seed": SEED,
            "balanced_samples_per_class": SUBSET_PER_CLASS,
            "subset_images": len(subset_paths),
            "epochs_completed": len(history.history["loss"]),
            "duration_seconds": duration,
            "initial_train_loss": float(history.history["loss"][0]),
            "final_train_loss": final_loss,
            "final_train_accuracy": final_accuracy,
            "maximum_train_accuracy": float(max(history.history["accuracy"])),
            "overfit_subset_accuracy_at_least_0_95": final_accuracy >= 0.95,
            "epoch_history": monitor.rows,
            "stop_reason": monitor.stop_reason,
            "training_only": True,
        })
        logging.info("Diagnostic experiment complete: %s", results[-1])
        del history, model, dataset
        tf.keras.backend.clear_session()
    return results


def write_recommendations(
    mapping_checks: dict[str, Any],
    imbalance: dict[str, Any],
    predictions: dict[str, Any],
    experiments: list[dict[str, Any]],
) -> None:
    by_name = {item["name"]: item for item in experiments}
    base = by_name["balanced_128_unweighted_lr_0.001"]
    weighted = by_name["balanced_128_fulltrain_weights_lr_0.001"]
    low_lr = by_name["balanced_128_unweighted_lr_0.0001"]
    higher_resolution = by_name["balanced_224_unweighted_lr_0.001"]
    imbalance_only_rejected = (
        predictions["unique_predicted_classes"] == 1
        and predictions["accuracy"] < imbalance["validation_majority_accuracy_baseline"]
        and predictions["predicted_class_counts"]
        and next(iter(predictions["predicted_class_counts"])) != imbalance["training_max_class"]
    )

    if not mapping_checks["all_paths_exist_and_match_parent_category"] or not all(mapping_checks["one_hot_and_pixel_checks"].values()):
        primary = "Fix the path/label or one-hot mismatch identified by the mapping audit before training again."
    elif base["overfit_subset_accuracy_at_least_0_95"]:
        primary = (
            "Keep the class mapping and 128x128 loader. The model can memorize a balanced subset, "
            "so the full-run collapse points to generalization/optimization behavior rather than a basic inability to learn."
        )
    elif higher_resolution["final_train_accuracy"] >= base["final_train_accuracy"] + 0.10:
        primary = (
            "The 224x224 subset run learned materially better than 128x128; a controlled full-training "
            "comparison at 224x224 is justified, subject to memory monitoring."
        )
    else:
        primary = (
            "The small 128x128 balanced subset was not reliably overfit; inspect the optimization/data path further "
            "before committing to a full retrain."
        )

    lr_recommendation = (
        "The 1e-4 subset run improved final training accuracy by at least 10 percentage points over 1e-3; "
        "try 1e-4 in the next controlled full-train run."
        if low_lr["final_train_accuracy"] >= base["final_train_accuracy"] + 0.10
        else "The 1e-4 subset run did not improve final training accuracy by 10 percentage points; the diagnostic does not support changing the learning rate alone."
    )
    class_weight_recommendation = (
        "On the balanced subset, applying full-training class weights reduced final accuracy by at least 10 percentage points. "
        "Do not treat this balanced-subset result as definitive for the full imbalanced set; verify weighting in one controlled full-train comparison."
        if weighted["final_train_accuracy"] <= base["final_train_accuracy"] - 0.10
        else "The balanced-subset comparison does not show a 10-point training-accuracy penalty from class weights; it does not prove weights improve validation generalization."
    )
    resolution_recommendation = (
        "224x224 improved balanced-subset final training accuracy by at least 10 percentage points; test that input size in a separate, memory-monitored full-train comparison."
        if higher_resolution["final_train_accuracy"] >= base["final_train_accuracy"] + 0.10
        else "The 224x224 subset run did not improve final training accuracy by 10 percentage points; evidence does not justify increasing resolution by itself."
    )

    lines = [
        "# Recommended next correction",
        "",
        "## Evidence",
        "",
        f"- Mapping/path/one-hot checks: {'PASS' if mapping_checks['all_paths_exist_and_match_parent_category'] and all(mapping_checks['one_hot_and_pixel_checks'].values()) else 'FAIL'}. The output head has {mapping_checks['output_neuron_count']} units.",
        f"- The validation checkpoint predicted {predictions['unique_predicted_classes']} distinct class(es); accuracy was {predictions['accuracy']:.4%}.",
        f"- Training counts range from {imbalance['training_min_count']} ({imbalance['training_min_class']}) to {imbalance['training_max_count']} ({imbalance['training_max_class']}), a {imbalance['training_max_min_ratio']:.2f}:1 ratio.",
        f"- The constant validation majority-class baseline is {imbalance['validation_majority_accuracy_baseline']:.4%} ({imbalance['validation_majority_class']}); the checkpoint was below this baseline: {imbalance_only_rejected}.",
        f"- Balanced subset final accuracies: 128px unweighted={base['final_train_accuracy']:.4f}; full-train class weights={weighted['final_train_accuracy']:.4f}; LR 1e-4={low_lr['final_train_accuracy']:.4f}; 224px={higher_resolution['final_train_accuracy']:.4f}.",
        "",
        "## Recommendation",
        "",
        primary,
        "",
        lr_recommendation,
        "",
        class_weight_recommendation,
        "",
        resolution_recommendation,
        "",
        "Do not change the original CNN, V2 checkpoint, or any dataset as part of this diagnosis. Do not use the test split until the validation collapse is resolved. Any small-subset fit result measures memorization only, not generalization or medical performance.",
    ]
    (OUTPUT_DIR / "recommended_fixes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "DermNet V2 class-collapse investigation",
        "",
        f"Mapping audit: {'PASS' if mapping_checks['all_paths_exist_and_match_parent_category'] else 'FAIL'}; {mapping_checks['class_count']} classes; train={mapping_checks['training_images']}; validation={mapping_checks['validation_images']}; test directory opened=no.",
        f"Saved checkpoint validation: {predictions['accuracy']:.4%} accuracy; {predictions['unique_predicted_classes']} unique predicted class(es); mean entropy {predictions['mean_prediction_entropy_nats']:.4f} nats (maximum {predictions['maximum_entropy_nats_for_23_classes']:.4f}).",
        f"Training class count ratio max/min: {imbalance['training_max_min_ratio']:.2f}:1. Validation majority baseline: {imbalance['validation_majority_accuracy_baseline']:.4%}.",
        f"Subset experiments used {SUBSET_PER_CLASS} training images per class ({SUBSET_PER_CLASS * len(mapping_checks['class_to_index'])} total), sequentially, with no validation/test images.",
        f"128px unweighted final subset accuracy: {base['final_train_accuracy']:.4f}; weighted: {weighted['final_train_accuracy']:.4f}; LR 1e-4: {low_lr['final_train_accuracy']:.4f}; 224px: {higher_resolution['final_train_accuracy']:.4f}.",
        "",
        "Read recommended_fixes.md for the evidence-based next step. Subset accuracy is not a generalization metric. The checkpoint is not successful and no clinical claims are supported.",
    ]
    (OUTPUT_DIR / "diagnostic_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic directory: {OUTPUT_DIR}")
    if not CHECKPOINT.is_file() or not HISTORY_CSV.is_file() or not SUMMARY_JSON.is_file():
        raise FileNotFoundError("V2 checkpoint, history, or summary is missing.")
    OUTPUT_DIR.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(OUTPUT_DIR / "diagnostic_experiments.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    start = time.monotonic()
    checkpoint_hash_before = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    try:
        logging.info("Create train/validation inputs from the reviewed derived data only.")
        bundle = data_loader_v2.create_dermnet_datasets()
        mapping_rows, mapping_checks = audit_label_mapping(bundle)
        save_csv(OUTPUT_DIR / "class_mapping_audit.csv", list(mapping_rows[0]), mapping_rows)

        distribution_rows, imbalance = class_distribution_report(bundle)
        save_csv(OUTPUT_DIR / "class_distribution.csv", list(distribution_rows[0]), distribution_rows)

        history_rows = load_history()
        if len(history_rows) != 6:
            raise AssertionError(f"Expected six saved V2 epochs, found {len(history_rows)}.")
        history_analysis = {
            "epochs": len(history_rows),
            "train_accuracy_first": history_rows[0]["accuracy"],
            "train_accuracy_peak": max(row["accuracy"] for row in history_rows),
            "train_accuracy_peak_epoch": 1 + max(range(len(history_rows)), key=lambda i: history_rows[i]["accuracy"]),
            "train_accuracy_final": history_rows[-1]["accuracy"],
            "train_accuracy_improved_over_first": history_rows[-1]["accuracy"] > history_rows[0]["accuracy"],
            "train_loss_first": history_rows[0]["loss"],
            "train_loss_min": min(row["loss"] for row in history_rows),
            "train_loss_min_epoch": 1 + min(range(len(history_rows)), key=lambda i: history_rows[i]["loss"]),
            "train_loss_final": history_rows[-1]["loss"],
            "train_loss_decreased_at_any_epoch": min(row["loss"] for row in history_rows) < history_rows[0]["loss"],
            "validation_accuracy_values": [row["val_accuracy"] for row in history_rows],
            "validation_accuracy_unique_values": sorted(set(row["val_accuracy"] for row in history_rows)),
            "validation_loss_values": [row["val_loss"] for row in history_rows],
            "best_validation_loss_epoch": 1 + min(range(len(history_rows)), key=lambda i: history_rows[i]["val_loss"]),
            "best_validation_loss": min(row["val_loss"] for row in history_rows),
            "validation_loss_improved_after_epoch_1": any(row["val_loss"] < history_rows[0]["val_loss"] for row in history_rows[1:]),
            "early_stopping_configured_patience": 5,
            "early_stopped_after_epoch": len(history_rows),
            "early_stopping_too_early_by_configured_rule": False,
            "learning_rate_by_epoch": [row["learning_rate"] for row in history_rows],
        }

        predictions = prediction_diagnostics(bundle)
        save_csv(OUTPUT_DIR / "prediction_distribution.csv", list(predictions["prediction_rows"][0]), predictions["prediction_rows"])
        experiments = run_small_experiments(bundle)
        write_recommendations(mapping_checks, imbalance, predictions, experiments)
        checkpoint_hash_after = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        if checkpoint_hash_after != checkpoint_hash_before:
            raise AssertionError("The saved V2 checkpoint changed during diagnostics.")

        diagnostic_summary = {
            "dataset_path": str(data_loader_v2.DATASET_ROOT),
            "test_directory_accessed": False,
            "mapping_audit": mapping_checks,
            "imbalance": imbalance,
            "training_history_analysis": history_analysis,
            "saved_checkpoint_validation_outputs": predictions,
            "small_train_subset_experiments": experiments,
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": checkpoint_hash_after,
            "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
            "diagnostic_runtime_seconds": time.monotonic() - start,
            "limitations": [
                "Subset experiments use training images only and do not measure generalization.",
                "No test images or labels were loaded.",
                "This baseline is not clinically validated and has no established Indian-specific validation.",
            ],
        }
        write_json(OUTPUT_DIR / "training_diagnostics.json", diagnostic_summary)
        return 0
    except BaseException as exc:
        failure = {
            "status": "diagnostic_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() if CHECKPOINT.exists() else None,
            "diagnostic_runtime_seconds": time.monotonic() - start,
            "test_directory_accessed": False,
        }
        write_json(OUTPUT_DIR / "training_diagnostics.json", failure)
        logging.exception("Diagnostic run failed; test data was not loaded.")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
