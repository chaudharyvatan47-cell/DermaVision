"""Train and evaluate DermaVision's first DermNet baseline.

This script runs integrity checks before creating a run directory. It uses the
reviewed derived dataset only, selects the best epoch on validation loss, and
opens test images only for a single final prediction pass after training.
Metrics are computed with NumPy so scikit-learn is not a required dependency.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

import data_loader
from cnn_model import DATASET_CONFIGS, build_cnn_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DERIVED_ROOT = PROJECT_ROOT / "dataset" / "derived" / "DermNet_review_v1"
AUDIT_ROOT = PROJECT_ROOT / "results" / "duplicate_review"
SUMMARY_PATH = AUDIT_ROOT / "final_dataset_summary.json"
EXCLUDED_PATH = AUDIT_ROOT / "excluded_conflicting_groups.csv"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "dermnet_baseline"
SPLITS = ("train", "validation", "test")
SUPPORTED_EXTENSIONS = data_loader.SUPPORTED_IMAGE_EXTENSIONS
SEED = data_loader.RANDOM_SEED
MAX_EPOCHS = 20
TRAINING_BATCH_SIZE = 8
MAX_EPOCH_SECONDS = 300.0  # Stop after a slow epoch, report incomplete, and do not test.


def _image_files(directory: Path) -> list[Path]:
    return sorted((p for p in directory.rglob("*") if p.is_file()), key=lambda p: str(p).casefold())


def verify_derived_dataset() -> tuple[list[str], dict[str, int], dict[str, dict[str, int]]]:
    """Audit every derived image and compare structure/counts to the saved audit."""
    if not DERIVED_ROOT.is_dir() or not SUMMARY_PATH.is_file() or not EXCLUDED_PATH.is_file():
        raise RuntimeError("Critical audit inputs or derived dataset are missing; stopping before training.")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    if Path(summary.get("derived_dataset_directory", "")).resolve() != DERIVED_ROOT.resolve():
        raise RuntimeError("final_dataset_summary.json points to a different derived dataset.")
    if summary.get("derived_cross_split_duplicate_groups") != 0:
        raise RuntimeError("Saved duplicate audit reports cross-split duplicate leakage.")

    class_names = data_loader.list_class_folders(DERIVED_ROOT / "train")
    if len(class_names) != DATASET_CONFIGS["DermNet"]:
        raise RuntimeError(f"Found {len(class_names)} train classes; expected 23.")
    expected_classes = set(class_names)
    excluded_groups: set[str] = set()
    with EXCLUDED_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("label_conflict_status") != "unresolved":
                raise RuntimeError("Excluded-conflict audit contains a non-unresolved row.")
            excluded_groups.add(row.get("group_id", ""))

    split_totals: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {}
    seen_hashes: dict[str, str] = {}
    split_hashes: dict[str, set[str]] = {name: set() for name in SPLITS}
    bad_files: list[str] = []
    for split in SPLITS:
        split_root = DERIVED_ROOT / split
        names = set(data_loader.list_class_folders(split_root))
        if names != expected_classes:
            raise RuntimeError(f"{split} class folders differ from train: missing={sorted(expected_classes-names)}, extra={sorted(names-expected_classes)}")
        class_counts[split] = {}
        total = 0
        for category in class_names:
            folder = split_root / category
            files = _image_files(folder)
            unsupported = [p for p in files if p.suffix.lower() not in SUPPORTED_EXTENSIONS]
            if unsupported:
                raise RuntimeError(f"Unsupported files found in {folder}: {unsupported[:3]}")
            class_counts[split][category] = len(files)
            total += len(files)
            for path in files:
                try:
                    with path.open("rb") as image_file:
                        digest = hashlib.sha256()
                        for block in iter(lambda: image_file.read(1024 * 1024), b""):
                            digest.update(block)
                    group = "sha256:" + digest.hexdigest()
                    with Image.open(path) as image:
                        image.verify()
                    if group in excluded_groups:
                        bad_files.append(f"excluded conflict hash present: {path}")
                    prior_split = seen_hashes.setdefault(group, split)
                    if prior_split != split:
                        bad_files.append(f"duplicate leakage across {prior_split}/{split}: {path}")
                    if group in split_hashes[split]:
                        bad_files.append(f"duplicate copy repeated within {split}: {path}")
                    split_hashes[split].add(group)
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    bad_files.append(f"unreadable image {path}: {exc}")
        split_totals[split] = total

    expected_counts = summary["derived_image_count_by_split"]
    if split_totals != expected_counts:
        raise RuntimeError(f"Derived split counts {split_totals} do not match saved counts {expected_counts}.")
    expected_classes_by_split = summary.get("derived_class_counts_by_split", {})
    for split in SPLITS:
        if expected_classes_by_split.get(split) != class_counts[split]:
            raise RuntimeError(f"Per-class {split} counts do not match the final dataset audit.")
    if sum(split_totals.values()) != summary.get("derived_image_count"):
        raise RuntimeError("Derived total does not match final_dataset_summary.json.")
    if bad_files:
        raise RuntimeError("Dataset integrity failure; no images were skipped:\n" + "\n".join(bad_files[:30]))
    if len(seen_hashes) != sum(split_totals.values()):
        raise RuntimeError("Not every derived image has a unique SHA-256 hash.")
    return class_names, split_totals, class_counts


def verify_loader_and_model(class_names: list[str], split_totals: dict[str, int]) -> dict[str, Any]:
    """Verify the actual train/validation loader and the unchanged 23-way CNN."""
    source = Path(data_loader.__file__).read_text(encoding="utf-8")
    if str(DERIVED_ROOT).casefold() not in str(data_loader.DATASET_ROOTS["DermNet"]).casefold():
        raise RuntimeError("DermNet loader is not configured to use the reviewed derived dataset.")
    if "test" in source.lower() and "image_dataset_from_directory" in source:
        # The test reference may be documentation only; only the two explicit
        # directory-loader calls are checked below via their train/validation args.
        pass
    if data_loader.DATASET_ROOTS["DermNet"].resolve() != DERIVED_ROOT.resolve():
        raise RuntimeError("Loader dataset path mismatch; stopping.")
    bundle = data_loader.create_dataset_pair("DermNet")
    if bundle["class_names"] != class_names:
        raise RuntimeError("Loader class order differs from derived dataset order.")
    for split_key, split_name in (("training", "train"), ("validation", "validation")):
        if bundle[f"{split_key}_image_count"] != split_totals[split_name]:
            raise RuntimeError(f"Loader {split_name} count mismatch.")
    if len(bundle["class_names"]) != 23:
        raise RuntimeError("Loader does not expose all 23 DermNet classes.")
    for key in ("training", "validation"):
        images, labels = next(iter(bundle[key]))
        if tuple(images.shape) != (32, 224, 224, 3) or tuple(labels.shape) != (32, 23):
            raise RuntimeError(f"Loader {key} batch has unexpected image/label shape.")
        if float(tf.reduce_min(images)) < 0 or float(tf.reduce_max(images)) > 1:
            raise RuntimeError(f"Loader {key} pixels fall outside [0, 1].")
        if not bool(tf.reduce_all(tf.reduce_sum(labels, axis=1) == 1.0)):
            raise RuntimeError(f"Loader {key} labels are not one-hot encoded.")
    expected_weights = data_loader.calculate_class_weights(bundle["training_counts"], class_names)
    if expected_weights != bundle["class_weights"]:
        raise RuntimeError("Class weights do not match the training-only balanced formula.")

    if DATASET_CONFIGS.get("DermNet") != 23:
        raise RuntimeError("CNN configuration is not set to 23 DermNet classes.")
    model = build_cnn_model(23)
    if tuple(model.input_shape) != (None, 224, 224, 3) or tuple(model.output_shape) != (None, 23):
        raise RuntimeError("Existing baseline CNN input/output shape is incompatible with the loader.")
    expected_layers = ["Conv2D", "MaxPooling2D", "Conv2D", "MaxPooling2D", "Conv2D", "MaxPooling2D", "Flatten", "Dense", "Dropout", "Dense"]
    actual_layers = [layer.__class__.__name__ for layer in model.layers]
    if actual_layers != expected_layers:
        raise RuntimeError(f"CNN architecture differs from expected baseline: {actual_layers}")
    return bundle


def unique_run_directory() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("experiment_%Y%m%d_%H%M%S")
    candidate = OUTPUT_ROOT / stamp
    suffix = 1
    while candidate.exists():
        candidate = OUTPUT_ROOT / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


class EpochTimer(tf.keras.callbacks.Callback):
    """Record epoch duration and stop after a slow epoch at a safe boundary."""
    def __init__(self, maximum_seconds: float) -> None:
        super().__init__()
        self.maximum_seconds = maximum_seconds
        self.epoch_seconds: list[float] = []
        self.slow_stop = False
        self._started = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self._started = time.monotonic()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        elapsed = time.monotonic() - self._started
        self.epoch_seconds.append(elapsed)
        logging.info("Epoch %d finished in %.1f seconds", epoch + 1, elapsed)
        if elapsed > self.maximum_seconds:
            self.slow_stop = True
            logging.warning("Stopping at epoch boundary: epoch exceeded %.0f seconds.", self.maximum_seconds)
            self.model.stop_training = True

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        # A slow epoch can contain hundreds of batches. Check during the epoch
        # too, so a CPU run can stop at the next batch boundary rather than
        # waiting indefinitely for all training and validation batches.
        elapsed = time.monotonic() - self._started
        if elapsed > self.maximum_seconds:
            self.slow_stop = True
            logging.warning(
                "Requesting a safe stop after training batch %d: epoch exceeded %.0f seconds.",
                batch + 1, self.maximum_seconds,
            )
            self.model.stop_training = True


def save_history(path: Path, history: dict[str, list[float]]) -> None:
    keys = list(history)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for idx in range(len(history.get("loss", []))):
            writer.writerow([idx + 1, *(history[key][idx] for key in keys)])


def save_metric_graphs(run_dir: Path, history: dict[str, list[float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, title, ylabel in (("accuracy", "Accuracy", "Accuracy"), ("loss", "Loss", "Categorical crossentropy")):
        for key, label, filename, color in ((metric, "Training", f"training_{metric}.png", "#2563eb"), (f"val_{metric}", "Validation", f"validation_{metric}.png", "#ea580c")):
            if key not in history:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            epochs = np.arange(1, len(history[key]) + 1)
            ax.plot(epochs, history[key], marker="o", color=color, linewidth=2)
            ax.set(title=f"DermNet {label} {title}", xlabel="Epoch", ylabel=ylabel)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(run_dir / filename, dpi=160)
            plt.close(fig)


def _classification_rows(matrix: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    rows = []
    for idx, label in enumerate(labels):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        support = int(matrix[idx, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": label, "precision": precision, "recall": recall, "f1_score": f1, "support": support})
    return rows


def _save_confusion_matrix(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15, 13))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted label", ylabel="True label", title="DermNet test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=70, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_error_contact_sheet(path: Path, errors: list[dict[str, Any]], max_images: int = 24) -> None:
    chosen = errors[:max_images]
    if not chosen:
        # An empty but valid image records that there were no test errors.
        canvas = Image.new("RGB", (900, 100), "white")
        from PIL import ImageDraw
        ImageDraw.Draw(canvas).text((25, 35), "No misclassified test images", fill="black")
        canvas.save(path)
        return
    from PIL import ImageDraw
    columns, cell_w, cell_h = 4, 300, 250
    rows = (len(chosen) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(chosen):
        image = Image.open(row["filepath"]).convert("RGB")
        image.thumbnail((cell_w - 16, 185))
        x, y = (i % columns) * cell_w, (i // columns) * cell_h
        canvas.paste(image, (x + (cell_w - image.width) // 2, y + 4))
        caption = f"True: {row['true_label']}\nPred: {row['predicted_label']}\n{row['filename']}"
        draw.multiline_text((x + 7, y + 193), caption, fill="black", spacing=2)
    canvas.save(path)


def final_test_pass(model: tf.keras.Model, class_names: list[str], run_dir: Path) -> dict[str, Any]:
    """Read test once and compute every prediction/metric from that one pass."""
    test_dir = DERIVED_ROOT / "test"
    raw = tf.keras.utils.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="categorical", class_names=class_names,
        color_mode="rgb", image_size=(224, 224), batch_size=32, shuffle=False,
    )
    filepaths = list(raw.file_paths)
    all_truth: list[int] = []
    all_pred: list[int] = []
    all_prob: list[np.ndarray] = []
    # One explicit pass through the test batches; no model.evaluate/predict second pass.
    for images, one_hot in raw:
        images = tf.cast(images, tf.float32) / 255.0
        probabilities = model(images, training=False).numpy()
        all_truth.extend(np.argmax(one_hot.numpy(), axis=1).tolist())
        all_pred.extend(np.argmax(probabilities, axis=1).tolist())
        all_prob.extend(probabilities)
    truth = np.asarray(all_truth, dtype=np.int64)
    predicted = np.asarray(all_pred, dtype=np.int64)
    probabilities = np.asarray(all_prob, dtype=np.float32)
    if len(truth) != len(filepaths) or len(truth) != 2122:
        raise RuntimeError(f"Final test pass returned {len(truth)} rows for {len(filepaths)} files; expected 2,122.")

    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    total = len(truth)
    accuracy = float(np.mean(truth == predicted)) if total else 0.0
    rows = _classification_rows(matrix, class_names)
    macro = {key: float(np.mean([row[key] for row in rows])) for key in ("precision", "recall", "f1_score")}
    weights = np.asarray([row["support"] for row in rows], dtype=np.float64)
    weighted = {key: float(np.average([row[key] for row in rows], weights=weights)) for key in ("precision", "recall", "f1_score")}

    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "filepath", "true_label", "predicted_label", "correct", *[f"probability_{i}" for i in range(len(class_names))]])
        for filepath, true_idx, pred_idx, probs in zip(filepaths, truth, predicted, probabilities):
            writer.writerow([Path(filepath).name, filepath, class_names[true_idx], class_names[pred_idx], true_idx == pred_idx, *[float(x) for x in probs]])
    errors = [
        {"filename": Path(filepath).name, "filepath": filepath, "true_label": class_names[t], "predicted_label": class_names[p], "true_index": int(t), "predicted_index": int(p)}
        for filepath, t, p in zip(filepaths, truth, predicted) if t != p
    ]
    with (run_dir / "misclassified_images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "filepath", "true_label", "predicted_label", "true_index", "predicted_index"])
        writer.writeheader()
        writer.writerows(errors)
    with (run_dir / "classification_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        writer.writerows(rows)
        for name, values, support in (("macro avg", macro, total), ("weighted avg", weighted, total)):
            writer.writerow({"class": name, **values, "support": support})
        writer.writerow({"class": "accuracy", "precision": accuracy, "recall": accuracy, "f1_score": accuracy, "support": total})
    _save_confusion_matrix(run_dir / "confusion_matrix.png", matrix, class_names)
    _save_error_contact_sheet(run_dir / "misclassified_contact_sheet.png", errors)
    return {"test_count": total, "accuracy": accuracy, "macro": macro, "weighted": weighted, "per_class": rows, "misclassified_count": len(errors)}


def main() -> int:
    started = time.monotonic()
    print("Phase 1: auditing derived dataset, loader, and CNN before training.", flush=True)
    class_names, split_totals, class_counts = verify_derived_dataset()
    bundle = verify_loader_and_model(class_names, split_totals)
    print(f"PASS: derived counts {split_totals}; {len(class_names)} classes; images readable; duplicate and exclusion checks passed.", flush=True)
    print(f"PASS: loader maps to {DERIVED_ROOT}; batches are 32x224x224x3, labels one-hot, pixels normalized; test excluded.", flush=True)
    print(f"PASS: existing CNN output shape {build_cnn_model(23).output_shape}; architecture unchanged.", flush=True)

    run_dir = unique_run_directory()
    log_path = run_dir / "training.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    hardware = {"python": sys.version, "tensorflow": tf.__version__, "platform": platform.platform(), "devices": [d.name for d in tf.config.list_physical_devices()], "strategy": "CPU" if not tf.config.list_physical_devices("GPU") else "GPU available"}
    print(f"Phase 2: training DermNet baseline in {run_dir}", flush=True)
    print(f"Hardware: {hardware['strategy']} ({hardware['devices']})", flush=True)
    # The loader's default batch size of 32 has already passed shape checks.
    # Rebuild both model-development iterators at 8 after the observed CPU
    # convolution-gradient memory failure, without changing the CNN itself.
    training_bundle = data_loader.create_dataset_pair("DermNet", batch_size=TRAINING_BATCH_SIZE)
    train_ds = training_bundle["training"]
    val_ds = training_bundle["validation"]
    tf.keras.utils.set_random_seed(SEED)
    model = build_cnn_model(len(class_names))
    # Recompile the same baseline layers with the requested explicit learning rate.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
    timer = EpochTimer(MAX_EPOCH_SECONDS)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(run_dir / "model.keras"), monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_history.csv")),
        timer,
    ]
    fit_started = time.monotonic()
    try:
        history_obj = model.fit(train_ds, validation_data=val_ds, epochs=MAX_EPOCHS, class_weight=training_bundle["class_weights"], callbacks=callbacks, verbose=2, shuffle=False)
    except Exception as exc:
        training_seconds = time.monotonic() - fit_started
        failure = {
            "project": "DermaVision", "dataset": "DermNet", "dataset_path": str(DERIVED_ROOT),
            "phase_status": "training_failed", "test_evaluation_performed": False,
            "counts": split_totals, "class_count": len(class_names),
            "model": {"architecture": "existing baseline CNN; Conv2D 32/64/128, pooling, Flatten, Dense(128), Dropout(0.5), 23-way Softmax", "input_shape": [224,224,3], "output_classes": 23, "training_batch_size": TRAINING_BATCH_SIZE, "max_epochs": MAX_EPOCHS},
            "training": {"duration_seconds": training_seconds, "completed_epochs": len(timer.epoch_seconds), "hardware": hardware},
            "failure": f"{type(exc).__name__}: {exc}",
            "limitations": ["No trained model checkpoint was established.", "The test split was not opened.", "This experimental baseline is not clinically validated; no established Indian-specific validation was performed."],
            "outputs_directory": str(run_dir),
        }
        write_json(run_dir / "experiment_summary.json", failure)
        (run_dir / "experiment_summary.txt").write_text(
            "Training failed before a best checkpoint was produced. Test data was not opened.\n"
            f"Failure: {type(exc).__name__}: {exc}\n"
            f"Training time before failure: {training_seconds:.1f} seconds.\n"
            "See training.log for details. No model or evaluation metrics are available.\n",
            encoding="utf-8",
        )
        logging.exception("Training failed; test evaluation was not performed.")
        return 2
    training_seconds = time.monotonic() - fit_started
    history = {key: [float(value) for value in values] for key, values in history_obj.history.items()}
    save_metric_graphs(run_dir, history)
    best_idx = int(np.argmin(history["val_loss"]))
    save_json_path = run_dir / "experiment_summary.json"
    summary: dict[str, Any] = {
        "project": "DermaVision", "dataset": "DermNet", "dataset_path": str(DERIVED_ROOT),
        "counts": split_totals, "class_count": len(class_names), "class_to_index": bundle["class_to_index"],
        "model": {"architecture": "existing baseline CNN; three Conv2D/MaxPooling blocks (32, 64, 128), Flatten, Dense(128), Dropout(0.5), 23-way Softmax", "input_shape": [224,224,3], "output_classes": 23, "optimizer": "Adam", "learning_rate": 0.001, "loss": "CategoricalCrossentropy", "metric": "accuracy", "loader_verification_batch_size": 32, "training_batch_size": TRAINING_BATCH_SIZE, "max_epochs": MAX_EPOCHS},
        "training": {"completed_epochs": len(history.get("loss", [])), "duration_seconds": training_seconds, "mean_epoch_seconds": float(np.mean(timer.epoch_seconds)) if timer.epoch_seconds else None, "early_or_slow_stop": bool(model.stop_training), "slow_epoch_stop": timer.slow_stop, "hardware": hardware},
        "best_validation": {"epoch": best_idx + 1, "loss": history["val_loss"][best_idx], "accuracy": history["val_accuracy"][best_idx]},
        "outputs_directory": str(run_dir), "limitations": ["This is an experimental baseline, not a clinically validated screening system.", "No established Indian-specific validation was performed.", "DermNet category labels and acquisition context may not represent the intended deployment population.", "Metrics describe only this held-out dataset split and do not establish real-world medical accuracy."],
    }
    write_json(save_json_path, summary)
    if timer.slow_stop:
        (run_dir / "experiment_summary.txt").write_text("Training stopped safely at an epoch boundary because CPU training exceeded the configured per-epoch time limit. No test evaluation was run. See experiment_summary.json, training_history.csv, and training.log for progress.\n", encoding="utf-8")
        print("Training stopped safely due to slow CPU epoch; test split was not opened. Partial artifacts and progress are saved.", flush=True)
        return 2
    if not (run_dir / "model.keras").is_file():
        raise RuntimeError("No best checkpoint was produced; refusing to evaluate test data.")

    print("Phase 3: best validation checkpoint saved; opening test split once for final evaluation.", flush=True)
    best_model = tf.keras.models.load_model(run_dir / "model.keras")
    evaluation = final_test_pass(best_model, class_names, run_dir)
    summary["test_evaluation"] = evaluation
    summary["test_evaluation"]["evaluation_passes"] = 1
    write_json(save_json_path, summary)
    text_lines = [
        "DermaVision DermNet CNN baseline experiment", "", f"Dataset: {DERIVED_ROOT}",
        f"Counts: train={split_totals['train']}, validation={split_totals['validation']}, test={split_totals['test']}",
        f"Architecture: existing 23-class baseline CNN; 224x224 RGB input; Conv2D 32/64/128, max pooling, Flatten, Dense 128, Dropout 0.5, Softmax 23.",
        f"Training duration: {training_seconds:.1f} seconds; epochs: {len(history.get('loss', []))}; device: {hardware['strategy']}.",
        f"Best validation epoch: {best_idx+1}; loss={history['val_loss'][best_idx]:.6f}; accuracy={history['val_accuracy'][best_idx]:.6f}.",
        f"Final test (one pass): accuracy={evaluation['accuracy']:.6f}; macro precision={evaluation['macro']['precision']:.6f}; macro recall={evaluation['macro']['recall']:.6f}; macro F1={evaluation['macro']['f1_score']:.6f}.",
        f"Test images: {evaluation['test_count']}; misclassified: {evaluation['misclassified_count']}.",
        "Limitations: experimental baseline; not clinically validated; no established Indian-specific validation; this held-out dataset does not establish real-world medical accuracy.",
        "Per-class precision/recall/F1/support are in classification_report.csv.", f"Artifacts: {run_dir}",
    ]
    (run_dir / "experiment_summary.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print("Phase 4: final test metrics (single pass):", flush=True)
    print(f"accuracy={evaluation['accuracy']:.4f}; macro precision={evaluation['macro']['precision']:.4f}; macro recall={evaluation['macro']['recall']:.4f}; macro F1={evaluation['macro']['f1_score']:.4f}", flush=True)
    print(f"All experiment artifacts saved in {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Experiment failed; test evaluation is not performed unless training/checkpoint selection completed.")
        raise
