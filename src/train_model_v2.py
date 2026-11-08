"""Train the low-memory 128x128 DermNet CNN, without loading test images."""

from __future__ import annotations

# Set conservative CPU runtime settings before TensorFlow is imported.
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import ctypes
import csv
import faulthandler
import json
import logging
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    # This can only happen if another import initialized the runtime first.
    pass

import data_loader_v2
from cnn_model_v2 import build_cnn_model_v2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "dermnet_baseline_v2"
MAX_EPOCHS = 20
MIN_AVAILABLE_RAM_MB = 512
MIN_INITIAL_AVAILABLE_RAM_MB = 1024
MAX_EPOCH_SECONDS = 600
RANDOM_SEED = 42


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def memory_snapshot() -> dict[str, float | None]:
    """Read system available RAM and this process's working set on Windows."""
    result: dict[str, float | None] = {"available_ram_mb": None, "system_ram_mb": None, "process_working_set_mb": None}
    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            result["available_ram_mb"] = status.ullAvailPhys / (1024 * 1024)
            result["system_ram_mb"] = status.ullTotalPhys / (1024 * 1024)
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), ctypes.c_ulong]
        get_memory_info.restype = ctypes.c_int
        if get_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
            result["process_working_set_mb"] = counters.WorkingSetSize / (1024 * 1024)
    except (AttributeError, OSError):
        logging.exception("Could not read Windows memory counters.")
    return result


class ResourceMonitor(tf.keras.callbacks.Callback):
    """Log RAM, CPU, epoch duration, and learning metrics during model.fit."""
    def __init__(self) -> None:
        super().__init__()
        self.epoch_started = 0.0
        self.cpu_started = 0.0
        self.previous_sample_time = 0.0
        self.previous_cpu_time = 0.0
        self.epoch_records: list[dict[str, Any]] = []
        self.stop_reason: str | None = None

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self.epoch_started = time.monotonic()
        self.cpu_started = time.process_time()
        self.previous_sample_time = self.epoch_started
        self.previous_cpu_time = self.cpu_started
        logging.info("Epoch %d started; RAM=%s", epoch + 1, memory_snapshot())

    def on_train_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
        now = time.monotonic()
        if batch % 50 != 0 and now - self.previous_sample_time < 30:
            return
        cpu_now = time.process_time()
        elapsed = max(now - self.previous_sample_time, 1e-6)
        cpu_percent = 100.0 * (cpu_now - self.previous_cpu_time) / elapsed
        metrics = logs or {}
        memory = memory_snapshot()
        logging.info(
            "Batch %d: loss=%s accuracy=%s epoch_elapsed=%.1fs cpu_percent=%.1f memory=%s",
            batch + 1, metrics.get("loss"), metrics.get("accuracy"),
            now - self.epoch_started, cpu_percent, memory,
        )
        self.previous_sample_time = now
        self.previous_cpu_time = cpu_now
        available = memory.get("available_ram_mb")
        if available is not None and available < MIN_AVAILABLE_RAM_MB:
            self.stop_reason = f"Available system RAM fell below {MIN_AVAILABLE_RAM_MB} MB ({available:.0f} MB)."
            logging.error("%s Requesting a safe stop at this batch boundary.", self.stop_reason)
            self.model.stop_training = True
        if now - self.epoch_started > MAX_EPOCH_SECONDS:
            self.stop_reason = f"An epoch exceeded the {MAX_EPOCH_SECONDS}-second safety limit."
            logging.warning("%s Requesting a safe stop at this batch boundary.", self.stop_reason)
            self.model.stop_training = True

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        elapsed = time.monotonic() - self.epoch_started
        cpu_elapsed = time.process_time() - self.cpu_started
        row = dict(logs or {})
        row.update({
            "epoch": epoch + 1,
            "epoch_duration_seconds": elapsed,
            "process_cpu_percent_average": 100.0 * cpu_elapsed / max(elapsed, 1e-6),
            **memory_snapshot(),
        })
        self.epoch_records.append(row)
        logging.info("Epoch %d complete: %s", epoch + 1, row)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def save_graphs(history: dict[str, list[float]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history.get("loss", [])) + 1))
    for metric, title, ylabel, filename in (
        ("accuracy", "DermNet training and validation accuracy", "Accuracy", "accuracy.png"),
        ("loss", "DermNet training and validation loss", "Categorical crossentropy", "loss.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        if metric in history:
            ax.plot(epochs, history[metric], marker="o", label="Training")
        validation_key = f"val_{metric}"
        if validation_key in history:
            ax.plot(epochs, history[validation_key], marker="o", label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=160)
        plt.close(fig)


def validation_prediction_diagnostic(model: tf.keras.Model, bundle: dict[str, Any]) -> dict[str, Any]:
    """Summarize best-checkpoint predictions on validation only; never touch test."""
    actual: list[int] = []
    predicted: list[int] = []
    for images, one_hot_labels in bundle["validation"]:
        probabilities = model(images, training=False).numpy()
        actual.extend(tf.argmax(one_hot_labels, axis=1).numpy().tolist())
        predicted.extend(probabilities.argmax(axis=1).tolist())
    actual_counts = {name: actual.count(index) for index, name in enumerate(bundle["class_names"])}
    predicted_counts = {name: predicted.count(index) for index, name in enumerate(bundle["class_names"])}
    return {
        "split": "validation",
        "images": len(actual),
        "accuracy": sum(a == p for a, p in zip(actual, predicted)) / max(len(actual), 1),
        "actual_class_counts": actual_counts,
        "predicted_class_counts": predicted_counts,
        "single_class_collapse": sum(count > 0 for count in predicted_counts.values()) == 1,
        "test_split_loaded": False,
    }


def run_training() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    log_path = OUTPUT_DIR / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    crash_log = (OUTPUT_DIR / "python_faulthandler.log").open("a", encoding="utf-8")
    faulthandler.enable(file=crash_log, all_threads=True)
    start = time.monotonic()
    fit_start: float | None = None
    bundle: dict[str, Any] | None = None
    monitor = ResourceMonitor()
    history_data: dict[str, list[float]] = {}
    status = "failed"
    error_text: str | None = None
    exit_code = 1
    summary: dict[str, Any] = {
        "project": "DermaVision", "dataset": "DermNet",
        "dataset_path": str(data_loader_v2.DATASET_ROOT),
        "status": "preflight_running", "test_split_loaded": False,
        "configuration": {
            "python": sys.version, "tensorflow": tf.__version__,
            "tensorflow_build": tf.sysconfig.get_build_info(),
            "devices": [device.name for device in tf.config.list_physical_devices()],
            "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            "oneDNN_enabled": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
            "image_size": [128, 128], "batch_size": 4,
            "parallel_map_calls": 1, "prefetch_batches": 1, "cache": False,
            "max_epochs": MAX_EPOCHS,
        },
        "outputs_directory": str(OUTPUT_DIR),
        "limitations": [
            "This is an experimental baseline, not a clinically validated screening system.",
            "No established Indian-specific validation was performed.",
            "No test evaluation is performed by this script.",
        ],
    }
    try:
        logging.info("Phase 1: verify only derived DermNet train and validation inputs.")
        tf.keras.utils.set_random_seed(RANDOM_SEED)
        bundle = data_loader_v2.create_dermnet_datasets()
        data_loader_v2.verify_sample_batches(bundle)
        initial_memory = memory_snapshot()
        summary["dataset_counts"] = {
            "train": bundle["training_image_count"],
            "validation": bundle["validation_image_count"],
            "test_count_from_saved_audit_only": json.loads(data_loader_v2.SUMMARY_PATH.read_text(encoding="utf-8"))["derived_image_count_by_split"]["test"],
            "classes": len(bundle["class_names"]),
        }
        summary["class_to_index"] = bundle["class_to_index"]
        summary["initial_memory"] = initial_memory
        summary["initial_batch_verification"] = {
            "image_shape": [4, 128, 128, 3], "one_hot_label_shape": [4, 23],
            "pixel_range": "[0, 1]", "status": "PASS",
        }
        logging.info("Loader verified: %s", bundle["pipeline"])
        logging.info("Class mapping: %s", bundle["class_to_index"])
        available = initial_memory.get("available_ram_mb")
        if available is not None and available < MIN_INITIAL_AVAILABLE_RAM_MB:
            raise MemoryError(
                f"Only {available:.0f} MB of RAM is available before training; "
                f"the safety minimum is {MIN_INITIAL_AVAILABLE_RAM_MB} MB."
            )

        model = build_cnn_model_v2(23)
        if tuple(model.input_shape) != (None, 128, 128, 3) or tuple(model.output_shape) != (None, 23):
            raise ValueError(f"Unexpected V2 model shapes: {model.input_shape}, {model.output_shape}")
        model.summary(print_fn=logging.info)
        summary["model"] = {
            "name": model.name,
            "input_shape": [128, 128, 3], "output_classes": 23,
            "trainable_parameters": int(model.count_params()),
            "layers": [layer.__class__.__name__ for layer in model.layers],
            "optimizer": "Adam", "loss": "CategoricalCrossentropy", "metric": "accuracy",
        }

        logging.info("Phase 2: fit training images; validation controls early stopping/checkpoint selection.")
        callbacks: list[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(OUTPUT_DIR / "model.keras"), monitor="val_loss", mode="min",
                save_best_only=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.CSVLogger(str(OUTPUT_DIR / "training_history.csv")),
            monitor,
        ]
        fit_start = time.monotonic()
        history = model.fit(
            bundle["training"], validation_data=bundle["validation"],
            epochs=MAX_EPOCHS, class_weight=bundle["class_weights"],
            callbacks=callbacks, verbose=2, shuffle=False,
        )
        history_data = {key: [float(value) for value in values] for key, values in history.history.items()}
        # Keras CSVLogger writes zero-based epoch indices; make the saved report
        # match the human-facing one-based epoch numbers in the graphs/summary.
        history_csv = OUTPUT_DIR / "training_history.csv"
        with history_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
        with history_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                row["epoch"] = str(int(row["epoch"]) + 1)
                writer.writerow(row)
        save_graphs(history_data)
        if not (OUTPUT_DIR / "model.keras").is_file():
            raise RuntimeError("Training returned without a successfully validated model checkpoint.")
        # Diagnose collapsed outputs on validation after best weights are
        # restored; this report cannot influence model selection and excludes test.
        validation_diagnostic = validation_prediction_diagnostic(model, bundle)
        write_json(OUTPUT_DIR / "validation_prediction_diagnostic.json", validation_diagnostic)
        summary["validation_prediction_diagnostic"] = validation_diagnostic
        best_epoch = int(min(range(len(history_data["val_loss"])), key=history_data["val_loss"].__getitem__)) + 1
        best_idx = best_epoch - 1
        training_status = (
            "stopped_safely_due_to_resource_limit" if monitor.stop_reason
            else "early_stopped" if len(history_data.get("loss", [])) < MAX_EPOCHS
            else "completed_max_epochs"
        )
        summary["training"] = {
            "status": training_status,
            "completed_epochs": len(history_data.get("loss", [])),
            "duration_seconds": time.monotonic() - fit_start,
            "best_validation_epoch": best_epoch,
            "best_validation_loss": history_data["val_loss"][best_idx],
            "best_validation_accuracy": history_data["val_accuracy"][best_idx],
            "stop_reason": monitor.stop_reason,
            "epoch_resources": monitor.epoch_records,
        }
        status = "completed" if not monitor.stop_reason else "stopped_safely"
        exit_code = 0 if not monitor.stop_reason else 2
        logging.info("Training phase ended with status=%s; test remains unopened.", status)
    except BaseException as exc:
        error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        status = "failed"
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        logging.error("Training pipeline failed; test was not loaded.\n%s", error_text)
        if fit_start is not None:
            summary["training_duration_before_failure_seconds"] = time.monotonic() - fit_start
        summary["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": error_text}
        # Any checkpoint file present can only have been written by a successful
        # validation callback; leave it intact but report that the run failed.
    finally:
        summary["status"] = status
        summary["test_split_loaded"] = False
        summary["total_runtime_seconds"] = time.monotonic() - start
        summary["final_memory"] = memory_snapshot()
        summary["resource_monitor"] = {"stop_reason": monitor.stop_reason, "epoch_records": monitor.epoch_records}
        if history_data:
            summary["history_rows"] = len(history_data.get("loss", []))
        write_json(OUTPUT_DIR / "experiment_summary.json", summary)
        lines = [
            "DermaVision DermNet memory-efficient CNN V2",
            f"Status: {status}",
            f"Dataset: {data_loader_v2.DATASET_ROOT}",
            f"Counts (train/validation/test): {summary.get('dataset_counts', 'preflight incomplete')}",
            "Architecture: 128x128 RGB; Conv2D/MaxPooling filters 32/64/128; GlobalAveragePooling2D; Dense(128); Dropout(0.5); 23-class Softmax.",
            f"Batch size: 4; maximum epochs: {MAX_EPOCHS}; test split loaded: no.",
            f"Total runtime: {summary['total_runtime_seconds']:.1f} seconds.",
            f"Initial RAM sample: {summary.get('initial_memory')}; final RAM sample: {summary['final_memory']}.",
        ]
        if "training" in summary:
            training_result = summary["training"]
            lines.extend([
                f"Training status: {training_result['status']}; completed epochs: {training_result['completed_epochs']}.",
                f"Training duration: {training_result['duration_seconds']:.1f} seconds.",
                f"Best validation epoch: {training_result['best_validation_epoch']}; loss: {training_result['best_validation_loss']:.6f}; accuracy: {training_result['best_validation_accuracy']:.6f}.",
                f"Epoch resource samples: {len(training_result['epoch_resources'])}; detailed batch RAM/CPU/loss/accuracy samples: training.log.",
            ])
        if error_text:
            lines.extend(["Error:", error_text])
        lines.extend(["Limitations: experimental baseline, not clinically validated; no established Indian-specific validation was performed."])
        (OUTPUT_DIR / "experiment_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        crash_log.flush()
        crash_log.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run_training())
