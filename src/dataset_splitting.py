"""Create reproducible, separate train/validation/test copies of both datasets.

Run from any directory with:
    python src/dataset_splitting.py

DermNet labels are read from its category folders and byte-identical files are
kept together. HAM10000 labels and lesion groups are read from its metadata.
The split allocator is group-aware and greedily approximates the requested
70% / 15% / 15% class distribution without separating a group. Source files
are copied, never moved or modified. No model training or augmentation occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PARTITION_NAMES = {"train", "training", "test", "testing", "val", "valid", "validation"}
SPLITS = (("train", 0.70), ("validation", 0.15), ("test", 0.15))
HAM_DIAGNOSES = {
    "akiec": "Actinic keratoses - intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def find_images(dataset_dir: Path) -> list[Path]:
    """List supported images recursively, ordered for reproducible processing."""
    return sorted(
        (path for path in dataset_dir.rglob("*")
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


def dermnet_label(image_path: Path, dataset_dir: Path) -> str:
    """Get a DermNet label from its category-folder layout, skipping split dirs."""
    folders = image_path.relative_to(dataset_dir).parts[:-1]
    for index, folder in enumerate(folders):
        if folder.casefold() in PARTITION_NAMES and index + 1 < len(folders):
            return folders[index + 1]
    if folders:
        return folders[0]
    raise ValueError(f"Cannot infer a category folder for DermNet image: {image_path}")


def hash_file(path: Path) -> str:
    """Return a SHA-256 digest for exact duplicate detection, reading in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ham_metadata(dataset_dir: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    """Find and validate a HAM CSV containing image_id, lesion_id, and dx."""
    last_error: str | None = None
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                field_map = {field.strip().lower(): field for field in (reader.fieldnames or [])}
                required = {"image_id", "lesion_id", "dx"}
                if not required.issubset(field_map):
                    continue
                metadata: dict[str, dict[str, str]] = {}
                for row_number, row in enumerate(reader, start=2):
                    image_id = (row.get(field_map["image_id"]) or "").strip()
                    lesion_id = (row.get(field_map["lesion_id"]) or "").strip()
                    dx = (row.get(field_map["dx"]) or "").strip().lower()
                    if not image_id or not lesion_id or not dx:
                        raise ValueError(
                            f"{csv_path.name}:{row_number} has a blank image_id, lesion_id, or dx."
                        )
                    if image_id in metadata:
                        raise ValueError(f"Duplicate image_id {image_id!r} in {csv_path.name}.")
                    metadata[image_id] = {"lesion_id": lesion_id, "dx": dx}
                if not metadata:
                    raise ValueError(f"No usable records were found in {csv_path.name}.")
                return csv_path, metadata
        except (OSError, UnicodeError, csv.Error) as error:
            last_error = f"Could not read {csv_path}: {error}"
    detail = f" Last CSV error: {last_error}" if last_error else ""
    raise FileNotFoundError(
        f"No readable HAM10000 CSV with image_id, lesion_id, and dx columns was found in "
        f"{dataset_dir}.{detail}"
    )


def build_dermnet_records(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build DermNet records and group exact duplicate files by content hash."""
    images = find_images(dataset_dir)
    if not images:
        raise ValueError(f"No supported images found under DermNet directory: {dataset_dir}")

    records: list[dict[str, Any]] = []
    hash_counts: Counter[str] = Counter()
    for index, image_path in enumerate(images, start=1):
        label = dermnet_label(image_path, dataset_dir)
        digest = hash_file(image_path)
        hash_counts[digest] += 1
        records.append({
            "path": image_path,
            "relative_path": image_path.relative_to(dataset_dir).as_posix(),
            "label": label,
            # Exact byte duplicates remain in one split, even across categories.
            "group_id": f"sha256:{digest}",
            "diagnostic_code": "",
            "image_id": image_path.stem,
            "lesion_id": "",
        })
        if index % 2000 == 0:
            print(f"DermNet: inspected {index:,}/{len(images):,} image files...")

    repeated_groups = sum(1 for count in hash_counts.values() if count > 1)
    repeated_files = sum(count - 1 for count in hash_counts.values() if count > 1)
    cross_class_groups: set[str] = set()
    group_labels: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_labels[record["group_id"]].add(record["label"])
    cross_class_groups = {group for group, labels in group_labels.items() if len(labels) > 1}
    diagnostics = {
        "image_count": len(records),
        "duplicate_groups": repeated_groups,
        "duplicate_extra_files": repeated_files,
        "cross_class_duplicate_groups": len(cross_class_groups),
        "patient_ids_available": False,
    }
    return records, diagnostics


def build_ham_records(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match every HAM image to its dx and lesion_id metadata row."""
    metadata_path, metadata = load_ham_metadata(dataset_dir)
    images = find_images(dataset_dir)
    if not images:
        raise ValueError(f"No supported images found under HAM10000 directory: {dataset_dir}")

    images_by_id: dict[str, Path] = {}
    for image_path in images:
        image_id = image_path.stem
        if image_id in images_by_id:
            raise ValueError(
                f"More than one image file has ID {image_id!r}: "
                f"{images_by_id[image_id]} and {image_path}"
            )
        images_by_id[image_id] = image_path

    metadata_ids = set(metadata)
    image_ids = set(images_by_id)
    missing_files = sorted(metadata_ids - image_ids)
    missing_metadata = sorted(image_ids - metadata_ids)
    if missing_files or missing_metadata:
        details = []
        if missing_files:
            details.append(f"metadata rows without image files: {len(missing_files)} (e.g. {missing_files[:5]})")
        if missing_metadata:
            details.append(f"image files without metadata rows: {len(missing_metadata)} (e.g. {missing_metadata[:5]})")
        raise ValueError(
            "HAM10000 image/metadata mismatch; no HAM split was written. " + "; ".join(details)
        )

    records: list[dict[str, Any]] = []
    for image_id, image_path in images_by_id.items():
        meta = metadata[image_id]
        dx = meta["dx"]
        label = HAM_DIAGNOSES.get(dx, f"Unmapped diagnosis {dx}")
        records.append({
            "path": image_path,
            "relative_path": image_path.relative_to(dataset_dir).as_posix(),
            "label": label,
            # All images for a lesion are assigned together by the allocator.
            "group_id": f"lesion:{meta['lesion_id']}",
            "diagnostic_code": dx,
            "image_id": image_id,
            "lesion_id": meta["lesion_id"],
        })

    lesion_labels: dict[str, set[str]] = defaultdict(set)
    for record in records:
        lesion_labels[record["lesion_id"]].add(record["diagnostic_code"])
    multi_diagnosis_lesions = sum(1 for labels in lesion_labels.values() if len(labels) > 1)
    return records, {
        "image_count": len(records),
        "metadata_path": metadata_path,
        "metadata_rows": len(metadata),
        "lesion_count": len(lesion_labels),
        "multi_diagnosis_lesions": multi_diagnosis_lesions,
    }


def assignment_cost(
    split_counts: dict[str, Counter[str]],
    split_totals: Counter[str],
    group_counts: Counter[str],
    class_totals: Counter[str],
    candidate_split: str,
) -> float:
    """Measure class and total deviations after placing a group in one split."""
    cost = 0.0
    for split_name, fraction in SPLITS:
        added_total = sum(group_counts.values()) if split_name == candidate_split else 0
        target_total = sum(class_totals.values()) * fraction
        actual_total = split_totals[split_name] + added_total
        cost += ((actual_total - target_total) / max(target_total, 1.0)) ** 2

        for label, dataset_count in class_totals.items():
            added_class = group_counts[label] if split_name == candidate_split else 0
            target_class = dataset_count * fraction
            actual_class = split_counts[split_name][label] + added_class
            cost += ((actual_class - target_class) / max(target_class, 1.0)) ** 2
    return cost


def assign_groups(records: list[dict[str, Any]], seed: int) -> dict[str, str]:
    """Greedily allocate indivisible groups to approximate 70/15/15 by class."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    class_totals: Counter[str] = Counter()
    for record in records:
        groups[record["group_id"]].append(record)
        class_totals[record["label"]] += 1

    # Larger groups are harder to place, so allocate them first. The seeded
    # shuffle provides reproducible tie-breaking among groups of equal size.
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    group_ids.sort(key=lambda group_id: -len(groups[group_id]))

    split_counts = {name: Counter() for name, _ in SPLITS}
    split_totals: Counter[str] = Counter()
    assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for group_id in group_ids:
        group_counts = Counter(record["label"] for record in groups[group_id])
        costs = {
            split_name: assignment_cost(split_counts, split_totals, group_counts, class_totals, split_name)
            for split_name, _ in SPLITS
        }
        best_cost = min(costs.values())
        best_splits = [name for name, cost in costs.items() if abs(cost - best_cost) < 1e-12]
        selected = rng.choice(best_splits)
        assignments[group_id] = selected
        split_counts[selected].update(group_counts)
        split_totals[selected] += sum(group_counts.values())
    return assignments


def output_relative_path(record: dict[str, Any], destination: Path, used: set[Path]) -> Path:
    """Choose a safe class-folder destination and avoid filename collisions."""
    base = destination / record["split"] / record["label"]
    target = base / record["path"].name
    if target in used:
        # Keep every source image even if separate folders reuse a filename.
        relative_key = record["relative_path"].encode("utf-8")
        suffix = hashlib.sha256(relative_key).hexdigest()[:10]
        target = base / f"{record['path'].stem}__{suffix}{record['path'].suffix.lower()}"
    if target in used:
        raise FileExistsError(f"Destination filename collision could not be resolved: {target}")
    used.add(target)
    return target


def write_manifest(records: list[dict[str, Any]], dataset_name: str, output_dir: Path) -> None:
    """Write the source-to-output mapping and relevant labels/group IDs."""
    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "source_relative_path", "split", "class_label",
                  "diagnostic_code", "image_id", "lesion_id", "group_id", "output_relative_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "dataset": dataset_name,
                "source_relative_path": record["relative_path"],
                "split": record["split"],
                "class_label": record["label"],
                "diagnostic_code": record["diagnostic_code"],
                "image_id": record["image_id"],
                "lesion_id": record["lesion_id"],
                "group_id": record["group_id"],
                "output_relative_path": record["output_relative_path"],
            })


def split_dataset(
    name: str,
    records: list[dict[str, Any]],
    destination: Path,
    seed: int,
    diagnostics: dict[str, Any],
) -> None:
    """Assign, copy, manifest, and print per-class counts for one dataset."""
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(
            f"Output already contains files: {destination}. Choose a new --processed directory "
            "or move existing outputs yourself; this script will not overwrite or delete them."
        )

    assignments = assign_groups(records, seed)
    for record in records:
        record["split"] = assignments[record["group_id"]]

    used_destinations: set[Path] = set()
    for record in records:
        target = output_relative_path(record, destination, used_destinations)
        record["output_relative_path"] = target.relative_to(destination).as_posix()

    # All metadata and destination paths have been validated before the first
    # copy. copy2 preserves source bytes and timestamps without modifying input.
    class_names = sorted({record["label"] for record in records}, key=str.casefold)
    counts_by_split = {split_name: Counter() for split_name, _ in SPLITS}
    for record in records:
        counts_by_split[record["split"]][record["label"]] += 1

    destination.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, start=1):
        target = destination / record["output_relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output file: {target}")
        shutil.copy2(record["path"], target)
        if index % 2000 == 0:
            print(f"{name}: copied {index:,}/{len(records):,} images...")
    write_manifest(records, name, destination)

    print(f"\n{name} split summary (seed={seed})")
    print(f"  Total images: {len(records):,}")
    for split_name, fraction in SPLITS:
        split_total = sum(counts_by_split[split_name].values())
        print(f"  {split_name.title()} ({fraction:.0%} target): {split_total:,} images")
        for label in class_names:
            class_count = counts_by_split[split_name][label]
            share = class_count / split_total * 100 if split_total else 0.0
            print(f"    {label}: {class_count:,} ({share:.2f}% of this split)")
    if "duplicate_groups" in diagnostics:
        print(f"  Exact duplicate groups kept together: {diagnostics['duplicate_groups']:,} "
              f"({diagnostics['duplicate_extra_files']:,} additional copies)")
        print(f"  Exact duplicate groups spanning category labels: "
              f"{diagnostics['cross_class_duplicate_groups']:,}")
        print("  Patient identifiers: none found in the inspected folder/filename structure.")
        print("  Limitation: byte-identical grouping cannot identify different images from the same patient.")
    if "lesion_count" in diagnostics:
        print(f"  HAM metadata: {diagnostics['metadata_path'].name}; "
              f"{diagnostics['lesion_count']:,} lesion groups")
        print(f"  Lesions with multiple dx labels: {diagnostics['multi_diagnosis_lesions']:,}")
    print(f"  Manifest: {destination / 'split_manifest.csv'}")


def resolve_path(path: Path, project_root: Path) -> Path:
    """Resolve relative CLI paths from the project root."""
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def main() -> int:
    """Inspect both inputs, then create independent split directories."""
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dermnet", type=Path, default=Path("dataset/raw/DermNet"),
                        help="DermNet input directory (relative paths are from project root)")
    parser.add_argument("--ham10000", type=Path, default=Path("dataset/raw/HAM10000"),
                        help="HAM10000 input directory (relative paths are from project root)")
    parser.add_argument("--processed", type=Path, default=Path("dataset/processed"),
                        help="Root for separate processed dataset folders")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed (default: 42)")
    args = parser.parse_args()

    dermnet_dir = resolve_path(args.dermnet, project_root)
    ham_dir = resolve_path(args.ham10000, project_root)
    processed_root = resolve_path(args.processed, project_root)

    if processed_root.exists() and not processed_root.is_dir():
        print(f"Processed output path is not a directory: {processed_root}", file=sys.stderr)
        return 1

    # Inspect and validate both datasets before writing any split copies.
    try:
        if not dermnet_dir.is_dir():
            raise FileNotFoundError(f"DermNet directory not found: {dermnet_dir}")
        if not ham_dir.is_dir():
            raise FileNotFoundError(f"HAM10000 directory not found: {ham_dir}")
        dermnet_records, dermnet_info = build_dermnet_records(dermnet_dir)
        ham_records, ham_info = build_ham_records(ham_dir)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Preflight failed; no split copies were written: {error}", file=sys.stderr)
        return 1

    # Confirm both target dataset folders are safe before beginning either copy.
    destinations = [processed_root / "DermNet", processed_root / "HAM10000"]
    for destination in destinations:
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            print(
                f"Output already contains files: {destination}. No data was copied. "
                "Choose another --processed path or move existing outputs yourself.",
                file=sys.stderr,
            )
            return 1

    try:
        split_dataset("DermNet", dermnet_records, destinations[0], args.seed, dermnet_info)
        split_dataset("HAM10000", ham_records, destinations[1], args.seed, ham_info)
    except (OSError, shutil.Error, ValueError) as error:
        print(
            f"Split output failed: {error}\nSome output files may already have been copied; "
            "source datasets remain unchanged. Inspect dataset/processed before retrying.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
