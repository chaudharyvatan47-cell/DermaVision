"""Compare class distributions in DermNet and HAM10000, without training.

Run from the project root (or any directory):
    python src/class_distribution.py

The script counts image files present on disk. DermNet labels come from its
class-folder structure; HAM10000 labels come from the metadata CSV's ``dx``
column, matched to image filenames through ``image_id``. The two datasets are
reported separately. Original images are only read for their filenames.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PARTITION_NAMES = {"train", "training", "test", "testing", "val", "valid", "validation"}

# HAM10000's diagnostic codes are read from the metadata, then expanded here
# for clearer charts and CSVs. Unknown codes remain visible in the report.
HAM_DIAGNOSES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def image_files(directory: Path) -> list[Path]:
    """Find supported image files recursively in a stable order."""
    return sorted(
        (path for path in directory.rglob("*")
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


def folder_category(image_path: Path, dataset_dir: Path) -> str:
    """Infer a folder label, skipping common train/test/validation levels."""
    folders = image_path.relative_to(dataset_dir).parts[:-1]
    class_folders = [folder for folder in folders if folder.casefold() not in PARTITION_NAMES]
    return class_folders[0] if class_folders else "Uncategorized (dataset root)"


def discover_folder_categories(dataset_dir: Path) -> set[str]:
    """Discover class folders, including empty folders in split layouts."""
    categories: set[str] = set()
    root_dirs = [path for path in dataset_dir.iterdir() if path.is_dir()]
    split_dirs = [path for path in root_dirs if path.name.casefold() in PARTITION_NAMES]

    if split_dirs:
        for split_dir in split_dirs:
            categories.update(path.name for path in split_dir.iterdir() if path.is_dir())
    else:
        # In a non-split layout, root-level folders are candidate class folders.
        # Metadata-based datasets do not use this result for their labels.
        categories.update(path.name for path in root_dirs)
    return categories


def find_ham_metadata(dataset_dir: Path) -> tuple[Path | None, dict[str, str], str | None]:
    """Locate a CSV with image_id and dx columns and return its image labels."""
    for csv_path in sorted(dataset_dir.glob("*.csv")):
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                normalized = {field.strip().lower(): field for field in fieldnames}
                if "image_id" not in normalized or "dx" not in normalized:
                    continue
                image_field = normalized["image_id"]
                diagnosis_field = normalized["dx"]
                labels: dict[str, str] = {}
                for row in reader:
                    image_id = (row.get(image_field) or "").strip()
                    diagnosis = (row.get(diagnosis_field) or "").strip().lower()
                    if image_id and diagnosis:
                        labels[image_id] = diagnosis
                return csv_path, labels, None
        except (OSError, UnicodeError, csv.Error) as error:
            return csv_path, {}, f"Could not read {csv_path.name}: {error}"
    return None, {}, "No CSV containing both image_id and dx columns was found."


def calculate_imbalance(counts: Counter[str]) -> dict[str, Any]:
    """Apply a simple screening heuristic, not a statistical diagnosis.

    Potential imbalance is flagged when the largest class is at least twice
    the smallest class, or is at least twice the equal-share expectation. This
    is a descriptive rule of thumb; it does not account for clinical context.
    """
    positive_counts = [count for count in counts.values() if count > 0]
    total = sum(counts.values())
    class_count = len(counts)
    if not counts or total == 0:
        return {"flagged": False, "ratio": None, "explanation": "No labeled images to compare."}

    largest = max(counts.values())
    smallest = min(counts.values())
    if smallest == 0:
        ratio: float | None = None
        flagged = True
        explanation = "At least one discovered category has zero images."
    else:
        ratio = largest / smallest
        equal_share = total / class_count
        flagged = ratio >= 2 or largest >= 2 * equal_share
        explanation = (
            f"Largest/smallest class ratio is {ratio:.2f}:1."
            if flagged else f"Largest/smallest class ratio is {ratio:.2f}:1; heuristic threshold not reached."
        )
    if len(positive_counts) < class_count and not smallest == 0:
        flagged = True
    return {"flagged": flagged, "ratio": ratio, "explanation": explanation}


def dataset_result(name: str, directory: Path, use_ham_metadata: bool = False) -> dict[str, Any]:
    """Count each image once and associate it with a discovered class label."""
    result: dict[str, Any] = {"name": name, "directory": directory, "exists": directory.is_dir()}
    if not result["exists"]:
        result.update({"counts": Counter(), "diagnoses": {}, "total_images": 0,
                       "unmatched_images": [], "metadata_file": None,
                       "metadata_error": "Dataset directory is missing."})
        return result

    files = image_files(directory)
    counts: Counter[str] = Counter()
    diagnosis_codes: dict[str, str] = {}
    unmatched: list[str] = []
    metadata_file: Path | None = None
    metadata_error: str | None = None

    if use_ham_metadata:
        metadata_file, labels_by_image, metadata_error = find_ham_metadata(directory)
        if labels_by_image:
            for image_path in files:
                code = labels_by_image.get(image_path.stem)
                if code is None:
                    unmatched.append(str(image_path.relative_to(directory)))
                    continue
                label = HAM_DIAGNOSES.get(code, f"Unmapped diagnostic code: {code}")
                counts[label] += 1
                diagnosis_codes[label] = code
        else:
            metadata_error = metadata_error or "Metadata CSV contains no usable image_id/dx rows."
    else:
        counts.update(folder_category(path, directory) for path in files)
        # Retain empty category directories so the analysis can report them.
        for category in discover_folder_categories(directory):
            counts.setdefault(category, 0)

    total_images = len(files)
    # Percentages use images with a usable category label as their denominator.
    # Unmatched images are reported separately and are not treated as a disease.
    distribution_total = sum(counts.values())
    percentages = {
        label: (count / distribution_total * 100 if distribution_total else 0.0)
        for label, count in counts.items()
    }
    imbalance = calculate_imbalance(counts)
    result.update({
        "counts": Counter(dict(sorted(counts.items(), key=lambda item: item[0].casefold()))),
        "diagnoses": diagnosis_codes,
        "percentages": percentages,
        "total_images": total_images,
        "labeled_images": distribution_total,
        "unmatched_images": unmatched,
        "metadata_file": metadata_file,
        "metadata_error": metadata_error,
        "imbalance": imbalance,
    })
    return result


def write_csv_report(result: dict[str, Any], output_path: Path) -> None:
    """Save category counts, percentages, and the imbalance screening result."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "diagnostic_code", "category", "image_count", "percentage",
              "dataset_image_total", "potential_imbalance", "largest_smallest_ratio"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if not result["exists"]:
            return
        imbalance = result["imbalance"]
        for category, count in result["counts"].items():
            writer.writerow({
                "dataset": result["name"],
                "diagnostic_code": result["diagnoses"].get(category, ""),
                "category": category,
                "image_count": count,
                "percentage": f"{result['percentages'].get(category, 0.0):.4f}",
                "dataset_image_total": result["total_images"],
                "potential_imbalance": "yes" if imbalance["flagged"] else "no",
                "largest_smallest_ratio": "" if imbalance["ratio"] is None else f"{imbalance['ratio']:.4f}",
            })


def save_chart(result: dict[str, Any], output_path: Path) -> str | None:
    """Create a labeled horizontal bar chart using Matplotlib and Seaborn."""
    if plt is None or sns is None:
        return "Charts were skipped: install matplotlib and seaborn in the active Python environment."
    if not result["exists"] or not result["counts"]:
        return "Chart was skipped because this dataset has no category counts."

    rows = sorted(result["counts"].items(), key=lambda item: (item[1], item[0].casefold()))
    labels = [label for label, _ in rows]
    values = [count for _, count in rows]
    height = max(4.5, 0.42 * len(labels) + 1.5)
    figure, axis = plt.subplots(figsize=(12, height))
    sns.barplot(x=values, y=labels, ax=axis, color="#3977a8")
    axis.set_title(f"{result['name']} class distribution")
    axis.set_xlabel("Number of images")
    axis.set_ylabel("Diagnostic category")
    axis.grid(axis="x", linestyle=":", alpha=0.4)
    axis.set_axisbelow(True)
    axis.bar_label(axis.containers[0], fmt="%d", padding=4)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return None


def print_summary(result: dict[str, Any]) -> None:
    """Print totals, extrema, and the heuristic imbalance assessment."""
    print(f"\n{result['name']} — {result['directory']}")
    if not result["exists"]:
        print("  Dataset directory not found; skipped.")
        return
    print(f"  Images found: {result['total_images']}")
    if result.get("metadata_file"):
        print(f"  HAM metadata: {result['metadata_file'].name}")
    if result.get("metadata_error"):
        print(f"  Metadata note: {result['metadata_error']}")
    counts: Counter[str] = result["counts"]
    if counts:
        max_count, min_count = max(counts.values()), min(counts.values())
        most = [name for name, count in counts.items() if count == max_count]
        least = [name for name, count in counts.items() if count == min_count]
        print(f"  Most represented: {', '.join(most)} ({max_count})")
        print(f"  Least represented: {', '.join(least)} ({min_count})")
        imbalance = result["imbalance"]
        status = "potential imbalance flagged" if imbalance["flagged"] else "heuristic threshold not reached"
        print(f"  Imbalance: {status}. {imbalance['explanation']}")
    if result.get("unmatched_images"):
        print(f"  Images without a matching HAM metadata row: {len(result['unmatched_images'])}")


def main() -> None:
    """Analyze both datasets independently and write separate outputs."""
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dermnet", type=Path, default=Path("dataset/raw/DermNet"),
                        help="DermNet directory (relative paths are from project root)")
    parser.add_argument("--ham10000", type=Path, default=Path("dataset/raw/HAM10000"),
                        help="HAM10000 directory (relative paths are from project root)")
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="Directory for charts and CSV reports")
    args = parser.parse_args()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    datasets = [
        dataset_result("DermNet", resolve(args.dermnet)),
        dataset_result("HAM10000", resolve(args.ham10000), use_ham_metadata=True),
    ]
    results_dir = resolve(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    chart_notes: list[str] = []
    for result in datasets:
        slug = result["name"].lower()
        write_csv_report(result, results_dir / f"{slug}_class_distribution.csv")
        chart_note = save_chart(result, results_dir / f"{slug}_class_distribution.png")
        if chart_note:
            chart_notes.append(f"{result['name']}: {chart_note}")
        print_summary(result)

    if chart_notes:
        for note in chart_notes:
            print(f"\n{note}")
    print(f"\nCSV reports and available charts saved in: {results_dir}")


if __name__ == "__main__":
    main()
