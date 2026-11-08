"""Inspect DermaVision image datasets and write a Markdown analysis report.

Run from any directory with:
    python src/dataset_analysis.py

By default, this reads dataset/raw/DermNet and dataset/raw/HAM10000 relative to
the project root. It never changes or deletes source images.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:
    raise SystemExit(
        "Pillow is required to inspect image dimensions. Install it in your "
        "active Python environment with: python -m pip install Pillow"
    ) from exc


# Pillow-readable formats commonly used for image datasets. Files with other
# extensions are not counted as images, avoiding accidental counts of labels,
# archives, or metadata files.
IMAGE_EXTENSIONS = {
    ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}
PARTITION_NAMES = {"train", "training", "test", "testing", "val", "valid", "validation"}


def find_images(dataset_dir: Path) -> list[Path]:
    """Return supported image files recursively, in stable path order."""
    return sorted(
        (path for path in dataset_dir.rglob("*")
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


def load_ham_labels(dataset_dir: Path) -> tuple[dict[str, str], str | None]:
    """Read HAM10000 image_id/dx labels from a matching CSV, if present."""
    candidates = sorted(dataset_dir.glob("*.csv"))
    for csv_path in candidates:
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = {field.strip().lower() for field in (reader.fieldnames or [])}
                if not {"image_id", "dx"}.issubset(fields):
                    continue
                # Find actual field spellings so harmless capitalization or
                # surrounding whitespace in a header does not break the match.
                image_id_field = next(f for f in reader.fieldnames or [] if f.strip().lower() == "image_id")
                label_field = next(f for f in reader.fieldnames or [] if f.strip().lower() == "dx")
                labels = {
                    str(row[image_id_field]).strip(): str(row[label_field]).strip()
                    for row in reader
                    if row.get(image_id_field) and row.get(label_field)
                }
                return labels, csv_path.name
        except (OSError, csv.Error, UnicodeError):
            continue
    return {}, None


def category_for_image(
    image_path: Path,
    dataset_dir: Path,
    metadata_labels: dict[str, str],
) -> tuple[str, str]:
    """Infer a class from a class-folder layout or HAM metadata.

    The partition folders (train/test/validation) are skipped. When metadata
    maps an image ID, that label takes precedence, as HAM10000's image folders
    are storage partitions rather than disease categories.
    """
    if metadata_labels:
        label = metadata_labels.get(image_path.stem)
        return (label, "metadata") if label else ("Unlabeled (no metadata match)", "unlabeled")

    relative_parts = image_path.relative_to(dataset_dir).parts[:-1]
    category_parts = [part for part in relative_parts if part.casefold() not in PARTITION_NAMES]
    if category_parts:
        return category_parts[0], "folder"
    return "Uncategorized (image at dataset root)", "uncategorized"


def analyze_dataset(name: str, dataset_dir: Path) -> dict[str, Any]:
    """Count images, categories, extensions, dimensions, and unreadable files."""
    result: dict[str, Any] = {
        "name": name,
        "path": dataset_dir,
        "exists": dataset_dir.is_dir(),
    }
    if not result["exists"]:
        result.update({"images": [], "categories": {}, "extensions": Counter(),
                       "dimensions": Counter(), "corrupted": [], "metadata_file": None,
                       "label_source": "unavailable"})
        return result

    images = find_images(dataset_dir)
    labels, metadata_file = load_ham_labels(dataset_dir)
    category_counts: Counter[str] = Counter()
    category_sources: dict[str, str] = {}
    extension_counts: Counter[str] = Counter()
    dimension_counts: Counter[tuple[int, int]] = Counter()
    corrupted: list[tuple[str, str]] = []

    for image_path in images:
        category, source = category_for_image(image_path, dataset_dir, labels)
        category_counts[category] += 1
        category_sources[category] = source
        extension_counts[image_path.suffix.lower() or "(no extension)"] += 1

        # verify() checks that the file can be decoded; reopen afterward to
        # obtain dimensions because verify() invalidates the first image object.
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                dimension_counts[image.size] += 1
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            corrupted.append((str(image_path.relative_to(dataset_dir)), str(exc)))

    result.update({
        "images": images,
        "categories": dict(sorted(category_counts.items(), key=lambda item: item[0].casefold())),
        "category_sources": category_sources,
        "extensions": extension_counts,
        "dimensions": dimension_counts,
        "corrupted": corrupted,
        "metadata_file": metadata_file,
        "label_source": "metadata CSV" if labels else "folder names",
    })
    return result


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format a small Markdown table while escaping cell separators."""
    safe_rows = [[cell.replace("|", "\\|").replace("\n", " ") for cell in row] for row in rows]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in safe_rows)
    return "\n".join(lines)


def build_report(results: list[dict[str, Any]]) -> str:
    """Build a readable report with a separate section for each dataset."""
    lines = [
        "# DermaVision Dataset Analysis Report",
        "",
        "Generated by `src/dataset_analysis.py`. Source images were read only; no training was performed.",
        "",
    ]
    for result in results:
        lines.extend([f"## {result['name']}", "", f"- **Directory:** `{result['path']}`"])
        if not result["exists"]:
            lines.extend(["- **Status:** Directory not found; no image analysis was performed.", ""])
            continue

        total = len(result["images"])
        lines.extend([
            f"- **Total supported image files:** {total}",
            f"- **Label source:** {result['label_source']}",
        ])
        if result["metadata_file"]:
            lines.append(f"- **Label metadata:** `{result['metadata_file']}`")
        lines.extend(["", "### Class distribution", ""])
        if result["categories"]:
            class_rows = []
            for category, count in result["categories"].items():
                source = result["category_sources"].get(category, "folder")
                class_rows.append([category, str(count), f"{(count / total * 100):.2f}%" if total else "0.00%", source])
            lines.append(markdown_table(["Category", "Images", "Share", "Identified from"], class_rows))
        else:
            lines.append("No image files were found.")

        lines.extend(["", "### Image extensions", ""])
        ext_rows = [[extension, str(count)] for extension, count in sorted(result["extensions"].items())]
        lines.append(markdown_table(["Extension", "Images"], ext_rows) if ext_rows else "No image files found.")

        lines.extend(["", "### Image dimensions", ""])
        dimension_rows = [
            [f"{width} × {height}", str(count)]
            for (width, height), count in sorted(result["dimensions"].items())
        ]
        lines.append(markdown_table(["Width × height (pixels)", "Readable images"], dimension_rows)
                     if dimension_rows else "No readable image dimensions found.")
        lines.extend(["", "### Corrupted or unreadable images", ""])
        if result["corrupted"]:
            lines.append(f"Found **{len(result['corrupted'])}** unreadable image(s):")
            lines.extend(f"- `{relative_path}` — {reason}" for relative_path, reason in result["corrupted"])
        else:
            lines.append("None found." if total else "No image files to check.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """Parse optional paths, inspect each dataset independently, and save output."""
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dermnet", type=Path, default=project_root / "dataset" / "raw" / "DermNet",
                        help="DermNet directory (default: dataset/raw/DermNet)")
    parser.add_argument("--ham10000", type=Path, default=project_root / "dataset" / "raw" / "HAM10000",
                        help="HAM10000 directory (default: dataset/raw/HAM10000)")
    parser.add_argument("--report", type=Path, default=project_root / "results" / "dataset_analysis_report.md",
                        help="Output Markdown report path")
    args = parser.parse_args()

    # Resolve user-provided relative paths from the project root for predictable
    # behavior no matter which directory the command is run from.
    def from_project(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    dataset_results = [
        analyze_dataset("DermNet", from_project(args.dermnet)),
        analyze_dataset("HAM10000", from_project(args.ham10000)),
    ]
    report_path = from_project(args.report)
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_report(dataset_results), encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Could not save report to {report_path}: {exc}") from exc

    for result in dataset_results:
        if result["exists"]:
            print(f"{result['name']}: {len(result['images'])} image(s), "
                  f"{len(result['categories'])} category/categories, "
                  f"{len(result['corrupted'])} unreadable image(s)")
        else:
            print(f"{result['name']}: directory not found at {result['path']}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
