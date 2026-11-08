"""Build a separate DermNet copy with duplicate groups handled conservatively.

This script reads the active processed split manifest and duplicate review CSV.
It does not alter raw images, existing processed splits, labels, or manifests.
Groups with conflicting original category labels remain unresolved and are
excluded from all derived splits. Same-label exact duplicates keep one stable
representative in their existing split. No diagnosis is inferred or corrected.

Run from any directory with:
    python src/build_dermnet_derived_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


VALID_SPLITS = {"train", "validation", "test"}
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EXPECTED_CONFLICT_GROUPS = 429


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV with BOM support and fail clearly if it is absent."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def json_list(values: list[str]) -> str:
    """Serialize list-valued audit columns unambiguously inside CSV cells."""
    return json.dumps(values, ensure_ascii=False)


def build_input_records(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and join the source manifest with the duplicate review decisions."""
    raw_root = project_root / "dataset" / "raw" / "DermNet"
    active_root = project_root / "dataset" / "processed" / "DermNet"
    review_path = project_root / "results" / "duplicate_review" / "dermnet_duplicate_review.csv"
    manifest_path = active_root / "split_manifest.csv"

    if not raw_root.is_dir() or not active_root.is_dir():
        raise FileNotFoundError(f"DermNet raw or active processed directory is missing: {raw_root} / {active_root}")

    review_rows = read_csv(review_path)
    manifest_rows = read_csv(manifest_path)
    required_review = {
        "group_id", "image_filename", "original_image_path", "original_category_label",
        "number_of_images_in_group", "duplicate_type", "label_conflict_status",
    }
    required_manifest = {
        "group_id", "source_relative_path", "split", "class_label", "output_relative_path",
    }
    if review_rows and not required_review.issubset(review_rows[0]):
        raise ValueError(f"Review CSV is missing required columns: {sorted(required_review - set(review_rows[0]))}")
    if manifest_rows and not required_manifest.issubset(manifest_rows[0]):
        raise ValueError(f"Split manifest is missing required columns: {sorted(required_manifest - set(manifest_rows[0]))}")
    if not review_rows or not manifest_rows:
        raise ValueError("Review CSV or active split manifest has no data rows.")

    manifest_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    manifest_by_source: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        if row["split"] not in VALID_SPLITS:
            raise ValueError(f"Unexpected active split {row['split']!r} in the manifest.")
        source_key = PurePosixPath(row["source_relative_path"]).as_posix().casefold()
        if source_key in manifest_by_source:
            raise ValueError(f"Duplicate source path in active manifest: {row['source_relative_path']}")
        manifest_by_source[source_key] = row
        manifest_by_group[row["group_id"]].append(row)

    review_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in review_rows:
        review_by_group[row["group_id"]].append(row)

    # Derive conflicts independently from original labels, then require the
    # review package to describe precisely those groups and source records.
    computed_conflicts = {
        group_id
        for group_id, rows in manifest_by_group.items()
        if len(rows) > 1 and len({row["class_label"] for row in rows}) > 1
    }
    if computed_conflicts != set(review_by_group):
        missing = sorted(computed_conflicts - set(review_by_group))
        unexpected = sorted(set(review_by_group) - computed_conflicts)
        raise ValueError(
            "Review CSV and active manifest disagree about conflicting groups. "
            f"Missing from review CSV: {len(missing)}; unexpected: {len(unexpected)}. "
            f"Examples: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    if len(computed_conflicts) != EXPECTED_CONFLICT_GROUPS:
        raise ValueError(
            f"Expected {EXPECTED_CONFLICT_GROUPS} conflicting groups from the audit; "
            f"found {len(computed_conflicts)}. No outputs will be written."
        )

    # Validate every reviewed path/label against the active manifest and raw
    # original before creating any output. The review labels are never changed.
    for group_id, rows in review_by_group.items():
        manifest_group = manifest_by_group[group_id]
        if len(rows) != len(manifest_group):
            raise ValueError(f"Review row count does not match manifest for {group_id}.")
        review_sources: set[str] = set()
        for row in rows:
            original_path = Path(row["original_image_path"]).resolve()
            try:
                relative = original_path.relative_to(raw_root.resolve()).as_posix()
            except ValueError as exc:
                raise ValueError(f"Review path is not under the DermNet raw directory: {original_path}") from exc
            source_key = relative.casefold()
            source_row = manifest_by_source.get(source_key)
            if source_row is None or source_row["group_id"] != group_id:
                raise ValueError(f"Review path is not mapped to its group in the active manifest: {original_path}")
            if row["original_category_label"] != source_row["class_label"]:
                raise ValueError(f"Review label differs from manifest for {original_path}.")
            if Path(row["image_filename"]).name != original_path.name:
                raise ValueError(f"Review filename differs from its original path: {original_path}")
            if not original_path.is_file():
                raise FileNotFoundError(f"Original DermNet image is missing: {original_path}")
            review_sources.add(source_key)
        if len(review_sources) != len(manifest_group):
            raise ValueError(f"Review CSV omits or repeats a source image in group {group_id}.")

    # Check all raw and active processed files before making a derived copy.
    records: list[dict[str, Any]] = []
    for row in manifest_rows:
        relative = PurePosixPath(row["source_relative_path"])
        source_path = raw_root.joinpath(*relative.parts)
        active_path = active_root.joinpath(*PurePosixPath(row["output_relative_path"]).parts)
        records.append({
            "group_id": row["group_id"],
            "source_relative_path": relative.as_posix(),
            "source_path": source_path,
            "active_path": active_path,
            "split": row["split"],
            "label": row["class_label"],
            "filename": source_path.name,
        })

    # The group IDs are the SHA-256 duplicate-detection results already
    # recorded in the active manifest and used to create the review package.
    # Validate their format here instead of re-reading gigabytes of image data.
    duplicate_groups = {group_id: rows for group_id, rows in manifest_by_group.items() if len(rows) > 1}
    for group_id, rows in duplicate_groups.items():
        expected = group_id.removeprefix("sha256:")
        if len(expected) != 64:
            raise ValueError(f"Unexpected duplicate group ID format: {group_id}")

    # The active manifest must already keep each exact duplicate group intact.
    group_splits = {
        group_id: sorted({row["split"] for row in rows})
        for group_id, rows in manifest_by_group.items()
    }
    leaking = {group_id: splits for group_id, splits in group_splits.items() if len(splits) > 1}
    if leaking:
        examples = list(leaking.items())[:5]
        raise ValueError(f"Active split manifest has duplicate-group leakage: {examples}")

    # Detect the raw-source train/test paths that were previously mistaken for
    # active split leakage. The active assignment remains sourced from manifest.
    raw_train_test_groups: list[str] = []
    for group_id, rows in review_by_group.items():
        raw_parts = {
            "train" if PurePosixPath(row["source_relative_path"]).parts[0].casefold() == "train"
            else "test" if PurePosixPath(row["source_relative_path"]).parts[0].casefold() == "test"
            else "other"
            for row in manifest_by_group[group_id]
        }
        if {"train", "test"}.issubset(raw_parts):
            raw_train_test_groups.append(group_id)

    diagnostics = {
        "project_root": str(project_root),
        "raw_root": str(raw_root),
        "active_root": str(active_root),
        "review_csv": str(review_path),
        "manifest": str(manifest_path),
        "input_image_count": len(records),
        "input_group_count": len(manifest_by_group),
        "duplicate_group_count": len(duplicate_groups),
        "conflicting_group_count": len(computed_conflicts),
        "raw_train_test_path_groups": sorted(raw_train_test_groups),
        "active_cross_split_group_count": len(leaking),
        "manifest_by_group": manifest_by_group,
        "conflict_ids": computed_conflicts,
    }
    return records, diagnostics


def choose_destination(record: dict[str, Any], destination: Path, used: set[str]) -> Path:
    """Preserve filenames where possible and disambiguate collisions deterministically."""
    target_dir = destination / record["split"] / record["label"]
    target = target_dir / record["filename"]
    relative_key = record["source_relative_path"]
    key = target.relative_to(destination).as_posix().casefold()
    if key in used:
        suffix = hashlib.sha256(relative_key.encode("utf-8")).hexdigest()[:10]
        target = target_dir / f"{Path(record['filename']).stem}__{suffix}{Path(record['filename']).suffix.lower()}"
        key = target.relative_to(destination).as_posix().casefold()
    if key in used:
        raise FileExistsError(f"Derived output path collision: {target}")
    used.add(key)
    return target


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Write one new CSV; refuse to overwrite any pre-existing file."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(project_root: Path) -> dict[str, Any]:
    """Validate, derive, and write audit reports without touching source files."""
    records, diagnostics = build_input_records(project_root)
    output_root = project_root / "dataset" / "derived" / "DermNet_review_v1"
    report_root = project_root / "results" / "duplicate_review"
    report_names = [
        "duplicate_group_decisions.csv",
        "excluded_conflicting_groups.csv",
        "retained_duplicate_groups.csv",
        "split_leakage_audit.csv",
        "final_dataset_summary.json",
        "final_dataset_summary.txt",
    ]
    existing_reports = [report_root / name for name in report_names if (report_root / name).exists()]
    if existing_reports:
        raise FileExistsError(
            "Refusing to overwrite existing report(s): " + ", ".join(map(str, existing_reports))
        )
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing derived dataset: {output_root}"
        )

    records_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_group[record["group_id"]].append(record)

    decision_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    retained_duplicate_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    keep_records: list[dict[str, Any]] = []
    kept_representatives: dict[str, dict[str, Any]] = {}
    duplicate_extra_consistent = 0
    conflict_images_by_split: Counter[str] = Counter()
    conflict_groups_by_split: Counter[str] = Counter()

    for group_id, group_records in sorted(records_by_group.items()):
        group_records.sort(key=lambda row: row["source_relative_path"].casefold())
        categories = sorted({row["label"] for row in group_records}, key=str.casefold)
        active_splits = sorted({row["split"] for row in group_records})
        raw_sources = [row["source_relative_path"] for row in group_records]
        filenames = [row["filename"] for row in group_records]
        is_conflict = len(categories) > 1
        is_duplicate = len(group_records) > 1
        is_review_group = group_id in diagnostics["conflict_ids"]
        if is_conflict != is_review_group:
            raise ValueError(f"Conflict status disagreement for {group_id}.")

        if is_review_group:
            status = "unresolved"
            decision = "exclude_unresolved_group_from_all_derived_splits"
            representative = None
            for row in group_records:
                conflict_images_by_split[row["split"]] += 1
            conflict_groups_by_split[active_splits[0]] += 1
            for row in group_records:
                excluded_rows.append({
                    "group_id": group_id,
                    "image_filename": row["filename"],
                    "original_image_path": str(row["source_path"].resolve()),
                    "original_category_label": row["label"],
                    "group_original_category_labels": json_list(categories),
                    "number_of_images_in_group": len(group_records),
                    "active_split": row["split"],
                    "duplicate_type": "Exact byte-level duplicate (SHA-256 match)",
                    "label_conflict_status": "unresolved",
                    "exclusion_reason": "Conflicting original category labels; no label selected",
                })
        else:
            status = "consistent_label" if is_duplicate else "not_a_duplicate_group"
            decision = "retain_one_representative" if is_duplicate else "retain_single_image"
            representative = group_records[0]
            keep_records.append(representative)
            kept_representatives[group_id] = representative
            if is_duplicate:
                duplicate_extra_consistent += len(group_records) - 1
                retained_duplicate_rows.append({
                    "group_id": group_id,
                    "original_category_label": categories[0],
                    "group_original_category_labels": json_list(categories),
                    "number_of_images_in_group": len(group_records),
                    "retained_count": 1,
                    "excluded_exact_copies_count": len(group_records) - 1,
                    "active_split": active_splits[0],
                    "representative_filename": representative["filename"],
                    "representative_original_path": str(representative["source_path"].resolve()),
                    "representative_derived_path": "",  # Filled after destination planning.
                    "all_original_filenames": json_list(filenames),
                    "all_original_paths": json_list([str(row["source_path"].resolve()) for row in group_records]),
                    "all_original_category_labels": json_list([row["label"] for row in group_records]),
                })

        if is_duplicate:
            decision_rows.append({
                "group_id": group_id,
                "number_of_images_in_group": len(group_records),
                "original_category_labels": json_list(categories),
                "original_image_filenames": json_list(filenames),
                "original_image_paths": json_list([str(row["source_path"].resolve()) for row in group_records]),
                "active_split": active_splits[0],
                "duplicate_type": "Exact byte-level duplicate (SHA-256 match)",
                "label_conflict_status": "unresolved" if is_review_group else "consistent",
                "decision": decision,
                "manual_review_status": "unresolved" if is_review_group else "not_required_for_label_consistency",
                "representative_original_path": str(representative["source_path"].resolve()) if representative else "",
            })
            leakage_rows.append({
                "group_id": group_id,
                "number_of_images_in_group": len(group_records),
                "original_category_labels": json_list(categories),
                "active_splits": json_list(active_splits),
                "derived_split": "excluded" if is_review_group else active_splits[0],
                "label_conflict_status": "unresolved" if is_review_group else "consistent",
                "cross_split_status": "PASS" if len(active_splits) == 1 else "FAIL",
                "source_paths": json_list(raw_sources),
            })

    # Plan all derived paths before creating output. Every kept group remains
    # in its active split; exact duplicate groups have only one representative.
    used_paths: set[str] = set()
    for record in keep_records:
        destination = choose_destination(record, output_root, used_paths)
        record["derived_relative_path"] = destination.relative_to(output_root).as_posix()
        record["derived_path"] = destination
    representative_by_source = {
        str(record["source_path"].resolve()): record
        for record in keep_records
    }
    for row in retained_duplicate_rows:
        rep = representative_by_source.get(row["representative_original_path"])
        if rep is None:
            raise ValueError(f"Retained representative is absent from derived records: {row['representative_original_path']}")
        row["representative_derived_path"] = str(rep["derived_path"])

    # Cross-split checks apply to every group and to the planned output records.
    derived_group_splits: dict[str, set[str]] = defaultdict(set)
    for record in keep_records:
        derived_group_splits[record["group_id"]].add(record["split"])
    derived_leaks = {group: splits for group, splits in derived_group_splits.items() if len(splits) > 1}
    if derived_leaks:
        raise ValueError(f"Derived group leakage detected before writing: {list(derived_leaks.items())[:5]}")

    # Recheck every output path and report name immediately before mutation.
    if output_root.exists():
        raise FileExistsError(f"Derived dataset appeared during preflight: {output_root}")
    existing_reports = [report_root / name for name in report_names if (report_root / name).exists()]
    if existing_reports:
        raise FileExistsError("Report appeared during preflight: " + ", ".join(map(str, existing_reports)))

    # Copy original bytes into the new tree. Originals and active processed
    # splits remain read-only; copy2 does not move or alter the source.
    output_root.mkdir(parents=True, exist_ok=False)
    copied = 0
    for record in keep_records:
        destination = record["derived_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite derived image: {destination}")
        shutil.copy2(record["source_path"], destination)
        copied += 1
        if copied % 2000 == 0:
            print(f"Copied {copied:,}/{len(keep_records):,} derived images...", flush=True)

    # Populate the representative path in the group decision records.
    for row in decision_rows:
        rep = row["representative_original_path"]
        if rep:
            rep_record = representative_by_source.get(rep)
            if rep_record is None:
                raise ValueError(f"Decision representative is absent from derived records: {rep}")
            row["representative_derived_path"] = str(rep_record["derived_path"])
        else:
            row["representative_derived_path"] = ""

    # Verify each copied file exists with the same byte length, then tally the
    # final derived split/category counts. copy2 preserves the original bytes.
    final_counts: dict[str, Counter[str]] = {split: Counter() for split in sorted(VALID_SPLITS)}
    for record in keep_records:
        if not record["derived_path"].is_file():
            raise FileNotFoundError(f"Derived image missing after copy: {record['derived_path']}")
        if record["derived_path"].stat().st_size != record["source_path"].stat().st_size:
            raise ValueError(f"Derived copy size differs from original: {record['source_path']}")
        final_counts[record["split"]][record["label"]] += 1

    duplicate_group_count = len(decision_rows)
    consistent_duplicate_count = len(retained_duplicate_rows)
    conflict_group_count = len(diagnostics["conflict_ids"])
    conflict_image_count = len(excluded_rows)
    summary = {
        "dataset": "DermNet",
        "source_raw_directory": diagnostics["raw_root"],
        "source_active_split_directory": diagnostics["active_root"],
        "source_manifest": diagnostics["manifest"],
        "duplicate_review_csv": diagnostics["review_csv"],
        "derived_dataset_directory": str(output_root),
        "resolution_policy": {
            "conflicting_groups": "unresolved; excluded from train, validation, and test",
            "consistent_exact_duplicate_groups": "keep one lexicographically first source_relative_path representative in the existing active split",
            "single_image_groups": "retain in their existing active split",
            "label_inference_or_voting": "none",
            "original_data_modified": False,
        },
        "input_image_count": diagnostics["input_image_count"],
        "input_unique_group_count": diagnostics["input_group_count"],
        "exact_duplicate_group_count": duplicate_group_count,
        "consistent_duplicate_groups_retained_once": consistent_duplicate_count,
        "conflicting_groups_marked_unresolved": conflict_group_count,
        "images_excluded_for_conflicts": conflict_image_count,
        "consistent_duplicate_extra_copies_not_repeated": duplicate_extra_consistent,
        "raw_source_train_test_path_groups": len(diagnostics["raw_train_test_path_groups"]),
        "raw_source_train_test_group_active_split_counts": {
            split: sum(
                1 for group_id in diagnostics["raw_train_test_path_groups"]
                if {row["split"] for row in diagnostics["manifest_by_group"][group_id]} == {split}
            )
            for split in sorted(VALID_SPLITS)
        },
        "active_manifest_cross_split_duplicate_groups": diagnostics["active_cross_split_group_count"],
        "derived_cross_split_duplicate_groups": len(derived_leaks),
        "derived_image_count": copied,
        "derived_image_count_by_split": {split: sum(counts.values()) for split, counts in final_counts.items()},
        "derived_class_counts_by_split": {split: dict(sorted(counts.items(), key=lambda item: item[0].casefold())) for split, counts in final_counts.items()},
        "audit_reports": {name: str(report_root / name) for name in report_names},
    }

    decision_fields = [
        "group_id", "number_of_images_in_group", "original_category_labels", "original_image_filenames",
        "original_image_paths", "active_split", "duplicate_type", "label_conflict_status", "decision",
        "manual_review_status", "representative_original_path", "representative_derived_path",
    ]
    excluded_fields = [
        "group_id", "image_filename", "original_image_path", "original_category_label",
        "group_original_category_labels", "number_of_images_in_group", "active_split", "duplicate_type",
        "label_conflict_status", "exclusion_reason",
    ]
    retained_fields = [
        "group_id", "original_category_label", "group_original_category_labels", "number_of_images_in_group",
        "retained_count", "excluded_exact_copies_count", "active_split", "representative_filename",
        "representative_original_path", "representative_derived_path", "all_original_filenames",
        "all_original_paths", "all_original_category_labels",
    ]
    leakage_fields = [
        "group_id", "number_of_images_in_group", "original_category_labels", "active_splits",
        "derived_split", "label_conflict_status", "cross_split_status", "source_paths",
    ]
    decision_rows.sort(key=lambda row: row["group_id"])
    excluded_rows.sort(key=lambda row: (row["group_id"], row["original_image_path"].casefold()))
    retained_duplicate_rows.sort(key=lambda row: row["group_id"])
    leakage_rows.sort(key=lambda row: row["group_id"])

    write_csv(report_root / report_names[0], decision_fields, decision_rows)
    write_csv(report_root / report_names[1], excluded_fields, excluded_rows)
    write_csv(report_root / report_names[2], retained_fields, retained_duplicate_rows)
    write_csv(report_root / report_names[3], leakage_fields, leakage_rows)
    summary_json = report_root / report_names[4]
    if summary_json.exists():
        raise FileExistsError(f"Refusing to overwrite report: {summary_json}")
    with summary_json.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    summary_txt = report_root / report_names[5]
    if summary_txt.exists():
        raise FileExistsError(f"Refusing to overwrite report: {summary_txt}")
    lines = [
        "DermNet derived dataset summary",
        "================================",
        f"Source images: {summary['input_image_count']:,}",
        f"Unique exact-hash groups: {summary['input_unique_group_count']:,}",
        f"Exact duplicate groups: {duplicate_group_count:,} ({consistent_duplicate_count:,} consistent; {conflict_group_count:,} conflicting)",
        f"Unresolved groups excluded from all splits: {conflict_group_count:,} groups / {conflict_image_count:,} images",
        f"Consistent duplicate extra copies not repeated: {duplicate_extra_consistent:,}",
        f"Final derived images: {copied:,}",
        "",
        "Final images by split:",
    ]
    for split in ("train", "validation", "test"):
        lines.append(f"  {split}: {sum(final_counts[split].values()):,}")
    lines.extend([
        "",
        f"Raw source groups with paths in both raw train and raw test directories: {len(diagnostics['raw_train_test_path_groups']):,}",
        "Those paths are original source locations; each such group has one active processed split assignment.",
        f"Active cross-split duplicate groups: {diagnostics['active_cross_split_group_count']}",
        f"Derived cross-split duplicate groups: {len(derived_leaks)}",
        "",
        f"Derived dataset: {output_root}",
        f"Reports: {report_root}",
        "",
        "Conflicting labels were preserved in the audit reports and not resolved, voted, or changed.",
        "Original images, labels, manifests, and processed splits were not modified.",
    ])
    with summary_txt.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return summary


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    try:
        result = run(project_root)
    except (OSError, ValueError, csv.Error, shutil.Error) as error:
        print(f"DermNet derived build stopped: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
