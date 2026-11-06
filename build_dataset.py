"""Build temporary FlowSIS training data from reviewed manifests.

Masks remain external, versioned NPZ bundles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from PIL import Image

from annotations import image_mask_path, parse_bool, read_csv
from masks import load_mask_bundle, mask2xywh, mask_overlap


class ObjectRecord(TypedDict):
    instance_id: str
    area: float
    bbox: list[int]
    category: int
    tool_name: str
    tool_class: str
    tool_id: str
    mask_path: str
    mask_index: int
    annotation_confidence: float


class ImageRecord(TypedDict):
    image_path: str
    video_path: str
    frame_index: int
    image_id: str
    height: int
    width: int
    tool_ids: list[str]
    sample_id: str
    capture_session_id: str
    objects: list[ObjectRecord]


LABEL_FIELDS = ("tool_name", "tool_class")


def object_label(obj: ObjectRecord, fields: tuple[str, ...]) -> str:
    """Join selected metadata fields while avoiding repeated name suffixes."""
    values = {"tool_name": obj["tool_name"], "tool_class": obj["tool_class"]}
    label = ""
    for field in fields:
        value = " ".join(values[field].split())
        if value and not (
            label.casefold() == value.casefold()
            or label.casefold().endswith(f" {value.casefold()}")
        ):
            label = f"{label} {value}".strip()
    if not label:
        raise ValueError(f"Selected label fields are empty for {obj['instance_id']}.")
    return label


def audit_masks(mask_dir: Path) -> dict:
    """Inspect every saved bundle, including ones outside the accepted subset."""
    counts: Counter[str] = Counter()
    overlaps = []
    for path in sorted(mask_dir.rglob("*.npz")):
        bundle = load_mask_bundle(path)
        counts["bundles"] += 1
        counts["masks"] += len(bundle.masks)
        if len(bundle.masks) < 2:
            continue
        counts["multi_instance_bundles"] += 1
        pixels = int(mask_overlap(bundle.masks).sum())
        if not pixels:
            continue
        pairs = []
        for left in range(len(bundle.masks)):
            for right in range(left + 1, len(bundle.masks)):
                shared = int(np.count_nonzero(bundle.masks[left] & bundle.masks[right]))
                if shared:
                    pairs.append(
                        {
                            "instances": [
                                bundle.identities[left],
                                bundle.identities[right],
                            ],
                            "pixels": shared,
                        }
                    )
        overlaps.append({"path": str(path.resolve()), "pixels": pixels, "pairs": pairs})
    return {**counts, "overlaps": overlaps}


def build_rows(
    root: Path,
    *,
    split_by: Literal["tool", "sample"] = "tool",
    label_fields: tuple[str, ...] = LABEL_FIELDS,
) -> tuple[list[ImageRecord], list[str], dict[str, int]]:
    if split_by not in {"tool", "sample"}:
        raise ValueError("split_by must be 'tool' or 'sample'.")
    if (
        not label_fields
        or len(set(label_fields)) != len(label_fields)
        or set(label_fields) - set(LABEL_FIELDS)
    ):
        raise ValueError(
            f"label_fields must select distinct fields from {LABEL_FIELDS}."
        )
    images = read_csv(root / "image_manifest.csv")
    objects = read_csv(root / "image_object_manifest.csv")
    decisions: dict[str, str] = {}
    for line in (root / "image_review.jsonl").read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            decisions[record["key"]] = record["decision"]
    image_ids = [image["Image ID"] for image in images]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate Image ID in image manifest.")
    unknown = set(decisions) - set(image_ids)
    if unknown:
        raise ValueError(f"Review log refers to unknown images: {sorted(unknown)}")
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for obj in objects:
        if obj["Image ID"] not in image_ids:
            raise ValueError(f"Unknown object image: {obj['Image ID']}")
        by_image[obj["Image ID"]].append(obj)

    rows: list[ImageRecord] = []
    skipped: Counter[str] = Counter()
    for image in images:
        image_id = image["Image ID"]
        if decisions.get(image_id) != "accepted":
            skipped["not_accepted"] += 1
            continue
        sample_id = image["Sample ID"].strip()
        if split_by == "sample" and not sample_id:
            skipped["missing_sample_id"] += 1
            continue
        instances = by_image[image_id]
        if not any(parse_bool(obj.get("Mask Valid")) for obj in instances):
            # Review acceptance alone is insufficient evidence of background.
            # The annotation workflow must also have saved an empty bundle.
            empty_path = image_mask_path(root / "masks/images", image["Filepath"])
            if any(
                parse_bool(obj.get("Visible"))
                or obj.get("Visibility", "") not in {"", "absent", "occluded"}
                for obj in instances
            ):
                raise ValueError(
                    f"Accepted empty image has visible instances: {image_id}"
                )
            bundle = load_mask_bundle(empty_path)
            if len(bundle.masks):
                raise ValueError(f"Accepted empty image has positive masks: {image_id}")
            if bundle.ignore is not None and bundle.ignore.any():
                skipped["ignore_pixels"] += 1
                continue
            image_path = (root / image["Filepath"]).resolve()
            with Image.open(image_path) as loaded:
                width, height = loaded.size
            if bundle.masks.shape[1:] != (height, width):
                raise ValueError(f"Image/mask dimensions disagree for {image_id}")
            rows.append(
                ImageRecord(
                    image_path=str(image_path),
                    video_path="",
                    frame_index=-1,
                    image_id=image_id,
                    height=height,
                    width=width,
                    tool_ids=[],
                    sample_id=sample_id,
                    capture_session_id=image["Capture Session ID"],
                    objects=[],
                )
            )
            continue
        # Never turn a partially labelled image into a background training sample.
        if not instances or any(
            not parse_bool(obj.get("Annotation Complete")) for obj in instances
        ):
            raise ValueError(f"Accepted image {image_id} has incomplete annotations.")
        if any(
            not all(
                parse_bool(obj.get(field))
                for field in (
                    "Reviewed",
                    "Mask Valid",
                    "Class Valid",
                    "Base Valid",
                    "Visible",
                )
            )
            for obj in instances
        ):
            skipped["not_fully_base_valid"] += 1
            continue
        paths = {(root / obj["Mask Path"]).resolve() for obj in instances}
        if any(not obj["Mask Path"] for obj in instances) or len(paths) != 1:
            raise ValueError(f"Image {image_id} must reference one shared mask bundle.")
        path = paths.pop()
        bundle = load_mask_bundle(path)
        if mask_overlap(bundle.masks).any():
            raise ValueError(f"Overlapping instance masks in {path}")
        indices = [int(obj["Mask Index"]) for obj in instances]
        if sorted(indices) != list(range(len(bundle.masks))):
            raise ValueError(f"Mask indices do not cover bundle exactly once: {path}")
        if len(set(bundle.identities)) != len(bundle.identities):
            raise ValueError(f"Duplicate instance identities in {path}")
        if bundle.ignore is not None and bundle.ignore.any():
            skipped["ignore_pixels"] += 1
            continue
        image_path = (root / image["Filepath"]).resolve()
        with Image.open(image_path) as loaded:
            width, height = loaded.size
        if bundle.masks.shape[1:] != (height, width):
            raise ValueError(f"Image/mask dimensions disagree for {image_id}")
        row_objects: list[ObjectRecord] = []
        for obj in instances:
            mask_index = int(obj["Mask Index"])
            if bundle.identities[mask_index] != obj["Instance ID"]:
                raise ValueError(f"Mask identity disagrees with manifest: {image_id}")
            box = mask2xywh(bundle.masks[mask_index])
            recorded_box = [
                int(obj[f"BBox {field}"]) for field in ("X", "Y", "Width", "Height")
            ]
            if box is None or box != recorded_box:
                raise ValueError(f"Empty mask or incorrect manifest box: {image_id}")
            if not obj["Tool ID"].strip() or not obj["Tool Class"].strip():
                raise ValueError(f"Missing physical tool identity or class: {image_id}")
            confidence = float(obj["Annotation Confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError(f"Invalid annotation confidence: {image_id}")
            row_objects.append(
                ObjectRecord(
                    instance_id=obj["Instance ID"],
                    area=float(np.count_nonzero(bundle.masks[mask_index])),
                    bbox=box,
                    category=-1,
                    tool_name=obj["Tool Name"],
                    tool_class=obj["Tool Class"],
                    tool_id=obj["Tool ID"],
                    mask_path=str(path),
                    mask_index=mask_index,
                    annotation_confidence=confidence,
                )
            )
        tool_ids = sorted({obj["Tool ID"] for obj in instances})
        if len(tool_ids) != len(instances):
            raise ValueError(f"Duplicate Tool ID within image: {image_id}")
        rows.append(
            ImageRecord(
                image_path=str(image_path),
                video_path="",
                frame_index=-1,
                image_id=image_id,
                height=height,
                width=width,
                tool_ids=tool_ids,
                sample_id=sample_id,
                capture_session_id=image["Capture Session ID"],
                objects=row_objects,
            )
        )
    names = sorted(
        {object_label(obj, label_fields) for row in rows for obj in row["objects"]}
    )
    label_ids = {label: index for index, label in enumerate(names)}
    for row in rows:
        for obj in row["objects"]:
            obj["category"] = label_ids[object_label(obj, label_fields)]
    if not rows:
        raise ValueError("No accepted images with complete, base-valid masks.")
    return rows, names, dict(skipped)


def split_rows(
    rows: list[ImageRecord],
    sizes: dict[str, float],
    seed: int,
    *,
    split_by: Literal["tool", "sample"] = "tool",
) -> list[str]:
    """Keep groups intact and reserve training coverage before allocating holdouts."""
    if split_by not in {"tool", "sample"}:
        raise ValueError("split_by must be 'tool' or 'sample'.")
    if (
        any(not math.isfinite(size) or size < 0 for size in sizes.values())
        or abs(sum(sizes.values()) - 1) > 1e-6
    ):
        raise ValueError("Split sizes must be nonnegative and sum to one.")
    if sizes.get("train", 0) <= 0:
        raise ValueError("A positive training fraction is required for class coverage.")
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        keys = [("sample", row["sample_id"])]
        if split_by == "tool":
            keys.extend(("tool", tool) for tool in row["tool_ids"])
        elif not row["sample_id"]:
            raise ValueError(
                f"Sample-based splitting requires a Sample ID: {row['image_id']}"
            )
        for key in keys:
            if not key[1]:
                continue
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    active = [split for split, size in sizes.items() if size > 0]
    components = list(groups.values())
    random.Random(seed).shuffle(components)
    if split_by == "tool":
        components.sort(key=len, reverse=True)
    counts: Counter[str] = Counter()
    assignments = [""] * len(rows)

    # Cover rare labels first. A singleton label forces its entire group into
    # training; requested fractions never override class coverage or grouping.
    group_labels = [
        {obj["category"] for index in indices for obj in rows[index]["objects"]}
        for indices in components
    ]
    label_groups: dict[int, list[int]] = defaultdict(list)
    for group_index, labels in enumerate(group_labels):
        for label in sorted(labels):
            label_groups[label].append(group_index)
    uncovered = set(label_groups)
    reserved: set[int] = set()
    while uncovered:
        label = min(uncovered, key=lambda value: (len(label_groups[value]), value))
        selected = min(
            label_groups[label],
            key=lambda index: (
                -len(group_labels[index] & uncovered),
                len(components[index]),
                index,
            ),
        )
        reserved.add(selected)
        uncovered.difference_update(group_labels[selected])
    # Background-only exports still need a nonempty training set.
    if not reserved and components:
        reserved.add(0)
    for group_index in sorted(reserved):
        indices = components[group_index]
        counts["train"] += len(indices)
        for index in indices:
            assignments[index] = "train"
    remaining = [
        indices for index, indices in enumerate(components) if index not in reserved
    ]
    for position, indices in enumerate(remaining):
        empty = [split for split in active if not counts[split]]
        candidates = (
            empty if empty and len(remaining) - position <= len(empty) else active
        )
        split = max(candidates, key=lambda name: len(rows) * sizes[name] - counts[name])
        counts[split] += len(indices)
        for index in indices:
            assignments[index] = split
    return assignments


def dataset_features(names: list[str]) -> Features:
    return Features(
        {
            "image_path": Value("string"),
            "video_path": Value("string"),
            "frame_index": Value("int64"),
            "image_id": Value("string"),
            "height": Value("int64"),
            "width": Value("int64"),
            "tool_ids": Sequence(Value("string")),
            "sample_id": Value("string"),
            "capture_session_id": Value("string"),
            "objects": [
                {
                    "instance_id": Value("string"),
                    "area": Value("float32"),
                    "bbox": Sequence(Value("float32"), length=4),
                    "category": ClassLabel(names=names),
                    "tool_name": Value("string"),
                    "tool_class": Value("string"),
                    "tool_id": Value("string"),
                    "mask_path": Value("string"),
                    "mask_index": Value("int64"),
                    "annotation_confidence": Value("float32"),
                }
            ],
        }
    )


def build_dataset(
    root: Path,
    output: Path,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    force: bool = False,
    split_by: Literal["tool", "sample"] = "tool",
    label_fields: tuple[str, ...] = LABEL_FIELDS,
) -> DatasetDict:
    root, output = root.resolve(), output.resolve()
    if output.exists() and not force:
        raise FileExistsError(
            f"Output already exists; use --force to replace it: {output}"
        )
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    asset_dirs = [root / name for name in ("images", "masks", "raw-images")]
    if (
        output == root
        or output in root.parents
        or any(output == path or path in output.parents for path in asset_dirs)
    ):
        raise ValueError(
            f"Output must not replace or be inside source asset directories: {output}"
        )
    audit = audit_masks(root / "masks")
    if audit["overlaps"]:
        report_path = output.with_suffix(".mask-audit.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(audit, indent=2) + "\n")
        raise ValueError(f"Overlapping masks found; build stopped. See {report_path}")
    rows, names, skipped = build_rows(
        root, split_by=split_by, label_fields=label_fields
    )
    sizes = {
        "train": 1 - validation_size - test_size,
        "validation": validation_size,
        "test": test_size,
    }
    assignments = split_rows(rows, sizes, seed, split_by=split_by)
    for row in rows:
        row["image_path"] = Path(os.path.relpath(row["image_path"], output)).as_posix()
        for obj in row["objects"]:
            obj["mask_path"] = Path(
                os.path.relpath(obj["mask_path"], output)
            ).as_posix()
    data = Dataset.from_list(rows, features=dataset_features(names))
    result = DatasetDict(
        {
            split: data.select(
                [i for i, assigned in enumerate(assignments) if assigned == split]
            )
            for split, size in sizes.items()
            if size > 0
        }
    )
    split_summary = {}
    train_labels = {
        obj["category"]
        for row, split in zip(rows, assignments, strict=True)
        if split == "train"
        for obj in row["objects"]
    }
    for split, subset in result.items():
        class_counts = Counter(
            names[obj["category"]] for row in subset for obj in row["objects"]
        )
        split_summary[split] = {
            "images": len(subset),
            "samples": len({row["sample_id"] for row in subset if row["sample_id"]}),
            "masks": sum(len(row["objects"]) for row in subset),
            "multi_tool_images": sum(len(row["objects"]) > 1 for row in subset),
            "background_images": sum(not row["objects"] for row in subset),
            "classes": dict(class_counts),
            "classes_absent": [name for name in names if name not in class_counts],
        }
    report = {
        "source": str(root),
        "seed": seed,
        "requested_fractions": sizes,
        "split_by": split_by,
        "split_grouping": (
            ["physical Tool ID", "Sample ID"] if split_by == "tool" else ["Sample ID"]
        ),
        "classes": names,
        "label_fields": list(label_fields),
        "split_policy": "training_class_coverage_first",
        "empty_splits": [split for split, subset in result.items() if not len(subset)],
        "classes_absent_from_train": [
            name for i, name in enumerate(names) if i not in train_labels
        ],
        "skipped": skipped,
        "mask_audit": audit,
        "splits": split_summary,
        "augmentation_recommendation": {
            "rotation_degrees": [0, 45],
            "overlap_min_overlays": 0,
            "overlap_max_overlays": 1,
            "overlap_p": 0.25,
        },
        "external_assets": "Paths are relative to the dataset directory; retain its layout relative to images and masks.",
    }
    # Omit empty holdouts: trainers can skip missing splits, whereas evaluating
    # an empty split can misleadingly report zero loss. Keep them in the report.
    result = DatasetDict(
        {split: subset for split, subset in result.items() if len(subset)}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.name}-"
    ) as temporary:
        staged = Path(temporary) / "dataset"
        previous = Path(temporary) / "previous"
        result.save_to_disk(staged)
        (staged / "asset_paths.json").write_text(
            json.dumps({"version": 1, "relative_to": "dataset_directory"}, indent=2)
            + "\n"
        )
        (staged / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
        if output.exists():
            if not force:
                raise FileExistsError(f"Output appeared during the build: {output}")
            output.rename(previous)
        try:
            staged.rename(output)
        except OSError:
            if previous.exists():
                previous.rename(output)
            raise
    print(json.dumps(report, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument(
        "--output-path", type=Path, default=Path("exports/segmentation-dataset")
    )
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-fields",
        nargs="+",
        choices=LABEL_FIELDS,
        default=list(LABEL_FIELDS),
        help="Ordered object metadata fields used to construct class labels.",
    )
    parser.add_argument(
        "--split-by",
        choices=("tool", "sample"),
        default="tool",
        help="Group by physical Tool IDs and Sample IDs (tool), or only by Sample IDs (sample).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output dataset after the new build succeeds.",
    )
    args = parser.parse_args()
    build_dataset(
        args.root,
        args.output_path,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
        force=args.force,
        split_by=args.split_by,
        label_fields=tuple(args.label_fields),
    )


if __name__ == "__main__":
    main()
