"""Prepare image manifests and object rows sorted by physical Tool ID."""

import argparse
import csv
import unicodedata
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from annotations import (
    IMAGE_OBJECT_FIELDS,
    assign_capture_sessions,
    read_csv,
    stable_media_id,
    validate_instance_metadata,
    write_image_manifests,
)

from PIL import ExifTags, Image, UnidentifiedImageError
import pillow_heif

ROOT = Path(__file__).resolve().parent / "data"
IMAGES_DIR = ROOT / "images"
RAW_IMAGES_DIR = ROOT / "raw-images"
MANIFEST_PATH = ROOT / "image_manifest.csv"
IMAGE_EXTENSIONS = {
    ".avif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}
ANGLES = tuple(range(0, 360, 45))
METADATA_COLUMNS = (
    "Make",
    "Model",
    "Flash",
    "ISO",
    "Exposure",
    "Aperture",
    "Focal length (mm)",
    "Time Taken",
)


def sort_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value.lower()


def normalize_label(value: str) -> str:
    separated = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in value
    )
    return "_".join(separated.lower().split())


def prompt_required(label: str) -> str:
    while True:
        value = normalize_label(input(f"{label}: "))
        if value:
            return value
        print("Please enter a value.")


def prompt_confidence() -> str:
    while True:
        value = input("Confidence (0 to 1): ").strip()
        try:
            confidence = float(value)
        except ValueError:
            print("Please enter a number from 0 to 1, such as 0.8 or 1.")
            continue

        if 0 <= confidence <= 1:
            return f"{confidence:g}"
        print("Confidence must be between 0 and 1.")


def prompt_continue(tool_id: str, image_count: int) -> bool:
    while True:
        response = (
            input(
                f"Tool ID {tool_id} already has manifest entries but also has "
                f"{image_count} undocumented image(s). Continue and add them? [y/N]: "
            )
            .strip()
            .lower()
        )
        if response in {"y", "yes"}:
            return True
        if response in {"", "n", "no"}:
            return False
        print("Please enter y for yes or n for no.")


def find_images(tool_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in tool_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(tool_dir)).lower(),
    )


def find_image_slots(
    tool_dir: Path,
    images_dir: Path,
    raw_images_dir: Path,
) -> list[Path]:
    """Return expected JPG paths ordered by their raw source filenames."""
    raw_tool_dir = raw_images_dir / tool_dir.relative_to(images_dir)
    if not raw_tool_dir.is_dir():
        return find_images(tool_dir)

    sources = find_images(raw_tool_dir)
    if not sources:
        return find_images(tool_dir)

    slots = []
    seen = set()
    for source in sources:
        destination = (images_dir / source.relative_to(raw_images_dir)).with_suffix(
            ".jpg"
        )
        if destination not in seen:
            slots.append(destination)
            seen.add(destination)

    for image in find_images(tool_dir):
        if image not in seen:
            slots.append(image)
            seen.add(image)
    return slots


def assign_complete_batches(
    image_slots: list[Path],
    rows_by_path: dict[str, dict[str, str]],
    manifest_dir: Path,
    next_sample_id: int,
) -> tuple[int, int, int]:
    """Assign complete eight-image slots without shifting past missing images."""
    group_count = 0
    incomplete_rows = 0
    for start in range(0, len(image_slots), len(ANGLES)):
        image_batch = image_slots[start : start + len(ANGLES)]
        batch_rows = [
            (
                rows_by_path.get(image.relative_to(manifest_dir).as_posix())
                if image.is_file()
                else None
            )
            for image in image_batch
        ]
        ungrouped_rows = [
            row
            for row in batch_rows
            if row and not row["Sample ID"] and not row["Angle"]
        ]

        if len(image_batch) == len(ANGLES) and len(ungrouped_rows) == len(ANGLES):
            for angle, row in zip(ANGLES, ungrouped_rows):
                row["Sample ID"] = str(next_sample_id)
                row["Angle"] = str(angle)
            next_sample_id += 1
            group_count += 1
        else:
            incomplete_rows += len(ungrouped_rows)

    return next_sample_id, group_count, incomplete_rows


def assign_multi_tool_samples(
    rows: list[dict[str, str]],
    objects_by_image: dict[str, list[dict[str, str]]],
    multi_relative: Path,
    undeclared_paths: set[str],
) -> int:
    """Assign one sample per fully declared setup folder, preserving existing IDs."""

    def setup_folder(filepath: str) -> Path | None:
        path = Path(filepath)
        if not path.is_relative_to(multi_relative):
            return None
        relative = path.relative_to(multi_relative)
        if len(relative.parts) < 2 or ".." in relative.parts:
            return None
        return multi_relative / relative.parts[0]

    setups: dict[Path, list[dict[str, str]]] = {}
    sample_images: dict[str, set[str]] = {}
    for row in rows:
        sample = row["Sample ID"]
        if sample.strip():
            sample_images.setdefault(sample, set()).add(row["Image ID"])
        setup = setup_folder(row["Filepath"])
        if setup is not None:
            setups.setdefault(setup, []).append(row)
        elif Path(row["Filepath"]).is_relative_to(multi_relative):
            print(
                f"Warning: no setup subfolder for {row['Filepath']}; Sample ID preserved."
            )

    blocked = {setup_folder(filepath) for filepath in undeclared_paths}
    next_sample_id = (
        max(
            (int(sample) for sample in sample_images if sample.strip().isdigit()),
            default=-1,
        )
        + 1
    )
    assigned = 0
    for setup, images in sorted(setups.items()):
        try:
            if setup in blocked:
                raise ValueError("declare every image in the setup folder first")
            tool_sets = set()
            for image in images:
                instances = objects_by_image.get(image["Image ID"], [])
                validate_instance_metadata(instances)
                if len(instances) < 2:
                    raise ValueError("each image must declare at least two tools")
                tool_sets.add(frozenset(obj["Tool ID"] for obj in instances))
            if len(tool_sets) != 1:
                raise ValueError("declared Tool IDs differ between images")
            samples = {
                image["Sample ID"] for image in images if image["Sample ID"].strip()
            }
            if len(samples) > 1:
                raise ValueError("folder contains multiple existing Sample IDs")
            sample = next(iter(samples)) if samples else str(next_sample_id)
            image_ids = {image["Image ID"] for image in images}
            if sample_images.get(sample, set()) - image_ids:
                raise ValueError(
                    f"Sample ID {sample} is also used outside this setup folder"
                )
        except ValueError as error:
            print(f"Warning: skipping sample assignment for {setup}: {error}.")
            continue

        if not samples:
            next_sample_id += 1
        unassigned = [image for image in images if not image["Sample ID"].strip()]
        for image in unassigned:
            image["Sample ID"] = sample
        if unassigned:
            assigned += len(unassigned)
            print(
                f"Assigned Sample ID {sample} to {len(unassigned)} image(s) in {setup}."
            )
    return assigned


def find_original(image: Path, images_dir: Path, raw_images_dir: Path) -> Path | None:
    relative_path = image.relative_to(images_dir)
    source_dir = raw_images_dir / relative_path.parent
    if not source_dir.is_dir():
        return None

    candidates = [
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.stem.lower() == image.stem.lower()
    ]
    if not candidates:
        return None

    preferred_extensions = {".heic": 0, ".heif": 1}
    return min(
        candidates,
        key=lambda path: (preferred_extensions.get(path.suffix.lower(), 2), path.name),
    )


def format_decimal(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def format_exposure(value: object) -> str:
    if value is None:
        return ""
    try:
        exposure = Fraction(value).limit_denominator(1_000_000)
    except (TypeError, ValueError, ZeroDivisionError):
        return str(value)
    if exposure.denominator == 1:
        return str(exposure.numerator)
    return f"{exposure.numerator}/{exposure.denominator}"


def format_time_taken(exif: Image.Exif, camera: dict[int, object]) -> str:
    raw_time = camera.get(ExifTags.Base.DateTimeOriginal) or exif.get(
        ExifTags.Base.DateTime
    )
    if not raw_time:
        return ""

    raw_time = str(raw_time).strip()
    try:
        timestamp = datetime.strptime(raw_time, "%Y:%m:%d %H:%M:%S").strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    except ValueError:
        return raw_time

    subseconds = str(camera.get(ExifTags.Base.SubsecTimeOriginal, "")).strip()
    if subseconds.isdigit():
        timestamp += f".{subseconds}"

    offset = str(
        camera.get(ExifTags.Base.OffsetTimeOriginal)
        or exif.get(ExifTags.Base.OffsetTime)
        or ""
    ).strip()
    if offset:
        timestamp += offset
    return timestamp


def extract_metadata(source: Path) -> dict[str, str]:
    with Image.open(source) as image:
        exif = image.getexif()
        camera = exif.get_ifd(ExifTags.IFD.Exif)

    flash = camera.get(ExifTags.Base.Flash)
    return {
        "Make": str(exif.get(ExifTags.Base.Make, "")),
        "Model": str(exif.get(ExifTags.Base.Model, "")),
        "Flash": "" if flash is None else str(bool(int(flash) & 1)).upper(),
        "ISO": format_decimal(camera.get(ExifTags.Base.ISOSpeedRatings)),
        "Exposure": format_exposure(camera.get(ExifTags.Base.ExposureTime)),
        "Aperture": format_decimal(camera.get(ExifTags.Base.FNumber)),
        "Focal length (mm)": format_decimal(camera.get(ExifTags.Base.FocalLength)),
        "Time Taken": format_time_taken(exif, camera),
    }


def main(
    images_dir: Path = IMAGES_DIR,
    raw_images_dir: Path = RAW_IMAGES_DIR,
    manifest_path: Path = MANIFEST_PATH,
    object_manifest_path: Path | None = None,
    *,
    renumber_samples: bool = False,
) -> None:
    if not images_dir.is_dir():
        print(f"Could not find {images_dir.name}/. Run img_convert.py first.")
        return
    if not raw_images_dir.is_dir():
        print(f"Could not find {raw_images_dir.name}/.")
        return
    if not manifest_path.is_file():
        print(f"Could not find {manifest_path.name}.")
        return

    pillow_heif.register_heif_opener()

    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        print(f"{manifest_path.name} does not have a header row.")
        return

    if "Time Taken" not in fieldnames:
        fieldnames.append("Time Taken")
        for row in rows:
            row["Time Taken"] = ""

    required_columns = {
        "Image ID",
        "Capture Session ID",
        "Sample ID",
        "Filepath",
        "Angle",
        *METADATA_COLUMNS,
    }
    missing_columns = required_columns.difference(fieldnames)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        print(f"{manifest_path.name} is missing columns: {missing}")
        return

    object_manifest_path = object_manifest_path or manifest_path.with_name(
        "image_object_manifest.csv"
    )
    objects = read_csv(object_manifest_path)
    objects_by_image: dict[str, list[dict[str, str]]] = {}
    for obj in objects:
        objects_by_image.setdefault(obj["Image ID"], []).append(obj)
    multi_dir = images_dir / "multi"
    multi_relative = multi_dir.relative_to(manifest_path.parent)
    missing_instances = [
        row["Filepath"]
        for row in rows
        if not objects_by_image.get(row["Image ID"])
        and not Path(row["Filepath"]).is_relative_to(multi_relative)
    ]
    if missing_instances:
        raise ValueError(f"Images have no instance metadata: {missing_instances}")

    # multi/ is organizational: its name must never become a physical Tool ID.
    by_filepath = {row["Filepath"]: row for row in rows}
    undeclared_multi = {
        row["Filepath"]
        for row in rows
        if Path(row["Filepath"]).is_relative_to(multi_relative)
        and not objects_by_image.get(row["Image ID"])
    }
    for image in find_images(multi_dir):
        filepath = image.relative_to(manifest_path.parent).as_posix()
        if filepath not in by_filepath:
            undeclared_multi.add(filepath)
    if undeclared_multi:
        print(
            "Multi-tool images require explicit instance declarations (automatic intake skipped):"
        )
        for filepath in sorted(undeclared_multi):
            print(f"- {filepath}")
        print(
            "Run python annotate.py declare --folder SETUP --tool-ids ID [ID ...] "
            "using tools already identified in image_object_manifest.csv. "
            "Then rerun prepare_manifests.py to assign one Sample ID per fully "
            "declared setup folder. Leave randomized multi-tool Angle blank."
        )

    normalized_labels = 0
    for row in objects:
        for column in ("Tool Class", "Tool Name"):
            normalized = normalize_label(row[column])
            if normalized != row[column]:
                row[column] = normalized
                normalized_labels += 1

    metadata_updates = 0
    for row in rows:
        if all(row[column].strip() for column in METADATA_COLUMNS):
            continue

        image = manifest_path.parent / row["Filepath"]
        try:
            source = find_original(image, images_dir, raw_images_dir)
        except ValueError:
            source = None
        if source is None:
            print(f"Warning: no original image found for {row['Filepath']}.")
            continue

        try:
            metadata = extract_metadata(source)
        except (OSError, UnidentifiedImageError, TypeError, ValueError) as error:
            print(f"Warning: could not read metadata from {source}: {error}")
            continue

        for column, value in metadata.items():
            if not row[column].strip() and value:
                row[column] = value
                metadata_updates += 1

    tool_dirs = sorted(
        (
            path
            for path in images_dir.iterdir()
            if path.is_dir() and path.name != "multi"
        ),
        key=lambda path: sort_key(path.name),
    )
    rows_by_tool = {
        tool_dir.name: [
            row for row in rows if Path(row["Filepath"]).parts[1] == tool_dir.name
        ]
        for tool_dir in tool_dirs
    }
    new_tool_ids = [
        tool_id for tool_id, tool_rows in rows_by_tool.items() if not tool_rows
    ]

    if new_tool_ids:
        print("New tool IDs found: " + ", ".join(new_tool_ids))
    else:
        print("No new tool IDs found.")

    sample_ids = [
        int(row["Sample ID"]) for row in rows if row["Sample ID"].strip().isdigit()
    ]
    next_sample_id = max(sample_ids, default=-1) + 1
    added_rows = 0

    for tool_dir in tool_dirs:
        tool_id = tool_dir.name
        images = find_images(tool_dir)
        image_slots = find_image_slots(tool_dir, images_dir, raw_images_dir)
        existing_rows = rows_by_tool[tool_id]

        if not images:
            if not existing_rows:
                print(f"\nTool ID {tool_id} has no image files. Skipping it.")
            continue

        documented_paths = {row["Filepath"] for row in existing_rows}
        undocumented_images = [
            image
            for image in images
            if image.relative_to(manifest_path.parent).as_posix()
            not in documented_paths
        ]
        if not undocumented_images:
            continue

        if existing_rows:
            print(
                f"\nWarning: tool ID {tool_id} contains "
                f"{len(undocumented_images)} image(s) that are not in the manifest."
            )
            if not prompt_continue(tool_id, len(undocumented_images)):
                print(f"Skipping tool ID {tool_id}.")
                continue

        existing_objects = [
            obj for row in existing_rows for obj in objects_by_image[row["Image ID"]]
        ]
        if any(len(objects_by_image[row["Image ID"]]) != 1 for row in existing_rows):
            raise ValueError(
                f"Folder {tool_id} contains multi-tool images. Add instance metadata "
                "explicitly; prepare_manifests.py only infers metadata for single-tool folders."
            )
        if existing_rows:
            tool_class = next(
                (row["Tool Class"] for row in existing_objects if row["Tool Class"]),
                "",
            )
            tool_name = next(
                (row["Tool Name"] for row in existing_objects if row["Tool Name"]),
                "",
            )
            confidence = next(
                (row["Confidence"] for row in existing_objects if row["Confidence"]),
                "",
            )
            manufacturer = next(
                (
                    row["Manufacturer"]
                    for row in existing_objects
                    if row.get("Manufacturer", "").strip()
                ),
                "",
            )
            if not tool_class:
                tool_class = prompt_required("Tool class")
            if not tool_name:
                tool_name = prompt_required("Tool name")
            if not confidence:
                confidence = prompt_confidence()
            print(
                f"Using existing details: class={tool_class}, name={tool_name}, "
                f"manufacturer={manufacturer or 'unknown'}, confidence={confidence}."
            )
        else:
            print(f"\nTool ID {tool_id} has {len(images)} image(s).")
            tool_class = prompt_required("Tool class")
            tool_name = prompt_required("Tool name")
            manufacturer = input(
                "Manufacturer (optional; Enter to leave blank): "
            ).strip()
            confidence = prompt_confidence()

        new_rows_by_path: dict[str, dict[str, str]] = {}
        for image in undocumented_images:
            filepath = image.relative_to(manifest_path.parent).as_posix()
            row = dict.fromkeys(fieldnames, "")
            row.update(
                {
                    "Image ID": stable_media_id("image", filepath),
                    "Filepath": filepath,
                }
            )

            source = find_original(image, images_dir, raw_images_dir)
            if source is None:
                print(f"Warning: no original image found for {image.name}.")
            else:
                try:
                    row.update(extract_metadata(source))
                except (
                    OSError,
                    UnidentifiedImageError,
                    TypeError,
                    ValueError,
                ) as error:
                    print(f"Warning: could not read metadata from {source}: {error}")

            obj = dict.fromkeys(IMAGE_OBJECT_FIELDS, "")
            obj.update(
                {
                    "Image ID": row["Image ID"],
                    "Instance ID": "instance-0",
                    "Tool ID": tool_id,
                    "Tool Class": tool_class,
                    "Tool Name": tool_name,
                    "Manufacturer": manufacturer,
                    "Confidence": confidence,
                }
            )
            objects.append(obj)
            objects_by_image[row["Image ID"]] = [obj]
            rows.append(row)
            new_rows_by_path[filepath] = row
            added_rows += 1

        rows_for_path = {row["Filepath"]: row for row in existing_rows}
        rows_for_path.update(new_rows_by_path)
        first_sample_id = next_sample_id
        next_sample_id, group_count, incomplete_rows = assign_complete_batches(
            image_slots,
            rows_for_path,
            manifest_path.parent,
            next_sample_id,
        )

        if group_count:
            print(
                f"Assigning {group_count} sample ID(s), starting at "
                f"{first_sample_id}, with angles 0 through 315 degrees."
            )
        if incomplete_rows:
            print(
                f"Warning: {incomplete_rows} ungrouped image(s) belong to "
                "incomplete eight-image batch(es). Sample ID and angle will be "
                "left blank; later complete batches are unaffected."
            )

    assign_multi_tool_samples(rows, objects_by_image, multi_relative, undeclared_multi)

    for row, session in zip(rows, assign_capture_sessions(rows), strict=True):
        if not row["Capture Session ID"]:
            row["Capture Session ID"] = session
    if renumber_samples:
        mapping = renumber_sample_ids(rows, objects)
        print(
            f"Renumbered {sum(old != new for old, new in mapping.items())} sample ID(s) by tool ID."
        )
    write_image_manifests(manifest_path, rows, object_manifest_path, objects)
    print(
        "Refreshed manifest navigation columns. Image rows are sorted by completion "
        "and tool IDs; object rows are sorted by tool ID."
    )

    if metadata_updates:
        print(f"Filled {metadata_updates} empty metadata field(s).")
    if normalized_labels:
        print(f"Normalized {normalized_labels} tool class or name field(s).")
    if added_rows:
        print(f"Added {added_rows} row(s) to {manifest_path.name}.")


def renumber_sample_ids(
    rows: list[dict[str, str]], objects: list[dict[str, str]]
) -> dict[str, str]:
    """Assign contiguous sample IDs by physical tools without changing membership."""
    tools_by_image: dict[str, set[str]] = {}
    for obj in objects:
        tools_by_image.setdefault(obj["Image ID"], set()).add(obj["Tool ID"])
    tools_by_sample: dict[str, set[str]] = {}
    for row in rows:
        if row["Sample ID"].strip():
            tools_by_sample.setdefault(row["Sample ID"], set()).update(
                tools_by_image.get(row["Image ID"], set())
            )
    ordered = sorted(
        tools_by_sample,
        key=lambda sample: (
            tuple(sorted(sort_key(tool) for tool in tools_by_sample[sample])),
            sort_key(sample),
        ),
    )
    mapping = {sample: str(index) for index, sample in enumerate(ordered)}
    for row in rows:
        row["Sample ID"] = mapping.get(row["Sample ID"], row["Sample ID"])
    # write_image_manifests refreshes the object copies from their image rows.
    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renumber-samples",
        action="store_true",
        help="renumber assigned samples consecutively by tool ID, preserving groups",
    )
    main(renumber_samples=parser.parse_args().renumber_samples)
