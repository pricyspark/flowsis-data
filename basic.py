import csv
import unicodedata
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from PIL import ExifTags, Image, UnidentifiedImageError
import pillow_heif

ROOT = Path(__file__).resolve().parent
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
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in value
    )
    return " ".join(without_punctuation.lower().split())


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
        "Tool ID",
        "Sample ID",
        "Filepath",
        "Tool Class",
        "Tool Name",
        "Confidence",
        "Angle",
        *METADATA_COLUMNS,
    }
    missing_columns = required_columns.difference(fieldnames)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        print(f"{manifest_path.name} is missing columns: {missing}")
        return

    normalized_labels = 0
    for row in rows:
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
        (path for path in images_dir.iterdir() if path.is_dir()),
        key=lambda path: sort_key(path.name),
    )
    rows_by_tool = {
        tool_dir.name: [row for row in rows if row["Tool ID"] == tool_dir.name]
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

        if existing_rows:
            tool_class = next(
                (row["Tool Class"] for row in existing_rows if row["Tool Class"]),
                "",
            )
            tool_name = next(
                (row["Tool Name"] for row in existing_rows if row["Tool Name"]),
                "",
            )
            confidence = next(
                (row["Confidence"] for row in existing_rows if row["Confidence"]),
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
                f"confidence={confidence}."
            )
        else:
            print(f"\nTool ID {tool_id} has {len(images)} image(s).")
            tool_class = prompt_required("Tool class")
            tool_name = prompt_required("Tool name")
            confidence = prompt_confidence()

        new_rows_by_path: dict[str, dict[str, str]] = {}
        for image in undocumented_images:
            filepath = image.relative_to(manifest_path.parent).as_posix()
            row = dict.fromkeys(fieldnames, "")
            row.update(
                {
                    "Tool ID": tool_id,
                    "Filepath": filepath,
                    "Tool Class": tool_class,
                    "Tool Name": tool_name,
                    "Confidence": confidence,
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

    if not added_rows and not metadata_updates and not normalized_labels:
        print("The manifest was not changed.")
        return

    rows.sort(
        key=lambda row: (
            sort_key(row["Tool ID"]),
            sort_key(row["Sample ID"]),
            sort_key(row["Angle"]),
            row["Filepath"].lower(),
        )
    )

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if metadata_updates:
        print(f"Filled {metadata_updates} empty metadata field(s).")
    if normalized_labels:
        print(f"Normalized {normalized_labels} tool class or name field(s).")
    if added_rows:
        print(f"Added {added_rows} row(s) to {manifest_path.name}.")


if __name__ == "__main__":
    main()
