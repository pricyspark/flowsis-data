import csv
from fractions import Fraction
from pathlib import Path

from PIL import ExifTags, Image, UnidentifiedImageError
import pillow_heif


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
RAW_IMAGES_DIR = ROOT / "raw-images"
MANIFEST_PATH = ROOT / "image_manifest.csv"
IMAGE_EXTENSIONS = {".avif", ".heic", ".jpeg", ".jpg", ".png", ".webp"}
ANGLES = tuple(range(0, 360, 45))
METADATA_COLUMNS = (
    "Make",
    "Model",
    "Flash",
    "ISO",
    "Exposure",
    "Aperture",
    "Focal length",
)


def sort_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value.lower()


def prompt_required(label: str) -> str:
    while True:
        value = input(f"{label}: ").strip()
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


def find_images(tool_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in tool_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(tool_dir)).lower(),
    )


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
        "Focal length": format_decimal(camera.get(ExifTags.Base.FocalLength)),
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

    existing_tool_ids = {row["Tool ID"] for row in rows}
    tool_dirs = sorted(
        (path for path in images_dir.iterdir() if path.is_dir()),
        key=lambda path: sort_key(path.name),
    )
    new_tool_dirs = [path for path in tool_dirs if path.name not in existing_tool_ids]

    if new_tool_dirs:
        print("New tool IDs found: " + ", ".join(path.name for path in new_tool_dirs))
    else:
        print("No new tool IDs found.")

    sample_ids = [
        int(row["Sample ID"])
        for row in rows
        if row["Sample ID"].strip().isdigit()
    ]
    next_sample_id = max(sample_ids, default=-1) + 1
    added_rows = 0

    for tool_dir in new_tool_dirs:
        tool_id = tool_dir.name
        images = find_images(tool_dir)

        if not images:
            print(f"\nTool ID {tool_id} has no image files. Skipping it.")
            continue

        print(f"\nTool ID {tool_id} has {len(images)} image(s).")
        tool_class = prompt_required("Tool class")
        tool_name = prompt_required("Tool name")
        confidence = prompt_confidence()

        has_complete_groups = len(images) % len(ANGLES) == 0
        if has_complete_groups:
            group_count = len(images) // len(ANGLES)
            first_sample_id = next_sample_id
            print(
                f"Assigning {group_count} sample ID(s), starting at "
                f"{first_sample_id}, with angles 0 through 315 degrees."
            )
        else:
            print(
                f"Warning: {len(images)} is not a multiple of 8. "
                "Sample ID and angle will be left blank for this tool."
            )

        for index, image in enumerate(images):
            row = dict.fromkeys(fieldnames, "")
            row.update(
                {
                    "Tool ID": tool_id,
                    "Filepath": image.relative_to(manifest_path.parent).as_posix(),
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
                except (OSError, UnidentifiedImageError, TypeError, ValueError) as error:
                    print(f"Warning: could not read metadata from {source}: {error}")

            if has_complete_groups:
                row["Sample ID"] = str(next_sample_id + index // len(ANGLES))
                row["Angle"] = str(ANGLES[index % len(ANGLES)])

            rows.append(row)
            added_rows += 1

        if has_complete_groups:
            next_sample_id += len(images) // len(ANGLES)

    if not added_rows and not metadata_updates:
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
    if added_rows:
        print(f"Added {added_rows} row(s) to {manifest_path.name}.")


if __name__ == "__main__":
    main()
