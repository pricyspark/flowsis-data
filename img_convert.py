import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import pillow_heif

from square_crop import MarkerDetectionError, crop_marked_square

ROOT = Path(__file__).resolve().parent / "data"
SOURCE_DIR = ROOT / "raw-images"
OUTPUT_DIR = ROOT / "images"
ORIENTATION_PATH = ROOT / "image_orientation_corrections.json"
IMAGE_EXTENSIONS = {
    ".avif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw images to cropped 2160x2160 JPEGs."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="reprocess and replace JPEGs that already exist",
    )
    return parser.parse_args()


def load_orientation_corrections(path: Path) -> dict[str, int]:
    """Read clockwise rotations keyed by paths relative to images/."""
    if not path.exists():
        return {}
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Image orientation corrections must be a JSON object.")
    corrections: dict[str, int] = {}
    for filepath, rotation in values.items():
        if (
            not isinstance(filepath, str)
            or not filepath
            or Path(filepath).is_absolute()
            or ".." in Path(filepath).parts
            or type(rotation) is not int
            or rotation not in (0, 90, 180, 270)
        ):
            raise ValueError(f"Invalid image orientation correction: {filepath!r}.")
        corrections[filepath] = rotation
    return corrections


def save_cropped_image(
    source: Path, destination: Path, clockwise_rotation: int = 0
) -> None:
    if clockwise_rotation not in (0, 90, 180, 270):
        raise ValueError("Clockwise rotation must be 0, 90, 180, or 270 degrees.")
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    cropped = crop_marked_square(bgr_image)
    if clockwise_rotation:
        cropped = cv2.rotate(
            cropped,
            {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }[clockwise_rotation],
        )
    rgb_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    output = Image.fromarray(rgb_crop)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.convert-tmp.jpg")
    try:
        output.save(temporary, "JPEG", quality=95)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(force: bool = False) -> None:
    pillow_heif.register_heif_opener()
    corrections = load_orientation_corrections(ORIENTATION_PATH)

    failures: list[tuple[Path, str, bool]] = []
    sources = sorted(
        path
        for path in SOURCE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for source in sources:
        destination = (OUTPUT_DIR / source.relative_to(SOURCE_DIR)).with_suffix(".jpg")

        if destination.exists() and not force:
            print(f"Skipping existing file: {destination}")
            continue

        try:
            rotation = corrections.get(destination.relative_to(OUTPUT_DIR).as_posix(), 0)
            save_cropped_image(source, destination, rotation)
        except (
            MarkerDetectionError,
            OSError,
            UnidentifiedImageError,
            ValueError,
        ) as error:
            removed = False
            if force and destination.exists():
                destination.unlink()
                removed = True
            failures.append((source.relative_to(SOURCE_DIR), str(error), removed))
            continue

        print(f"{source} -> {destination}")

    if not failures:
        print("\nAll images were converted and cropped successfully.")
        return

    print(f"\nWarning: {len(failures)} image(s) could not be converted and cropped:")
    for source, error, removed in failures:
        removal = "; existing JPEG removed" if removed else ""
        print(f"- {source}: {error}{removal}")


if __name__ == "__main__":
    main(force=parse_args().force)
