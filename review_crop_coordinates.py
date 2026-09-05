import argparse
import random
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import pillow_heif

from square_crop import CORNERS, MarkerDetectionError, find_marker_corners


ROOT = Path(__file__).resolve().parent
RAW_IMAGES_DIR = ROOT / "raw-images"
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
        description="Draw detected crop coordinates on saved image previews."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=32,
        help="total number of previews; use 0 for every raw image (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed used for repeatable sampling (default: 0)",
    )
    parser.add_argument(
        "--include",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help="always include a specific raw image; may be repeated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory; defaults to coordinate-review/ in this repository",
    )
    return parser.parse_args()


def find_raw_images(raw_images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def select_images(
    images: list[Path],
    raw_images_dir: Path,
    count: int,
    seed: int,
    included: list[Path],
) -> list[Path]:
    if count < 0:
        raise ValueError("count must be at least 0")
    if count == 0:
        return images

    selected = []
    selected_set = set()
    for path in included:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path not in selected_set:
            selected.append(path)
            selected_set.add(path)

    groups: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        resolved = image.resolve()
        if resolved in selected_set:
            continue
        relative = image.relative_to(raw_images_dir)
        tool_id = relative.parts[0] if len(relative.parts) > 1 else ""
        groups[tool_id].append(resolved)

    generator = random.Random(seed)
    queues = []
    for tool_id in sorted(groups):
        generator.shuffle(groups[tool_id])
        queues.append(deque(groups[tool_id]))

    while len(selected) < count and any(queues):
        for queue in queues:
            if queue and len(selected) < count:
                path = queue.popleft()
                selected.append(path)
                selected_set.add(path)

    return selected


def load_bgr(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def add_title(image: np.ndarray, title: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def coordinate_label_position(
    point: np.ndarray,
    corner: str,
    text: str,
    image_shape: tuple[int, ...],
) -> tuple[int, int]:
    text_width, text_height = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2,
    )[0]
    x, y = point
    x += 18 if corner in {"tl", "bl"} else -text_width - 18
    y += text_height + 18 if corner in {"tl", "tr"} else -18
    x = int(np.clip(x, 5, image_shape[1] - text_width - 5))
    y = int(np.clip(y, text_height + 5, image_shape[0] - 5))
    return x, y


def make_preview(
    image: np.ndarray,
    source: Path,
    raw_images_dir: Path,
    max_dimension: int = 1600,
) -> tuple[np.ndarray, str]:
    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    preview = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    title = source.relative_to(raw_images_dir).as_posix()

    try:
        corners = find_marker_corners(image)
    except MarkerDetectionError as error:
        add_title(preview, f"FAILED: {title}")
        cv2.putText(
            preview,
            str(error),
            (20, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return preview, f"FAILED: {error}"

    preview_corners = np.rint(corners * scale).astype(np.int32)
    cv2.polylines(
        preview,
        [preview_corners[[0, 1, 3, 2]]],
        True,
        (0, 255, 0),
        4,
        cv2.LINE_AA,
    )
    for corner, point, original_point in zip(CORNERS, preview_corners, corners):
        cv2.drawMarker(
            preview,
            tuple(point),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            32,
            4,
            cv2.LINE_AA,
        )
        text = f"{corner.upper()} ({original_point[0]:.0f}, {original_point[1]:.0f})"
        position = coordinate_label_position(point, corner, text, preview.shape)
        cv2.putText(
            preview,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    add_title(preview, title)
    return preview, "OK"


def make_contact_sheet(previews: list[np.ndarray], width: int = 320) -> np.ndarray:
    tiles = []
    for preview in previews:
        height = round(preview.shape[0] * width / preview.shape[1])
        tiles.append(cv2.resize(preview, (width, height), interpolation=cv2.INTER_AREA))

    tile_height = max(tile.shape[0] for tile in tiles)
    columns = min(4, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * tile_height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        top = index // columns * tile_height
        left = index % columns * width
        sheet[top : top + tile.shape[0], left : left + width] = tile
    return sheet


def main() -> None:
    args = parse_args()
    pillow_heif.register_heif_opener()
    raw_images_dir = RAW_IMAGES_DIR.resolve()
    images = find_raw_images(raw_images_dir)
    selected = select_images(
        images,
        raw_images_dir,
        args.count,
        args.seed,
        args.include,
    )
    if not selected:
        raise SystemExit("No raw images found.")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "coordinate-review"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    previews = []
    passed = 0
    for index, source in enumerate(selected, start=1):
        try:
            image = load_bgr(source)
            preview, status = make_preview(image, source, raw_images_dir)
        except (OSError, ValueError) as error:
            print(f"Warning: could not review {source}: {error}")
            continue

        relative = source.relative_to(raw_images_dir)
        filename = "__".join(relative.with_suffix("").parts) + ".jpg"
        destination = output_dir / filename
        if not cv2.imwrite(str(destination), preview, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            print(f"Warning: could not write {destination}")
            continue
        previews.append(preview)
        passed += status == "OK"
        print(f"[{index}/{len(selected)}] {relative}: {status}")

    if previews:
        contact_sheet = make_contact_sheet(previews)
        cv2.imwrite(
            str(output_dir / "contact_sheet.jpg"),
            contact_sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

    print(f"\nReview folder: {output_dir}")
    print(f"Detections: {passed} passed, {len(previews) - passed} failed")


if __name__ == "__main__":
    main()
