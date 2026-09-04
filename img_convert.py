from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
import pillow_heif


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "raw-images"
OUTPUT_DIR = ROOT / "images"


def main() -> None:
    pillow_heif.register_heif_opener()

    for source in sorted(path for path in SOURCE_DIR.rglob("*") if path.is_file()):
        destination = (OUTPUT_DIR / source.relative_to(SOURCE_DIR)).with_suffix(".jpg")

        if destination.exists():
            print(f"Skipping existing file: {destination}")
            continue

        try:
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.save(destination, "JPEG", quality=95)
        except UnidentifiedImageError:
            print(f"Skipping non-image file: {source}")
            continue

        print(f"{source} -> {destination}")


if __name__ == "__main__":
    main()
