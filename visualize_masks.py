"""Save previews of existing mask bundles without loading an annotation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image

from masks import load_mask_bundle

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}


def find_image(mask_path: Path, root: Path = ROOT / "data") -> Path:
    """Resolve the image/frame mirrored by a standard repository mask path."""
    try:
        relative = mask_path.resolve().relative_to((root / "masks").resolve())
    except ValueError as error:
        raise ValueError(
            "Use --image for masks outside the repository mask tree."
        ) from error
    if relative.parts[0] == "images":
        stem = root / relative.with_suffix("")
    elif relative.parts[0] == "videos":
        stem = root / "frames" / relative.with_suffix("")
    else:
        raise ValueError("Use --image for masks outside masks/images or masks/videos.")
    matches = sorted(
        path
        for path in stem.parent.iterdir()
        if path.is_file()
        and path.stem == stem.name
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one source image for {mask_path}, found {len(matches)}. "
            "Use --image to specify it."
        )
    return matches[0]


def save_preview(
    mask_path: Path,
    image_path: Path,
    output_path: Path,
    *,
    alpha: float = 0.45,
    max_size: int = 1000,
) -> Path:
    """Render original, overlay, and mask-only panels with bundle identities."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be greater than 0 and at most 1.")
    if max_size < 1:
        raise ValueError("max_size must be positive.")
    bundle = load_mask_bundle(mask_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    height, width = bundle.masks.shape[1:]
    if image.size != (width, height):
        raise ValueError(
            f"Image size {image.size} differs from mask size {(width, height)}."
        )
    scale = min(1.0, max_size / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(size, Image.Resampling.LANCZOS)
    original = np.asarray(image, dtype=np.float32) / 255
    overlay = original.copy()
    mask_view = np.zeros_like(original)

    figure = Figure(figsize=(15, 6), layout="constrained")
    FigureCanvasAgg(figure)
    colors = colormaps["tab20"]
    legend = []
    for index, (mask, identity) in enumerate(zip(bundle.masks, bundle.identities)):
        visible = np.asarray(
            Image.fromarray(mask).resize(size, Image.Resampling.NEAREST)
        )
        color = np.asarray(colors(index % 20)[:3])
        overlay[visible] = (1 - alpha) * overlay[visible] + alpha * color
        # Blend overlapping instances rather than hiding earlier masks entirely.
        occupied = visible & np.any(mask_view != 0, axis=2)
        mask_view[visible & ~occupied] = color
        mask_view[occupied] = (mask_view[occupied] + color) / 2
        legend.append(Patch(color=color, label=f"{index}: {identity}"))

    if bundle.ignore is not None:
        ignored = np.asarray(
            Image.fromarray(bundle.ignore).resize(size, Image.Resampling.NEAREST)
        )
        rows, cols = np.indices(ignored.shape)
        # Black/white stripes distinguish ignore pixels from instance colors.
        stripes = np.where(((rows + cols) // 4) % 2 == 0, 0.25, 1.0)
        overlay[ignored] = (
            (1 - alpha) * overlay[ignored] + alpha * stripes[ignored, None]
        )
        mask_view[ignored] = stripes[ignored, None]
        legend.append(
            Patch(facecolor="white", edgecolor="black", hatch="///", label="Ignore")
        )

    axes = figure.subplots(1, 3)
    for axis, pixels, title in zip(
        axes, (original, overlay, mask_view), ("Original", "Mask overlay", "Masks")
    ):
        axis.imshow(pixels, interpolation="nearest")
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle(
        f"{image_path.name} | {mask_path.name} | {len(bundle.identities)} instances"
    )
    if legend:
        axes[2].legend(handles=legend, loc="upper left", bbox_to_anchor=(1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="png", dpi=120)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=ROOT / "data/masks/images",
        help="one .npz bundle or a directory searched recursively (default: data/masks/images)",
    )
    parser.add_argument("--image", type=Path, help="source image for a single bundle")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "mask-review")
    parser.add_argument(
        "--count",
        type=int,
        default=32,
        help="preview limit; 0 means all (default: 32)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.45, help="overlay opacity (default: 0.45)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1000,
        help="maximum image dimension per panel (default: 1000)",
    )
    args = parser.parse_args()
    if args.count < 0 or args.max_size < 1 or not 0 < args.alpha <= 1:
        parser.error("count must be nonnegative, max-size positive, and alpha in (0, 1].")
    if args.image and not args.path.is_file():
        parser.error("--image requires a single mask bundle.")
    paths = [args.path] if args.path.is_file() else sorted(args.path.rglob("*.npz"))
    if args.count:
        paths = paths[:args.count]
    if not paths:
        raise SystemExit(f"No mask bundles found in {args.path}.")
    failed = 0
    for path in paths:
        relative = path.relative_to(args.path) if args.path.is_dir() else Path(path.name)
        output = args.output_dir / relative.with_suffix(".png")
        try:
            image = args.image or find_image(path)
            save_preview(path, image, output, alpha=args.alpha, max_size=args.max_size)
        except (OSError, ValueError, KeyError) as error:
            print(f"FAILED {path}: {error}")
            failed += 1
        else:
            print(f"Saved {output}")
    print(f"Saved {len(paths) - failed} previews; {failed} failed.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
