"""Mouse controls and frame previews for the video annotation prototype."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from image_review import Candidate
from image_refinement import tool_crop
from image_ui import collect_clicks  # Shared mouse editor for image and video prompts.
from video_data import Mask

if TYPE_CHECKING:
    from matplotlib.backend_bases import KeyEvent, MouseEvent


def overlay_preview(
    image: Image.Image, mask: Mask, color: tuple[int, int, int] = (0, 204, 77)
) -> Image.Image:
    """Blend only a thumbnail-sized mask, avoiding full-frame RGBA buffers."""
    with Image.fromarray(mask) as source:
        foreground = np.asarray(source.resize(image.size, Image.Resampling.NEAREST))
    pixels = np.array(image)
    pixels[foreground] = (
        pixels[foreground].astype(np.uint16) * 3
        + np.asarray(color, dtype=np.uint16) * 2
    ) // 5
    return Image.fromarray(pixels)


def render_preview(
    image: Image.Image,
    mask: Mask,
    bounds: tuple[int, int, int, int],
    size: int = 768,
) -> Image.Image:
    view = image.crop(bounds)
    view.thumbnail((size, size), Image.Resampling.LANCZOS)
    left, top, right, bottom = bounds
    return overlay_preview(view, mask[top:bottom, left:right])


def save_contact_sheet(
    items: list[tuple[Image.Image, str]], destination: Path, size: int = 768
) -> None:
    """Save up to twelve bounded thumbnails without creating GUI figures."""
    columns = min(4, len(items))
    rows = math.ceil(len(items) / columns)
    with Image.new("RGB", (columns * size, rows * (size + 40)), "#202020") as sheet:
        draw = ImageDraw.Draw(sheet)
        for index, (image, title) in enumerate(items):
            x, y = (index % columns) * size, (index // columns) * (size + 40)
            draw.text((x + 8, y + 8), title, fill="white")
            sheet.paste(image, (x + (size - image.width) // 2, y + 40))
        sheet.save(destination, format="PNG")


def show_frame(
    image: Image.Image,
    mask: Mask,
    title: str,
    candidates: list[Candidate],
    zoom: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    plt.close("all")
    gc.collect()
    figure, axis = plt.subplots(figsize=(9, 8))
    visible = mask.copy()
    for candidate in candidates:
        visible |= candidate.mask
    bounds = (
        tool_crop(visible, padding=0.4)
        if zoom and visible.any()
        else (0, 0, image.width, image.height)
    )
    left, top, right, bottom = bounds
    view = render_preview(image, mask, bounds, size=1024)
    for index, candidate in enumerate(candidates):
        color = plt.get_cmap("tab10")(index % 10)
        view = overlay_preview(
            view,
            candidate.mask[top:bottom, left:right],
            (round(color[0] * 255), round(color[1] * 255), round(color[2] * 255)),
        )
        rows, columns = np.nonzero(candidate.mask)
        if len(rows):
            axis.text(
                int(columns.min()),
                int(rows.min()),
                f"{index}: {candidate.prompt} {candidate.score:.2f}",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.7},
            )
    axis.imshow(
        view,
        extent=(left - 0.5, right - 0.5, bottom - 0.5, top - 0.5),
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.axis("off")
    plt.show(block=False)
    plt.pause(0.001)


def select_region(image: Image.Image) -> tuple[int, int, int, int] | None:
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseButton
    from matplotlib.widgets import RectangleSelector

    plt.close("all")
    gc.collect()
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(image)
    axis.set_title("Drag a rectangle. Enter: select. Esc: cancel.")
    accepted = False

    def selected(press: MouseEvent, release: MouseEvent) -> None:
        pass

    selector = RectangleSelector(
        axis,
        selected,
        button=[MouseButton.LEFT],
        interactive=True,
        minspanx=1,
        minspany=1,
    )

    def key(event: KeyEvent) -> None:
        nonlocal accepted
        if event.key in {"enter", "escape"}:
            accepted = event.key == "enter"
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", key)
    plt.show(block=True)
    if not accepted:
        return None
    left, right, top, bottom = selector.extents
    box = (
        max(0, math.floor(left)),
        max(0, math.floor(top)),
        min(image.width, math.ceil(right)),
        min(image.height, math.ceil(bottom)),
    )
    return box if box[0] < box[2] and box[1] < box[3] else None
