"""Combined multi-tool overview and original-pixel crop panels."""

from __future__ import annotations

import colorsys
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from image_refinement import tool_crop
from masks import mask2xywh

if TYPE_CHECKING:
    from image_review import Candidate
    from matplotlib.backend_bases import KeyEvent, MouseEvent


@dataclass(frozen=True)
class ToolPreview:
    tool_id: str
    label: str
    mask: NDArray[np.bool_]
    original: NDArray[np.bool_] | None = None
    candidate_index: int | None = None
    edit_note: str = ""
    disputed: NDArray[np.bool_] | None = None
    locked: bool = False
    candidates: tuple[int, ...] = ()


def candidate_color(index: int) -> tuple[float, float, float]:
    """Keep each candidate's color stable without wrapping a short palette."""
    return colorsys.hsv_to_rgb((0.59 + index * 0.618033988749895) % 1, 0.8, 0.92)


def dispute_crops(disputed: NDArray[np.bool_]) -> list[tuple[int, int, int, int]]:
    """Cover every disputed pixel with small crops and surrounding image context."""
    height, width = disputed.shape
    crops = []
    for top in range(0, height, 256):
        for left in range(0, width, 256):
            ys, xs = np.nonzero(disputed[top : top + 256, left : left + 256])
            if not len(xs):
                continue
            bounds = []
            for low, high, limit in (
                (left + int(xs.min()), left + int(xs.max()) + 1, width),
                (top + int(ys.min()), top + int(ys.max()) + 1, height),
            ):
                size = min(limit, max(128, high - low + 64))
                start = max(0, min((low + high - size) // 2, limit - size))
                bounds.append((start, start + size))
            crop = bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]
            if crop not in crops:
                crops.append(crop)
    return crops


def show_overlap_review(
    image: Image.Image, tools: Sequence[ToolPreview], title: str
) -> None:
    """Inspect original-pixel disputes before and after ordering, with linked zoom."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    disputed = np.zeros((image.height, image.width), dtype=np.bool_)
    for tool in tools:
        if tool.disputed is not None:
            disputed |= tool.disputed
    crops = dispute_crops(disputed)
    if not crops:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 6), sharex=True, sharey=True)
    region = 0
    overlay_visible = True
    overlays = []

    def render() -> None:
        overlays.clear()
        for legend in list(figure.legends):
            legend.remove()
        left, top, right, bottom = crops[region]
        extent = (left - 0.5, right - 0.5, bottom - 0.5, top - 0.5)
        pixels = np.asarray(image.crop((left, top, right, bottom)).convert("RGB"))
        for axis, label in zip(
            axes, ["Original image", "Before ordering", "After ordering"], strict=True
        ):
            axis.clear()
            axis.imshow(pixels, extent=extent, interpolation="nearest")
            axis.set_title(label)
            axis.set_xlabel("Image x (pixels)")
        axes[0].set_ylabel("Image y (pixels)")
        legend_handles = []
        for index, tool in enumerate(tools):
            source = (
                tool.original
                if tool.original is not None
                and tool.disputed is not None
                and tool.disputed.any()
                else tool.mask
            )
            before = source[top:bottom, left:right]
            after = tool.mask[top:bottom, left:right]
            if not (before.any() or after.any()):
                continue
            color = candidate_color(
                tool.candidate_index if tool.candidate_index is not None else index
            )
            legend_handles.append(
                Patch(facecolor=color, label=f"Tool {tool.tool_id} — {tool.label}")
            )
            for axis, mask in zip(axes[1:], [before, after], strict=True):
                overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
                overlay[mask] = (*[round(channel * 255) for channel in color], 115)
                overlays.append(
                    axis.imshow(overlay, extent=extent, interpolation="nearest")
                )
                overlays[-1].set_visible(overlay_visible)
        crop_disputed = disputed[top:bottom, left:right]
        # Pad contours so even a dispute touching the crop edge has an outline.
        for axis in axes[1:]:
            axis.contour(
                np.arange(left - 1, right + 1),
                np.arange(top - 1, bottom + 1),
                np.pad(crop_disputed, 1),
                levels=[0.5],
                colors=["magenta"],
                linewidths=0.8,
            )
        axes[0].set_xlim(left - 0.5, right - 0.5)
        axes[0].set_ylim(bottom - 0.5, top - 0.5)
        figure.suptitle(
            f"{title} — disputed region {region + 1}/{len(crops)}\n"
            "Toolbar zoom/pan links all panels; [ / ] = previous/next; "
            "m = toggle masks; Esc = close\n"
            "Magenta outlines disputed pixels. Return to the terminal to accept or edit.",
            fontsize=10,
        )
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(4, len(legend_handles)),
            fontsize=9,
        )
        figure.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.77, wspace=0.12)
        manager = figure.canvas.manager
        if manager is not None and manager.toolbar is not None:
            manager.toolbar.update()
            manager.toolbar.push_current()
        figure.canvas.draw_idle()

    def on_key(event: KeyEvent) -> None:
        nonlocal region, overlay_visible
        if event.key in {"[", "]"}:
            region = (region + (1 if event.key == "]" else -1)) % len(crops)
            render()
        elif event.key == "m":
            overlay_visible = not overlay_visible
            for overlay in overlays:
                overlay.set_visible(overlay_visible)
            figure.canvas.draw_idle()
        elif event.key == "escape":
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show(block=False)
    plt.pause(0.001)


def show_tool_review(
    image: Image.Image,
    candidates: Sequence[Candidate],
    title: str,
    tools: Sequence[ToolPreview],
) -> None:
    import matplotlib.pyplot as plt

    columns = min(2, max(1, len(tools)))
    rows = max(1, math.ceil(len(tools) / columns))
    figure = plt.figure(figsize=(6 * (columns + 1), 5 * rows))
    grid = figure.add_gridspec(rows, columns + 1, width_ratios=[1.3, *([1] * columns)])
    overview = figure.add_subplot(grid[:, 0])
    overview_pixels = np.array(image.convert("RGB"))
    figure.suptitle(title, fontsize=11, y=0.98)
    overview.set_title(
        "Overview — numbered mask candidates\nColor = mask pixels; dotted rectangles = boxes",
        fontsize=10,
    )
    overview.axis("off")
    disputed = np.zeros(overview_pixels.shape[:2], dtype=np.bool_)
    for index, candidate in enumerate(candidates):
        box = mask2xywh(candidate.mask)
        if box is None:
            continue
        x, y, width, height = box
        unassigned = "[unassigned]" in candidate.prompt
        color = candidate_color(index)
        overview.add_patch(
            plt.Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=color if unassigned else "gray",
                linestyle=":",
            )
        )
        overview.text(
            x,
            y,
            f"{index}: {candidate.prompt.split(' [', 1)[0]}",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.7},
        )
        # Unassigned candidates remain visible until assigned, excluded, or ignored.
        if unassigned:
            overview_pixels[candidate.mask] = (
                0.5 * overview_pixels[candidate.mask] + 0.5 * np.array(color) * 255
            ).astype(np.uint8)
    for index, tool in enumerate(tools):
        color = candidate_color(
            tool.candidate_index
            if tool.candidate_index is not None
            else len(candidates) + index
        )
        axis = figure.add_subplot(grid[index // columns, 1 + index % columns])
        axis.set_title(f"Tool {tool.tool_id} — {tool.label}", color=color, fontsize=10)
        axis.axis("off")
        if not tool.mask.any():
            axis.text(
                0.5,
                0.5,
                "No visible mask assigned",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            continue
        overview_pixels[tool.mask] = (
            0.55 * overview_pixels[tool.mask] + 0.45 * np.array(color) * 255
        ).astype(np.uint8)
        overview_box = mask2xywh(tool.mask)
        assert overview_box is not None
        overview.text(
            overview_box[0],
            overview_box[1] + overview_box[3],
            f"tool {tool.tool_id}",
            color=color,
            bbox={"facecolor": "black", "alpha": 0.7},
        )
        extent = tool.mask if tool.original is None else tool.mask | tool.original
        if tool.disputed is not None:
            extent = extent | tool.disputed
            disputed |= tool.disputed
        left, top, right, bottom = tool_crop(extent)
        axis.imshow(image.crop((left, top, right, bottom)))
        crop_mask = tool.mask[top:bottom, left:right]
        overlay = np.zeros((*crop_mask.shape, 4), dtype=np.uint8)
        overlay[crop_mask] = (*[round(channel * 255) for channel in color], 115)
        axis.imshow(overlay, interpolation="nearest")
        if tool.disputed is not None and tool.disputed.any():
            axis.contour(
                tool.disputed[top:bottom, left:right],
                levels=[0.5],
                colors=["magenta"],
                linewidths=1,
            )
            axis.text(
                0.02,
                0.98,
                "Magenta outline: disputed region",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8},
            )
        if tool.original is not None and not np.array_equal(tool.original, tool.mask):
            axis.text(
                0.02,
                0.02,
                tool.edit_note or f"Refined mask — restore {tool.tool_id} to undo",
                transform=axis.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8},
            )
    overview.imshow(overview_pixels)
    if disputed.any():
        overview.contour(disputed, levels=[0.5], colors=["magenta"], linewidths=1)
    figure.subplots_adjust(
        left=0.03, right=0.98, bottom=0.06, top=0.88, wspace=0.15, hspace=0.3
    )
    plt.show(block=False)
    plt.pause(0.001)


def paint_polygon(
    mask: NDArray[np.bool_], vertices: Sequence[tuple[float, float]], *, add: bool
) -> None:
    """Paint pixel centers inside a polygon, with bounded temporary coordinate arrays."""
    from matplotlib.path import Path

    if len(vertices) < 3:
        raise ValueError("A polygon needs at least three vertices.")
    points: NDArray[np.float64] = np.asarray(vertices, dtype=np.float64)
    if points.shape != (len(vertices), 2) or not np.isfinite(points).all():
        raise ValueError("Polygon coordinates must be finite x/y pairs.")
    height, width = mask.shape
    left = max(0, int(np.floor(points[:, 0].min())))
    right = min(width, int(np.ceil(points[:, 0].max())) + 1)
    top = max(0, int(np.floor(points[:, 1].min())))
    bottom = min(height, int(np.ceil(points[:, 1].max())) + 1)
    if left >= right or top >= bottom:
        return
    path = Path(points)
    for start in range(top, bottom, 256):
        stop = min(bottom, start + 256)
        x: NDArray[np.int64]
        y: NDArray[np.int64]
        x, y = np.meshgrid(np.arange(left, right), np.arange(start, stop))
        inside = path.contains_points(
            np.column_stack((x.ravel(), y.ravel())), radius=1e-9
        ).reshape(x.shape)
        region = mask[start:stop, left:right]
        region[inside] = add


def draw_tool_mask(
    image: Image.Image, initial: NDArray[np.bool_], title: str = "Tool mask"
) -> NDArray[np.bool_] | None:
    """Trace visible tool pixels with additive/subtractive polygons; closing cancels."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import PolygonSelector

    if initial.shape != (image.height, image.width):
        raise ValueError("Initial drawing mask must have full-image dimensions.")
    figure, axis = plt.subplots(figsize=(10, 8))
    figure.suptitle(title, fontsize=12)
    figure.subplots_adjust(top=0.85)
    axis.imshow(image)
    mask = initial.copy()
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask] = (0, 180, 255, 128)
    layer = axis.imshow(overlay, interpolation="nearest")
    axis.axis("off")
    mode = True
    history: list[tuple[list[tuple[float, float]], bool]] = []
    result: NDArray[np.bool_] | None = None

    def refresh() -> None:
        overlay[:] = 0
        overlay[mask] = (0, 180, 255, 128)
        layer.set_data(overlay)
        axis.set_title(
            f"{'ADD' if mode else 'REMOVE'} visible pixels: click polygon vertices; click first vertex to close\n"
            "+ add | - remove (holes/background) | z undo polygon | Enter save | Esc cancel",
            fontsize=10,
        )
        figure.canvas.draw_idle()

    def select(vertices: list[tuple[float, float]], *_events: object) -> None:
        if len(vertices) < 3:
            return
        history.append((list(vertices), mode))
        paint_polygon(mask, vertices, add=mode)
        selector.clear()
        refresh()

    selector = PolygonSelector(axis, select, props={"color": "cyan", "linewidth": 1.2})

    def keypress(event: KeyEvent) -> None:
        nonlocal mode, result
        if event.key in {"+", "=", "-"}:
            mode = event.key != "-"
            selector.clear()
        elif event.key == "z":
            selector.clear()
            if history:
                history.pop()
                mask[:] = initial
                for vertices, add in history:
                    paint_polygon(mask, vertices, add=add)
        elif event.key == "enter":
            if selector.verts:
                print(
                    "Close the current polygon by clicking its first vertex before saving."
                )
                return
            result = mask.copy()
            plt.close(figure)
            return
        elif event.key == "escape":
            plt.close(figure)
            return
        refresh()

    connection = figure.canvas.mpl_connect("key_press_event", keypress)
    refresh()
    try:
        plt.show(block=True)
    finally:
        selector.disconnect_events()
        figure.canvas.mpl_disconnect(connection)
        plt.close(figure)
    return result


def collect_clicks(
    image: Image.Image,
    mask: NDArray[np.bool_],
    initial_points: list[tuple[float, float, int]],
    title: str = "Tool point prompts",
) -> list[tuple[float, float, int]] | None:
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseButton

    plt.close("all")
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.imshow(image)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = (0.0, 0.8, 0.3, 0.4)
    axis.imshow(overlay)
    figure.suptitle(title, fontsize=12)
    figure.subplots_adjust(top=0.85)
    axis.set_title(
        "Left click: target tool; right click: other tools/background.\n"
        "Backspace: undo. Enter: run SAM3. Esc: cancel. Turn off toolbar zoom/pan to add points.",
        fontsize=10,
    )
    points = list(initial_points)
    markers = [
        axis.scatter([x], [y], c="lime" if label else "red", marker="+", s=100)
        for x, y, label in points
    ]
    accepted = False

    def click(event: MouseEvent) -> None:
        if axis.get_navigate_mode() is not None:
            return
        if event.inaxes != axis or event.xdata is None or event.ydata is None:
            return
        if event.button not in {MouseButton.LEFT, MouseButton.RIGHT}:
            return
        label = int(event.button == MouseButton.LEFT)
        x = min(max(float(event.xdata), 0), image.width - 1)
        y = min(max(float(event.ydata), 0), image.height - 1)
        points.append((x, y, label))
        markers.append(
            axis.scatter([x], [y], c="lime" if label else "red", marker="+", s=100)
        )
        figure.canvas.draw_idle()

    def key(event: KeyEvent) -> None:
        nonlocal accepted
        if event.key == "backspace" and points:
            points.pop()
            markers.pop().remove()
            figure.canvas.draw_idle()
        elif event.key in {"enter", "escape"}:
            accepted = event.key == "enter"
            plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", click)
    figure.canvas.mpl_connect("key_press_event", key)
    plt.show(block=True)
    return points if accepted else None
