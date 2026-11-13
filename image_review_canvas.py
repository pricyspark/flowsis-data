"""Persistent multi-tool canvas with local edits and optional terminal commands."""

from __future__ import annotations

import queue
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from image_local_edit import proposal_groups
from image_refinement import tool_crop
from image_ui import ToolPreview, candidate_color

if TYPE_CHECKING:
    from image_review import Candidate
    from matplotlib.backend_bases import KeyEvent, MouseEvent
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.widgets import PolygonSelector, RectangleSelector


@dataclass(frozen=True)
class SampleReference:
    image: Image.Image
    label: str


class CommandInput:
    """One pending terminal read shared across canvases, while GUI events run."""

    def __init__(self) -> None:
        self.commands: queue.Queue[str] = queue.Queue()
        self.reading = False
        self.available = True

    def start(self, prompt: str) -> None:
        if self.reading or not self.available:
            return
        self.reading = True

        def read() -> None:
            try:
                self.commands.put(input(prompt))
            except EOFError:
                self.available = False
                self.commands.put("q")
            except KeyboardInterrupt:
                self.commands.put("q")
            finally:
                self.reading = False

        threading.Thread(target=read, daemon=True).start()

    def read(self, prompt: str) -> str:
        """Finish a canvas terminal read before another reviewer reads stdin."""
        if self.reading or not self.commands.empty():
            return self.commands.get()
        if not self.available:
            return "q"
        return input(prompt)


class MultiToolCanvas:
    MODES = (
        "Select proposal", "Correct mask", "Add piece", "Erase polygon",
        "Add polygon", "Box", "Crossing owner", "Brush add", "Brush erase", "Split piece",
    )

    def __init__(
        self,
        command_input: CommandInput | None = None,
        references: Sequence[SampleReference] = (),
    ) -> None:
        self.terminal = command_input or CommandInput()
        self.references = list(references)
        self.commands: queue.Queue[str] = queue.Queue()
        self.figure: Figure | None = None
        self.selected_tool = ""
        self.mode = self.MODES[0]
        self.points: list[tuple[float, float, int]] = []
        self.vertices: list[tuple[float, float]] = []
        self.stroke: list[tuple[float, float]] = []
        self.box: tuple[float, float, float, float] | None = None
        self.tools: list[ToolPreview] = []
        self.candidates: list[Candidate] = []
        self.groups: list[list[int]] = []
        self.image: Image.Image | None = None
        self.title = ""
        self._tool_labels: list[str] = []
        self._selectors: list[PolygonSelector | RectangleSelector] = []
        self._discard_preview = False
        self._notice = ""
        self._show_proposals = True
        self._last_masks: dict[str, NDArray[np.bool_]] = {}
        self._drawing = False
        self._pending_artists: list[Line2D] = []

    def _build(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

        self.figure = plt.figure(figsize=(16, 10))
        self.overview = self.figure.add_axes((0.23, 0.31, 0.47, 0.60))
        self.detail = self.figure.add_axes((0.73, 0.42, 0.26, 0.46))
        self.tool_axis = self.figure.add_axes((0.01, 0.67, 0.20, 0.23))
        self.mode_widget = RadioButtons(self.figure.add_axes((0.01, 0.28, 0.20, 0.36)), self.MODES)
        self.mode_widget.on_clicked(self._set_mode)
        self.radius = Slider(
            self.figure.add_axes((0.79, 0.36, 0.17, 0.02)),
            "Brush px", 1, 60, valinit=6,
        )
        self.destination = TextBox(
            self.figure.add_axes((0.82, 0.30, 0.14, 0.035)), "Split to Tool "
        )
        self.mask_indices = TextBox(self.figure.add_axes((0.08, 0.22, 0.12, 0.035)), "Masks ")
        self.proposal_toggle = CheckButtons(
            self.figure.add_axes((0.01, 0.14, 0.20, 0.06)),
            ["Show proposals"], [True],
        )
        self.proposal_toggle.on_clicked(self._toggle_proposals)
        self.buttons = []
        actions = [
            ("Apply", self.apply), ("Cancel", self.cancel), ("Undo", lambda: self.submit("undo")),
            ("Redo", lambda: self.submit("redo")), ("Reset image", lambda: self.submit("reset")),
            ("Lock/unlock", self.toggle_lock), ("Discard unused", self.discard),
            ("Accept", lambda: self.submit("a")), ("Skip", lambda: self.submit("s")),
            ("Quit", lambda: self.submit("q")),
        ]
        for index, (label, action) in enumerate(actions):
            button = Button(
                self.figure.add_axes((0.01 + index * 0.099, 0.035, 0.092, 0.045)),
                label,
            )
            button.on_clicked(lambda _event, callback=action: callback())
            self.buttons.append(button)
        self.status = self.figure.text(0.23, 0.105, "", fontsize=9)
        self.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.reference_axes = []
        for index, reference in enumerate(self.references[:5]):
            axis = self.figure.add_axes((0.24 + index * 0.148, 0.15, 0.135, 0.12))
            axis.imshow(reference.image)
            axis.set_title(reference.label, fontsize=7)
            axis.axis("off")
            self.reference_axes.append(axis)

    def show(
        self, image: Image.Image, candidates: Sequence[Candidate], title: str,
        tools: Sequence[ToolPreview],
    ) -> None:
        import matplotlib.pyplot as plt

        if self.figure is None or not plt.fignum_exists(self.figure.number):
            self._tool_labels = []
            self._build()
        changed = any(
            tool.tool_id not in self._last_masks
            or not np.array_equal(tool.mask, self._last_masks[tool.tool_id])
            for tool in tools
        )
        if changed:
            self._clear_pending()
        self.image = image
        self.candidates = list(candidates)
        self.title = title
        self.tools = list(tools)
        self._last_masks = {tool.tool_id: tool.mask.copy() for tool in tools}
        self.groups = proposal_groups(candidates)
        labels = [f"{tool.tool_id}: {tool.label}" for tool in tools]
        if labels != self._tool_labels:
            from matplotlib.widgets import RadioButtons

            self.tool_axis.clear()
            active = next(
                (index for index, tool in enumerate(tools)
                 if tool.tool_id == self.selected_tool),
                0,
            )
            self.tool_widget = RadioButtons(self.tool_axis, labels, active=active)
            self.tool_widget.on_clicked(self._select_tool)
            self._tool_labels = labels
        if self.selected_tool not in {tool.tool_id for tool in tools}:
            self.selected_tool = tools[0].tool_id
        self.render()
        plt.show(block=False)
        plt.pause(0.001)

    def render(self) -> None:
        if self.image is None or self.figure is None:
            return
        from matplotlib.widgets import PolygonSelector, RectangleSelector

        limits = (
            (self.overview.get_xlim(), self.overview.get_ylim())
            if self.overview.images else None
        )
        for selector in self._selectors:
            selector.disconnect_events()
        self._selectors.clear()
        self.overview.clear()
        self._pending_artists = []
        self.overview.imshow(self.image)
        self.overview.set_title(
            f"{self.title}\nSelected tool {self.selected_tool} | {self.mode}",
            fontsize=10,
        )
        self.overview.axis("off")
        if self._show_proposals:
            for group in self.groups:
                index = group[0]
                candidate = self.candidates[index]
                if not any("[unassigned]" in self.candidates[value].prompt for value in group):
                    continue
                ys, xs = np.nonzero(candidate.mask)
                self.overview.contour(
                    candidate.mask, levels=[0.5],
                    colors=[candidate_color(index)], linewidths=0.7,
                )
                self.overview.text(
                    xs.min(), ys.min(), "/".join(map(str, group)),
                    color="white", fontsize=8,
                    bbox={"facecolor": "black", "alpha": 0.6},
                )
        for index, tool in enumerate(self.tools):
            overlay = np.zeros((*tool.mask.shape, 4), dtype=np.uint8)
            overlay[tool.mask] = (
                *[round(channel * 255) for channel in candidate_color(index)], 110
            )
            self.overview.imshow(overlay)
            if tool.disputed is not None and tool.disputed.any():
                self.overview.contour(
                    tool.disputed, levels=[0.5], colors=["magenta"], linewidths=0.8
                )
        if limits is not None:
            self.overview.set_xlim(limits[0])
            self.overview.set_ylim(limits[1])
        else:
            extents = [tool.mask for tool in self.tools]
            extents.extend(part.mask for part in self.candidates if part.prompt != "[removed]")
            extent_mask = np.logical_or.reduce(extents) if extents else None
            if extent_mask is not None and extent_mask.any():
                left, top, right, bottom = tool_crop(extent_mask)
                self.overview.set_xlim(left - 0.5, right - 0.5)
                self.overview.set_ylim(bottom - 0.5, top - 0.5)
        self.detail.clear()
        tool = next(tool for tool in self.tools if tool.tool_id == self.selected_tool)
        region = tool.mask if tool.original is None else tool.mask | tool.original
        if tool.disputed is not None:
            region = region | tool.disputed
        if region.any():
            left, top, right, bottom = tool_crop(region)
            extent = (left - 0.5, right - 0.5, bottom - 0.5, top - 0.5)
            self.detail.imshow(self.image.crop((left, top, right, bottom)), extent=extent)
            overlay = np.zeros((bottom - top, right - left, 4), dtype=np.uint8)
            selected_index = next(
                index for index, item in enumerate(self.tools)
                if item.tool_id == self.selected_tool
            )
            overlay[tool.mask[top:bottom, left:right]] = (
                *[round(channel * 255) for channel in candidate_color(selected_index)], 110
            )
            self.detail.imshow(overlay, extent=extent)
            if tool.original is not None:
                difference = (
                    tool.original[top:bottom, left:right]
                    ^ tool.mask[top:bottom, left:right]
                )
                if difference.any():
                    self.detail.contour(
                        np.arange(left, right), np.arange(top, bottom), difference,
                        levels=[0.5], colors=["yellow"], linewidths=0.8,
                    )
        self.detail.set_title(
            f"Tool {tool.tool_id} {'[locked]' if tool.locked else ''}\n"
            f"{tool.label}\nYellow: edited pixels",
            fontsize=9,
        )
        self.detail.axis("off")
        if not region.any():
            self.detail.text(
                0.5, 0.5, "Select a proposal or add points / a box",
                ha="center", transform=self.detail.transAxes,
            )
        for axis in (self.overview, self.detail):
            polygon = PolygonSelector(axis, self._polygon, props={"color": "cyan"})
            rectangle = RectangleSelector(axis, self._rectangle, button=[1], useblit=False)
            self._selectors.extend([polygon, rectangle])
            polygon.set_active(self.mode in {"Erase polygon", "Add polygon", "Split piece"})
            rectangle.set_active(self.mode == "Box")
        self.overview.set_autoscale_on(False)
        self.detail.set_autoscale_on(False)
        self._draw_pending()
        self.figure.canvas.draw_idle()

    def _draw_pending(self) -> None:
        assert self.figure is not None
        for artist in self._pending_artists:
            artist.remove()
        self._pending_artists.clear()
        for axis in (self.overview, self.detail):
            for x, y, label in self.points:
                self._pending_artists.extend(axis.plot(
                    x, y, "+" if label else "x", color="lime" if label else "red", markersize=10
                ))
            if self.stroke:
                xs, ys = zip(*self.stroke)
                width = (
                    2 * self.radius.val * axis.bbox.width
                    / abs(np.diff(axis.get_xlim())[0]) * 72 / self.figure.dpi
                )
                self._pending_artists.extend(
                    axis.plot(xs, ys, color="cyan", linewidth=width, alpha=0.5)
                )
        self.status.set_text(self._notice or (
            "Select a tool at left. Shift-click adds a proposal. Stage corrections, then Apply.\n"
            "Right-click = background in point modes. Ctrl+Z / Ctrl+Y = undo/redo. Terminal commands remain available."
        ))

    def _select_tool(self, label: str) -> None:
        self.selected_tool = label.partition(":")[0]
        self._clear_pending()
        self.render()

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._clear_pending()
        self.render()

    def _toggle_proposals(self, _label: str) -> None:
        self._show_proposals = not self._show_proposals
        self.render()

    def _polygon(self, vertices: list[tuple[float, float]]) -> None:
        self.vertices = [(float(x), float(y)) for x, y in vertices]

    def _rectangle(self, start: MouseEvent, end: MouseEvent) -> None:
        if (
            start.xdata is not None and start.ydata is not None
            and end.xdata is not None and end.ydata is not None
        ):
            self.box = (
                min(start.xdata, end.xdata), min(start.ydata, end.ydata),
                max(start.xdata, end.xdata), max(start.ydata, end.ydata),
            )

    def on_press(self, event: MouseEvent) -> None:
        if self.figure is None or event.inaxes not in (self.overview, self.detail):
            return
        toolbar = self.figure.canvas.manager.toolbar
        if toolbar is not None and toolbar.mode:
            return
        if event.xdata is None or event.ydata is None or self.image is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if not 0 <= x < self.image.width or not 0 <= y < self.image.height:
            return
        if self.mode in {"Correct mask", "Add piece"} and event.button in {1, 3}:
            self.points.append((x, y, 1 if event.button == 1 else 0))
            self._draw_pending()
            self.figure.canvas.draw_idle()
        elif self.mode.startswith("Brush") and event.button == 1:
            self.stroke = [(x, y)]
            self._drawing = True
        elif self.mode == "Crossing owner" and event.button == 1:
            self.submit(f"own {self.selected_tool} {x} {y}")
        elif self.mode == "Select proposal" and event.button == 1:
            available = [
                index for group in self.groups for index in group
                if self.candidates[index].mask[int(y), int(x)]
                and ("[unassigned]" in self.candidates[index].prompt or f"[tool {self.selected_tool}]" in self.candidates[index].prompt)
            ]
            if available:
                index = max(available, key=lambda index: self.candidates[index].score)
                values = {index}
                if event.key == "shift":
                    tool = next(tool for tool in self.tools if tool.tool_id == self.selected_tool)
                    values.update(tool.candidates)
                self.submit(f"g {self.selected_tool} " + ",".join(map(str, sorted(values))))

    def on_motion(self, event: MouseEvent) -> None:
        if (
            self.figure is None or not self._drawing
            or event.inaxes not in (self.overview, self.detail)
            or event.xdata is None or event.ydata is None
        ):
            return
        x, y = float(event.xdata), float(event.ydata)
        if self.image is None or not 0 <= x < self.image.width or not 0 <= y < self.image.height:
            return
        if self.stroke and (
            np.hypot(x - self.stroke[-1][0], y - self.stroke[-1][1])
            < max(1, self.radius.val / 3)
        ):
            return
        self.stroke.append((x, y))
        self._draw_pending()
        self.figure.canvas.draw_idle()

    def on_release(self, _event: MouseEvent) -> None:
        self._drawing = False

    def on_key(self, event: KeyEvent) -> None:
        if event.key in {"ctrl+z", "ctrl+y"}:
            self.submit("undo" if event.key == "ctrl+z" else "redo")
        elif event.key == "escape":
            self.cancel()

    def _clear_pending(self) -> None:
        self.points.clear()
        self.vertices.clear()
        self.stroke.clear()
        self.box = None
        self._drawing = False

    def cancel(self) -> None:
        self._notice = ""
        self._clear_pending()
        self.render()

    def report(self, message: str) -> None:
        self._notice = message
        if self.figure is not None:
            self.status.set_text(message)
            self.figure.canvas.draw_idle()

    def submit(self, command: str) -> None:
        self._notice = ""
        self.commands.put(command)

    def apply(self) -> None:
        if self.figure is None:
            return
        target = self.selected_tool
        if self.mode in {"Correct mask", "Add piece"} and self.points:
            command = "correct" if self.mode == "Correct mask" else "add"
            points = " ".join(f"{x},{y},{label}" for x, y, label in self.points)
            self.submit(f"{command} {target} {points}")
        elif self.mode == "Box" and self.box is not None:
            self.submit(f"box {target} " + " ".join(map(str, self.box)))
        elif self.mode in {"Erase polygon", "Add polygon", "Split piece"} and self.vertices:
            points = " ".join(f"{x},{y}" for x, y in self.vertices)
            if self.mode == "Split piece":
                self.submit(f"split {target} {self.destination.text.strip()} {points}")
            else:
                mode = "add" if self.mode == "Add polygon" else "erase"
                self.submit(f"paint {target} {mode} {points}")
        elif self.mode.startswith("Brush") and self.stroke:
            mode = "add" if self.mode == "Brush add" else "erase"
            points = " ".join(f"{x},{y}" for x, y in self.stroke)
            self.submit(f"brush {target} {mode} {self.radius.val} {points}")
        elif self.mode == "Select proposal" and self.mask_indices.text.strip():
            self.submit(f"g {target} {self.mask_indices.text.strip()}")
        else:
            self.status.set_text("Stage points, a polygon, a box, or a brush stroke before Apply.")
            self.figure.canvas.draw_idle()

    def toggle_lock(self) -> None:
        tool = next(tool for tool in self.tools if tool.tool_id == self.selected_tool)
        self.submit(f"{'unlock' if tool.locked else 'lock'} {self.selected_tool}")

    def discard(self) -> None:
        if self._discard_preview:
            self.submit("discard confirm")
            self._discard_preview = False
        else:
            self._show_proposals = True
            self.render()
            self.submit("discard")
            self._discard_preview = True
            self.status.set_text("Unused proposals shown. Click Discard unused again to exclude them; Undo restores them.")

    def read_command(self, prompt: str) -> str:
        import matplotlib.pyplot as plt

        if self.figure is None or self.figure.canvas.required_interactive_framework is None:
            return self.terminal.read(prompt)
        self.terminal.start(prompt)
        while plt.fignum_exists(self.figure.number):
            for commands in (self.commands, self.terminal.commands):
                try:
                    command = commands.get_nowait().strip()
                    self._notice = ""
                    return command
                except queue.Empty:
                    pass
            plt.pause(0.05)
        return "q"

    def close(self) -> None:
        if self.figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self.figure)
