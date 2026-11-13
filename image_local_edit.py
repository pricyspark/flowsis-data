"""Small image-space edits and proposal grouping for instrument annotation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from image_review import Candidate
    from image_review_state import ReviewState


def coordinates(text: str, *, labels: bool = False) -> list[tuple[float, ...]]:
    width = 3 if labels else 2
    points = [tuple(float(value) for value in token.split(",")) for token in text.split()]
    if not points or any(len(point) != width for point in points):
        raise ValueError("Use x,y,label points." if labels else "Use x,y vertices.")
    if not all(np.isfinite(point).all() for point in points):
        raise ValueError("Coordinates must be finite.")
    if labels and any(point[2] not in {0, 1} for point in points):
        raise ValueError("Point labels must be 1 (tool) or 0 (background).")
    return points


def stroke_region(
    shape: tuple[int, int], points: Sequence[tuple[float, float]], radius: float
) -> NDArray[np.bool_]:
    if not np.isfinite(radius) or not 0 < radius <= max(shape):
        raise ValueError("Brush radius must be positive and no larger than the image.")
    if not points or any(
        not np.isfinite((x, y)).all() or not 0 <= x < shape[1] or not 0 <= y < shape[0]
        for x, y in points
    ):
        raise ValueError("Brush points must be inside the image.")
    pixels = np.zeros(shape, dtype=np.uint8)
    vertices = np.rint(points).astype(np.int32)
    size = max(1, round(radius))
    if len(vertices) > 1:
        cv2.polylines(pixels, [vertices], False, 1, thickness=2 * size + 1)
    for x, y in vertices:
        cv2.circle(pixels, (int(x), int(y)), size, 1, thickness=-1)
    return pixels.astype(np.bool_)


def crossing_region(overlap: NDArray[np.bool_], x: float, y: float) -> NDArray[np.bool_]:
    if (
        not np.isfinite((x, y)).all()
        or not 0 <= x < overlap.shape[1]
        or not 0 <= y < overlap.shape[0]
    ):
        raise ValueError("Choose a crossing inside the image.")
    _, components = cv2.connectedComponents(overlap.astype(np.uint8), connectivity=8)
    label = components[int(y), int(x)]
    if label == 0:
        raise ValueError("Click a disputed pixel to choose its owner.")
    return components == label


def proposal_groups(candidates: Sequence[Candidate], threshold: float = 0.95) -> list[list[int]]:
    """Group near-identical proposals for display; retain every source candidate."""
    groups: list[list[int]] = []
    for index, candidate in enumerate(candidates):
        if not candidate.mask.any() or candidate.prompt == "[removed]":
            continue
        for group in groups:
            reference = candidates[group[0]].mask
            intersection = np.count_nonzero(candidate.mask & reference)
            union = np.count_nonzero(candidate.mask | reference)
            if intersection / union >= threshold:
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


def find_tool(state: ReviewState, tool_id: str) -> dict[str, str]:
    row = next((row for row in state.instances if row["Tool ID"] == tool_id), None)
    if row is None:
        raise ValueError("Choose a declared Tool ID.")
    return row


def ensure_editable(state: ReviewState, row: dict[str, str]) -> None:
    if row["Instance ID"] in state.locked:
        raise ValueError(f"Tool {row['Tool ID']} is locked; unlock it before editing.")


def apply_local_edit(
    command: str,
    argument: str,
    state: ReviewState,
    resolve: Callable[[dict[str, str], bool], Candidate],
    spatial: (
        Callable[[list[tuple[float, float, int]], NDArray[np.bool_] | None], Candidate]
        | None
    ),
    box_spatial: Callable[[tuple[float, float, float, float]], Candidate] | None,
) -> None:
    from image_review import ClickPrompt
    from image_ui import paint_polygon
    from masks import mask_overlap

    blank = np.zeros(state.shape, dtype=np.bool_)

    def assign_edit(
        row: dict[str, str],
        mask: NDArray[np.bool_],
        prompt: str,
        score: float | None = None,
    ) -> None:
        ensure_editable(state, row)
        for other in state.instances:
            if other["Instance ID"] in state.locked:
                mask = mask & ~resolve(other, False).mask
        if not mask.any():
            raise ValueError("Edit would erase the tool; previous mask retained.")
        identity = row["Instance ID"]
        source = resolve(row, False)
        state.before_edits[identity] = source
        index = len(state.candidates)
        state.candidates.append(
            replace(
                source, mask=mask, prompt=prompt, identity=f"candidate-{index}",
                score=source.score if score is None else score,
            )
        )
        state.generated.add(index)
        state.excluded.update(state.assignments.get(identity, []))
        state.assignments[identity] = [index]
        if state.states.get(identity) not in {"clear", "partial", "severe"}:
            state.states[identity] = "clear"
        state.cleaned.pop(identity, None)
        state.refinements.pop(identity, None)
        state.active_prompts.pop(identity, None)

    target, _, options = argument.partition(" ")
    row = find_tool(state, target)
    if command != "own":
        ensure_editable(state, row)
    source = resolve(row, False)
    if command in {"correct", "add"}:
        if spatial is None:
            raise ValueError("Point prompting is unavailable.")
        clicks = [(x, y, int(label)) for x, y, label in coordinates(options, labels=True)]
        seed = source.mask if command == "correct" and source.mask.any() else None
        if any(not 0 <= x < state.shape[1] or not 0 <= y < state.shape[0] for x, y, _ in clicks):
            raise ValueError("Point coordinates must be inside the image.")
        prediction = spatial(clicks, seed)
        if prediction.mask.shape != blank.shape or prediction.mask.dtype != np.bool_:
            raise ValueError("SAM3 must return a boolean full-image mask.")
        mask = prediction.mask if command == "correct" else source.mask | prediction.mask
        score = prediction.score if command == "correct" else max(source.score, prediction.score)
        assign_edit(row, mask, f"{command} visible tool {target}", score)
        identity = row["Instance ID"]
        index = state.assignments[identity][0]
        prompt = ClickPrompt(identity, clicks, seed, index, command == "add")
        state.active_prompts[identity] = prompt
        state.click_prompts[index] = prompt
    elif command == "box":
        bounds = tuple(float(value) for value in options.split())
        if len(bounds) != 4 or box_spatial is None:
            raise ValueError("Use box TOOL_ID LEFT TOP RIGHT BOTTOM with box prompting enabled.")
        left, top, right, bottom = bounds
        if not all(np.isfinite(bounds)) or not (
            0 <= left < right <= state.shape[1]
            and 0 <= top < bottom <= state.shape[0]
        ):
            raise ValueError("Box corners must be ordered and inside the image.")
        prediction = box_spatial((left, top, right, bottom))
        if prediction.mask.shape != blank.shape or prediction.mask.dtype != np.bool_:
            raise ValueError("SAM3 must return a boolean full-image mask.")
        assign_edit(row, prediction.mask, f"box prompt tool {target}", prediction.score)
    elif command in {"paint", "brush"}:
        mode, _, text = options.partition(" ")
        if mode not in {"add", "erase"}:
            raise ValueError("Choose add or erase for the local edit.")
        if command == "brush":
            radius, _, text = text.partition(" ")
            points = [(x, y) for x, y in coordinates(text)]
            region = stroke_region(blank.shape, points, float(radius))
        else:
            region = blank.copy()
            paint_polygon(region, [(x, y) for x, y in coordinates(text)], add=True)
        mask = source.mask | region if mode == "add" else source.mask & ~region
        assign_edit(row, mask, f"local {mode} tool {target}")
    elif command == "split":
        destination, _, text = options.partition(" ")
        other = find_tool(state, destination)
        ensure_editable(state, other)
        if destination == target:
            raise ValueError("Choose a different destination tool.")
        region = blank.copy()
        paint_polygon(region, [(x, y) for x, y in coordinates(text)], add=True)
        piece = source.mask & region
        if not piece.any():
            raise ValueError("The selected region contains no source-tool pixels.")
        assign_edit(row, source.mask & ~piece, f"split from tool {target}")
        assign_edit(other, resolve(other, False).mask | piece, f"split into tool {destination}")
    else:
        xy = [float(value) for value in options.split()]
        if len(xy) != 2:
            raise ValueError("Use own TOOL_ID X Y at a crossing.")
        source = resolve(row, True)
        overlap = mask_overlap(np.stack([resolve(item, True).mask for item in state.instances]))
        region = crossing_region(overlap, *xy) & source.mask
        if not region.any():
            raise ValueError("The chosen tool does not claim this crossing.")
        if row["Instance ID"] in state.locked and np.any(region & ~resolve(row, False).mask):
            raise ValueError("Ownership would change the locked tool; unlock it first.")
        for other in state.instances:
            other_id = other["Instance ID"]
            if other_id != row["Instance ID"] and np.any(resolve(other, False).mask & region):
                ensure_editable(state, other)
                if not np.any(resolve(other, False).mask & ~region):
                    raise ValueError("Crossing ownership would erase another tool.")
            if other_id in state.ownership:
                state.ownership[other_id] &= ~region
        identity = row["Instance ID"]
        state.ownership[identity] = state.ownership.get(identity, blank) | region
        state.depth_order.clear()
