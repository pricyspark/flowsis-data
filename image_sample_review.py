"""Review a single-tool sample together, with explicit image targets for edits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from image_review import Candidate
from image_ui import candidate_color, collect_clicks, draw_tool_mask
from mask_cleanup import clean_mask, parse_cleanup
from masks import mask2xywh


class SampleBackend(Protocol):
    def predict(
        self, images: Sequence[Image.Image], prompts: Sequence[str]
    ) -> list[list[Candidate]]: ...

    def spatial(
        self,
        image: Image.Image,
        points: list[tuple[float, float, int]],
        seed: NDArray[np.bool_] | None = None,
    ) -> Candidate: ...


@dataclass
class SampleImage:
    image: Image.Image
    filepath: str
    candidates: list[Candidate]
    prompt: str
    decision: str = "pending"
    selected: list[Candidate] = field(default_factory=list)
    ignored: set[int] = field(default_factory=set)
    before_cleanup: dict[int, Candidate] = field(default_factory=dict)


def show_sample(
    images: Sequence[SampleImage], sample_id: str, focus: int | None = None
) -> None:
    import matplotlib.pyplot as plt

    views = list(enumerate(images)) if focus is None else [(focus, images[focus])]
    columns = min(4, len(views))
    rows = (len(views) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows) if focus is None else (12, 9),
        squeeze=False,
    )
    figure.suptitle(
        f"Sample {sample_id} — image numbers start at 0"
        if focus is None
        else f"Sample {sample_id} — focused image {focus}"
    )
    for slot, axis in enumerate(axes.flat):
        axis.axis("off")
        if slot >= len(views):
            continue
        index, item = views[slot]
        preview = item.image.copy()
        preview.thumbnail((800, 800) if focus is None else (1600, 1600))
        extent = (-0.5, item.image.width - 0.5, item.image.height - 0.5, -0.5)
        axis.imshow(preview, extent=extent)
        for candidate_index, candidate in enumerate(item.candidates):
            if item.decision == "accepted" and not any(
                candidate is selected for selected in item.selected
            ):
                continue
            mask = np.asarray(
                Image.fromarray(candidate.mask).resize(
                    preview.size, Image.Resampling.NEAREST
                )
            )
            overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
            overlay[mask] = (
                *[round(channel * 255) for channel in candidate_color(candidate_index)],
                122,
            )
            axis.imshow(overlay, extent=extent, interpolation="nearest")
            box = mask2xywh(candidate.mask)
            if box is not None:
                axis.text(
                    box[0],
                    box[1],
                    f"mask {candidate_index}: {candidate.score:.2f}"
                    + (" [ignored]" if candidate_index in item.ignored else ""),
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.7},
                )
        axis.set_title(
            f"Image {index} | {item.decision}\n{item.filepath}\n{item.prompt}",
            fontsize=9,
        )
    figure.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def _candidate_index(item: SampleImage, text: str) -> int:
    if not text and len(item.candidates) == 1:
        return 0
    if not text.isdecimal() or not 0 <= int(text) < len(item.candidates):
        raise ValueError("Choose exactly one available mask candidate by number.")
    return int(text)


def review_sample(
    images: list[SampleImage],
    sample_id: str,
    backend: SampleBackend,
    *,
    normalize_prompt: Callable[[str], str] = str,
    batch_size: int = 4,
    read_command: Callable[[str], str] | None = None,
) -> bool:
    """Resolve images in place; return whether the user requested quitting.

    Overview commands require image numbers. Focused commands implicitly target
    the focused image; accepting or skipping it returns to the overview.
    """
    import matplotlib.pyplot as plt

    print(f"\nSingle-tool sample {sample_id}: {len(images)} image(s).")
    for index, item in enumerate(images):
        print(f"Image {index}: {item.filepath}")
    focus: int | None = None
    while any(item.decision == "pending" for item in images):
        if focus is None:
            show_sample(images, sample_id)
            print(
                "a=accept remaining unambiguous masks; a X [MASK]=accept image; "
                "focus X=review image individually; "
                "p TEXT=re-prompt pending images; p X TEXT=re-prompt image; "
                "clicks X / draw X [MASK]=spatial prompts; "
                "e X=empty; r [X]=reject; s [X]=skip; "
                "i X [MASKS]=ignore; v X STATE [MASK]=visibility; "
                "clean X [MASK] [OP SIZE] / unclean X [MASK]; q=quit"
            )
        else:
            show_sample(images, sample_id, focus)
            print(
                f"Focused image {focus}: {images[focus].filepath}\n"
                "a [MASK]=accept; p TEXT=re-prompt; clicks/draw [MASK]; "
                "i [MASKS]=ignore; v STATE [MASK]; clean/unclean [MASK] [OP SIZE]; "
                "e=empty; r=reject; s=skip; back=sample overview; q=quit"
            )
        prompt = "sample> " if focus is None else f"image {focus}> "
        response = (read_command or input)(prompt).strip()
        plt.close("all")
        command, _, argument = response.partition(" ")
        argument = argument.strip()
        try:
            if command == "q" and not argument:
                return True
            if command == "back" and not argument:
                focus = None
                continue
            if command == "focus":
                if not argument.isdecimal() or not 0 <= int(argument) < len(images):
                    raise ValueError(f"Use focus X; image X must be 0–{len(images) - 1}.")
                focus = int(argument)
                continue
            if focus is not None:
                argument = f"{focus} {argument}".strip()
            if command in {"a", "r", "s"} and not argument:
                pending = [item for item in images if item.decision == "pending"]
                if command == "a":
                    # Validate the entire selection before changing any decisions.
                    choices = []
                    for item in pending:
                        available = [
                            candidate
                            for index, candidate in enumerate(item.candidates)
                            if index not in item.ignored and candidate.mask.any()
                        ]
                        if len(available) != 1:
                            raise ValueError(
                                "Each pending image needs exactly one nonignored, "
                                "nonempty mask. Use a X MASK, e X, p X TEXT, or s X."
                            )
                        choices.append(available)
                    for item, selected in zip(pending, choices, strict=True):
                        item.selected = selected
                        item.decision = "accepted"
                else:
                    for item in pending:
                        item.decision = "rejected" if command == "r" else "skipped"
                continue
            target, _, rest = argument.partition(" ")
            if command == "p" and target and not target.lstrip("+-").isdecimal():
                prompt = normalize_prompt(argument)
                predicted: list[list[Candidate]] = []
                pending = [item for item in images if item.decision == "pending"]
                for offset in range(0, len(pending), batch_size):
                    batch = pending[offset : offset + batch_size]
                    predicted.extend(
                        backend.predict(
                            [item.image for item in batch], [prompt] * len(batch)
                        )
                    )
                for item, candidates in zip(pending, predicted, strict=True):
                    item.candidates = candidates
                    item.prompt = prompt
                    item.selected = []
                    item.ignored.clear()
                    item.before_cleanup.clear()
                    item.decision = "pending"
                continue
            rest = rest.strip()
            if not target.isdecimal() or not 0 <= int(target) < len(images):
                raise ValueError(
                    f"Use {command} X ...; image X must be 0–{len(images) - 1}."
                )
            item = images[int(target)]
            if command == "p":
                if not rest:
                    raise ValueError("Use p X TEXT.")
                prompt = normalize_prompt(rest)
                candidates = backend.predict([item.image], [prompt])[0]
                item.prompt = prompt
            elif command in {"clicks", "draw"}:
                index = _candidate_index(item, rest) if rest else None
                seed = (
                    item.candidates[index].mask
                    if index is not None
                    else np.zeros((item.image.height, item.image.width), dtype=np.bool_)
                )
                title = f"Sample {sample_id} | Image {target} | {item.filepath}"
                if command == "clicks":
                    points = collect_clicks(item.image, seed, [], title)
                    if not points:
                        continue
                    candidate = backend.spatial(item.image, points)
                else:
                    mask = draw_tool_mask(item.image, seed, title)
                    if mask is None:
                        continue
                    if not mask.any():
                        raise ValueError(
                            "Drawing is empty; use e X to accept an empty image."
                        )
                    candidate = backend.spatial(item.image, [], mask)
                candidates = [candidate]
                item.prompt = candidate.prompt
            elif command == "a":
                index = _candidate_index(item, rest)
                if index in item.ignored or not item.candidates[index].mask.any():
                    raise ValueError("Choose a nonignored, nonempty mask.")
                item.selected = [item.candidates[index]]
                item.decision = "accepted"
                focus = None
                continue
            elif command in {"e", "r", "s"}:
                if rest:
                    raise ValueError(f"Use {command} X.")
                item.selected = []
                item.decision = {
                    "e": "accepted",
                    "r": "rejected",
                    "s": "skipped",
                }[command]
                focus = None
                continue
            elif command == "i":
                indices = {int(value) for value in rest.split(",")} if rest else set()
                if any(index < 0 or index >= len(item.candidates) for index in indices):
                    raise ValueError("Ignore indices must identify available masks.")
                item.ignored = indices
                item.decision = "pending"
                continue
            elif command == "v":
                state, _, text = rest.partition(" ")
                if state not in {"clear", "partial", "severe"}:
                    raise ValueError("Visibility must be clear, partial, or severe.")
                index = _candidate_index(item, text.strip())
                item.candidates[index] = replace(item.candidates[index], visibility=state)
                item.decision = "pending"
                continue
            elif command in {"clean", "unclean"}:
                parts = rest.split()
                index = _candidate_index(
                    item, parts.pop(0) if parts and parts[0].isdecimal() else ""
                )
                if command == "unclean":
                    if parts:
                        raise ValueError("Use unclean X [MASK].")
                    if index in item.before_cleanup:
                        item.candidates[index] = item.before_cleanup.pop(index)
                else:
                    operation, size = parse_cleanup(" ".join(parts))
                    source = item.candidates[index]
                    protected = np.zeros_like(source.mask)
                    for ignored_index in item.ignored - {index}:
                        protected |= item.candidates[ignored_index].mask
                    mask = clean_mask(source.mask, operation, size, protected=protected)
                    if not mask.any():
                        raise ValueError(
                            "Cleanup would erase the tool; previous mask retained."
                        )
                    item.before_cleanup.setdefault(index, source)
                    item.candidates[index] = replace(source, mask=mask)
                item.decision = "pending"
                continue
            else:
                raise ValueError("Invalid sample review command.")
            # New prompts replace the image's alternatives and reopen its decision.
            item.candidates = candidates
            item.selected = []
            item.ignored.clear()
            item.before_cleanup.clear()
            item.decision = "pending"
        except ValueError as error:
            print(error)
    return False
