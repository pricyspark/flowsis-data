"""Explicit image-instance assignment in the Matplotlib/terminal reviewer."""

from __future__ import annotations

import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from annotations import (
    canonical_tool_label,
    stable_media_id,
    validate_instance_metadata,
)
from image_ui import ToolPreview
from mask_cleanup import clean_mask, parse_cleanup
from masks import mask_overlap


@dataclass(frozen=True)
class Candidate:
    mask: NDArray[np.bool_]
    score: float
    prompt: str
    identity: str
    visibility: str = "clear"


@dataclass
class ImageReview:
    decision: str
    instances: list[dict[str, str]]
    selected: list[Candidate]
    ignored: list[Candidate]
    depth_order: list[str] = field(default_factory=list)


@dataclass
class ClickPrompt:
    instance_id: str
    points: list[tuple[float, float, int]] = field(default_factory=list)
    seed: NDArray[np.bool_] | None = None
    candidate_index: int | None = None
    separate: bool = False


def review_image_instances(
    image: Image.Image,
    candidates: Sequence[Candidate],
    instances: Sequence[dict[str, str]],
    *,
    key: str,
    show: Callable[[Image.Image, Sequence[Candidate], str], None],
    predict: Callable[[str], list[Candidate]],
    log: Callable[..., None],
    refine: Callable[[Candidate, str], tuple[Candidate | None, str]] | None = None,
    draw: (
        Callable[[Image.Image, NDArray[np.bool_], str], NDArray[np.bool_] | None] | None
    ) = None,
    collect_points: (
        Callable[
            [Image.Image, NDArray[np.bool_], list[tuple[float, float, int]], str],
            list[tuple[float, float, int]] | None,
        ]
        | None
    ) = None,
    spatial: (
        Callable[[list[tuple[float, float, int]], NDArray[np.bool_] | None], Candidate]
        | None
    ) = None,
    show_tools: (
        Callable[[Image.Image, Sequence[Candidate], str, Sequence[ToolPreview]], None]
        | None
    ) = None,
    inspect_order: (
        Callable[[Image.Image, Sequence[ToolPreview], str], None] | None
    ) = None,
) -> ImageReview:
    import matplotlib.pyplot as plt

    candidates = list(candidates)
    instances = [dict(row) for row in instances]
    assignments: dict[str, list[int]] = {}
    states: dict[str, str] = {}
    excluded: set[int] = set()
    ignored: set[int] = set()
    refinements: dict[str, Candidate] = {}
    cleaned: dict[str, Candidate] = {}
    active_prompts: dict[str, ClickPrompt] = {}
    click_prompts: dict[int, ClickPrompt] = {}
    generated: set[int] = set()
    removed: set[int] = set()
    ordered: dict[str, Candidate] = {}
    depth_order: list[str] = []
    inspect_pending = False
    blank = np.zeros((image.height, image.width), dtype=np.bool_)

    def assigned_candidate(
        row: dict[str, str], *, original: bool = False, apply_order: bool = True
    ) -> Candidate:
        identity = row["Instance ID"]
        if not original and apply_order and identity in ordered:
            return ordered[identity]
        if not original and identity in cleaned:
            return replace(cleaned[identity], visibility=states.get(identity, "clear"))
        if not original and identity in refinements:
            return replace(
                refinements[identity], visibility=states.get(identity, "clear")
            )
        fragments = [candidates[index] for index in assignments.get(identity, [])]
        mask = (
            np.logical_or.reduce([part.mask for part in fragments])
            if fragments
            else np.zeros((image.height, image.width), dtype=np.bool_)
        )
        return Candidate(
            mask,
            max((part.score for part in fragments), default=0),
            row["Tool Class"],
            identity,
            states.get(identity, "clear"),
        )

    def indices(text: str) -> list[int]:
        values = sorted({int(value) for value in text.split(",")})
        if not values or any(value < 0 or value >= len(candidates) for value in values):
            raise ValueError("Provide valid comma-separated candidate indices.")
        missing = sorted(set(values) & removed)
        if missing:
            raise ValueError(
                f"Candidates {missing} have been removed; select active candidates."
            )
        return values

    def record(action: str) -> None:
        log(
            key,
            "in_progress",
            action=action,
            assignments={
                identity: [
                    {
                        "index": index,
                        "candidate_id": candidates[index].identity,
                        "prompt": candidates[index].prompt,
                        "score": candidates[index].score,
                    }
                    for index in values
                ]
                for identity, values in assignments.items()
            },
            visibility=dict(states),
            excluded=sorted(excluded),
            ignored=sorted(ignored),
            removed=sorted(removed),
            cleaned=sorted(cleaned),
            depth_order_tool_ids=list(depth_order),
            refinements={
                identity: {"prompt": candidate.prompt, "score": candidate.score}
                for identity, candidate in refinements.items()
            },
        )

    print(
        "Commands: g=assign, click=correct, a=accept, s=skip, q=quit. "
        "See README.md for details."
    )
    while True:
        raw_selected = [assigned_candidate(row, apply_order=False) for row in instances]
        overlap = mask_overlap(np.stack([part.mask for part in raw_selected]))
        overlap_pixels = int(np.count_nonzero(overlap))
        overlapping_tools = [
            row["Tool ID"]
            for row, part in zip(instances, raw_selected, strict=True)
            if np.any(part.mask & overlap)
        ]
        labels = []
        tools = {row["Instance ID"]: row["Tool ID"] for row in instances}
        for index, candidate in enumerate(candidates):
            if index in removed:
                labels.append(replace(candidate, prompt="[removed]"))
                continue
            owner = next(
                (
                    identity
                    for identity, values in assignments.items()
                    if index in values
                ),
                "unassigned",
            )
            if owner in tools:
                owner = f"tool {tools[owner]}"
            if index in ignored:
                owner = "ignore"
            elif index in excluded:
                owner = "excluded"
            labels.append(replace(candidate, prompt=f"{candidate.prompt} [{owner}]"))
        previews = []
        for row, source in zip(instances, raw_selected, strict=True):
            identity = row["Instance ID"]
            original = None
            edit_note = ""
            if identity in ordered:
                original = source.mask
                edit_note = "Ordered mask — order reset to undo"
            elif identity in refinements or identity in cleaned:
                original = assigned_candidate(row, original=True).mask
                if identity in cleaned:
                    edit_note = f"Cleaned mask — unclean {row['Tool ID']} to undo"
            previews.append(
                ToolPreview(
                    row["Tool ID"],
                    canonical_tool_label(row["Tool Name"], row["Tool Class"], {}),
                    assigned_candidate(row).mask,
                    original,
                    edit_note=edit_note,
                    candidate_index=(
                        assignments[identity][0] if assignments.get(identity) else None
                    ),
                    disputed=source.mask & overlap if overlap_pixels else None,
                )
            )
        if show_tools is None:
            if ordered:
                show(
                    image,
                    [
                        replace(assigned_candidate(row), prompt=f"tool {row['Tool ID']}")
                        for row in instances
                    ],
                    f"{key} — resolved tool masks (source candidate numbers unchanged)",
                )
            else:
                show(image, labels, key)
        else:
            show_tools(image, labels, key, previews)
        if inspect_pending and inspect_order is not None:
            inspect_order(image, previews, key)
            record("inspect_order")
        inspect_pending = False
        if depth_order:
            print(
                f"Front-to-back order: {' > '.join(depth_order)}. "
                "Inspect the close-ups, then 'a' to accept; 'inspect' reopens them; 'order reset' undoes ordering."
            )
        elif overlap_pixels:
            print(
                f"Overlap: {overlap_pixels} pixels claimed by tools {', '.join(overlapping_tools)}. "
                "Correct boundaries or use 'order TOOL_ID ...' from front to back. "
                "Use 'order impossible' to reject an arrangement without a consistent order."
            )
        print(
            f"Mask candidates: {len(candidates) - len(removed)} active (numbers match the preview)"
        )
        for index, candidate in enumerate(labels):
            if index in removed:
                continue
            print(f"  {index}: {candidate.prompt} | score {candidate.score:.2f}")
        if len(removed) == len(candidates):
            prompt = instances[0]["Tool Class"]
            reason = (
                "All candidates were removed."
                if removed
                else "No masks were predicted."
            )
            print(
                f"{reason} Try 'p {prompt}' or a more specific prompt. "
                "Do not accept an empty annotation for a visible tool."
            )
        print("Declared tools (use Tool ID in commands):")
        for row in instances:
            identity = row["Instance ID"]
            print(
                f"  tool {row['Tool ID']} | {canonical_tool_label(row['Tool Name'], row['Tool Class'], {})} | "
                f"confidence {row['Confidence']} | "
                f"{states.get(identity, 'unresolved')} | masks {assignments.get(identity, [])}"
                + (
                    " | refined (restore TOOL_ID to undo)"
                    if identity in refinements
                    else ""
                )
                + (
                    " | cleaned (unclean TOOL_ID to undo)"
                    if identity in cleaned
                    else ""
                )
            )
        for row in instances:
            saved = [
                f"{index} ({len(prompt.points)} clicks)"
                for index, prompt in sorted(click_prompts.items())
                if prompt.instance_id == row["Instance ID"]
            ]
            if saved:
                current = active_prompts.get(row["Instance ID"])
                active_label = current.candidate_index if current is not None else None
                print(
                    f"  tool {row['Tool ID']} click prompts: {', '.join(saved)}; active: {active_label}"
                )
        if len(instances) == 1 and candidates:
            identity = instances[0]["Instance ID"]
            available = [
                index
                for index, candidate in enumerate(candidates)
                if candidate.mask.any()
                and index not in excluded
                and index not in ignored
            ]
            if available and identity not in assignments and identity not in states:
                print(
                    f"If candidate {available[0]} covers this tool, enter "
                    f"'g {instances[0]['Tool ID']} {available[0]}', then 'a'. "
                    "Group multiple pieces with comma-separated candidate numbers."
                )
        response = input("review> ").strip()
        plt.close("all")
        command, _, argument = response.partition(" ")
        try:
            if depth_order and command in {
                "n",
                "p",
                "draw",
                "subtract",
                "click",
                "polygon",
                "clean",
                "unclean",
                "refine",
                "restore",
                "g",
                "u",
                "v",
                "remove",
                "x",
                "i",
                "e",
            }:
                log(
                    key,
                    "in_progress",
                    action="overlap_order_invalidated",
                    depth_order_tool_ids=list(depth_order),
                    command=command,
                )
                ordered.clear()
                depth_order.clear()
                print(
                    "Ordering cleared before editing. Resolve any overlaps again before accepting."
                )
            if command in {"q", "s", "r"}:
                record(command)
                return ImageReview(
                    {"q": "quit", "s": "skipped", "r": "rejected"}[command],
                    instances,
                    [],
                    [],
                )
            if command == "inspect":
                if argument or not depth_order:
                    raise ValueError("Use inspect after setting a front-to-back order.")
                if inspect_order is None:
                    raise ValueError("Overlap inspection is unavailable in this reviewer.")
                inspect_pending = True
            elif command == "order":
                if argument.strip() == "reset":
                    ordered.clear()
                    depth_order.clear()
                    record("reset_overlap_order")
                    continue
                if argument.strip() == "impossible":
                    record("incompatible_depth_order")
                    return ImageReview("rejected", instances, [], [])
                tool_ids = argument.replace(",", " ").split()
                by_tool = {
                    row["Tool ID"]: part
                    for row, part in zip(instances, raw_selected, strict=True)
                }
                if not tool_ids or len(set(tool_ids)) != len(tool_ids):
                    raise ValueError(
                        "List each Tool ID once, front to back: order TOOL_ID ..."
                    )
                if any(tool_id not in by_tool for tool_id in tool_ids):
                    raise ValueError(
                        "Unknown Tool ID in order. Use the declared physical Tool IDs."
                    )
                if any(not by_tool[tool_id].mask.any() for tool_id in tool_ids):
                    raise ValueError(
                        "Only tools with assigned visible masks can be ordered."
                    )
                missing = set(overlapping_tools) - set(tool_ids)
                if missing:
                    raise ValueError(
                        f"Include every overlapping tool in the order; missing: {', '.join(sorted(missing))}."
                    )
                if not overlap_pixels:
                    raise ValueError(
                        "No assigned instance masks overlap; ordering is unnecessary."
                    )
                resolved: dict[str, Candidate] = {}
                removed_pixels: dict[str, int] = {}
                occupied = blank.copy()
                for tool_id in tool_ids:
                    source = by_tool[tool_id]
                    mask = source.mask & ~occupied
                    if not mask.any():
                        raise ValueError(
                            f"Ordering would erase tool {tool_id}. Correct duplicate masks, change the order, "
                            f"or explicitly mark 'v {tool_id} occluded' if fully hidden. Previous order retained."
                        )
                    resolved[source.identity] = replace(source, mask=mask)
                    removed_pixels[tool_id] = int(
                        np.count_nonzero(source.mask & occupied)
                    )
                    occupied |= source.mask
                ordered = resolved
                depth_order = tool_ids
                inspect_pending = True
                log(
                    key,
                    "in_progress",
                    action="overlap_order",
                    depth_order_tool_ids=list(depth_order),
                    depth_order_instance_ids=[
                        by_tool[tool_id].identity for tool_id in tool_ids
                    ],
                    overlap_pixels=overlap_pixels,
                    removed_pixels=removed_pixels,
                )
                print(
                    "Ordering applied to disputed pixels only. Inspect the preview before 'a'; edits clear the order."
                )
            elif command == "n":
                fields = shlex.split(argument)
                if len(fields) == 4:
                    tool_id, tool_class, name, confidence = fields
                    identity = stable_media_id("instance", f"{key}/{tool_id}")
                elif len(fields) == 5:
                    identity, tool_id, tool_class, name, confidence = fields
                else:
                    raise ValueError('Use: n TOOL_ID "CLASS" "NAME" CONFIDENCE.')
                row = {
                    "Image ID": key,
                    "Instance ID": identity,
                    "Tool ID": tool_id,
                    "Tool Class": tool_class,
                    "Tool Name": name,
                    "Confidence": confidence,
                }
                validate_instance_metadata([*instances, row])
                instances.append(row)
                log(key, "in_progress", action="declare", instance=row)
            elif command == "p" and argument:
                additions = predict(argument)
                start = len(candidates)
                print(
                    f"Prompt {argument!r} returned {len(additions)} mask candidate(s)."
                )
                candidates.extend(
                    replace(candidate, identity=f"candidate-{start + index}")
                    for index, candidate in enumerate(additions)
                )
                log(
                    key,
                    "in_progress",
                    action="reprompt",
                    prompt=argument,
                    candidate_indices=list(range(start, len(candidates))),
                )
            elif command in {"draw", "subtract", "click", "polygon"}:
                parts = argument.split()
                if (
                    (command in {"draw", "polygon"} and len(parts) != 1)
                    or (command == "subtract" and len(parts) != 3)
                    or (
                        command == "click"
                        and (
                            len(parts) not in {1, 2}
                            or (
                                len(parts) == 2
                                and parts[1] not in {"reset", "new"}
                                and not parts[1].isdigit()
                            )
                        )
                    )
                ):
                    raise ValueError(
                        "Use draw TOOL_ID or subtract TOOL_ID BASE_INDEX OTHER_TOOL_ID[,OTHER_TOOL_ID...]; click TOOL_ID [new|INDEX|reset] or polygon TOOL_ID for SAM3 prompts."
                    )
                target_tool = parts[0]
                by_tool = {row["Tool ID"]: row for row in instances}
                if target_tool not in by_tool:
                    raise ValueError(
                        "Unknown Tool ID. Use an ID from the declared tools list."
                    )
                target_instance = by_tool[target_tool]["Instance ID"]
                details: dict[str, object]
                click_prompt: ClickPrompt | None = None
                if command == "click":
                    if collect_points is None or spatial is None:
                        raise ValueError(
                            "SAM3 point prompting is unavailable in this reviewer."
                        )
                    option = parts[1] if len(parts) == 2 else ""
                    if option == "new":
                        click_prompt = ClickPrompt(target_instance, separate=True)
                    elif option.isdigit():
                        selected_index = indices(option)[0]
                        click_prompt = click_prompts.get(selected_index)
                        if (
                            click_prompt is None
                            or click_prompt.instance_id != target_instance
                        ):
                            raise ValueError(
                                f"Candidate {selected_index} has no click prompt for tool {target_tool}."
                            )
                    else:
                        click_prompt = active_prompts.get(target_instance)
                        if click_prompt is None:
                            matching = [
                                click_prompts[index]
                                for index in assignments.get(target_instance, [])
                                if index in click_prompts
                                and click_prompts[index].instance_id == target_instance
                            ]
                            click_prompt = (
                                matching[-1]
                                if matching
                                else ClickPrompt(target_instance)
                            )
                        if option == "reset":
                            click_prompt = replace(click_prompt, points=[], seed=None)
                    edit_index = click_prompt.candidate_index
                    if edit_index is not None and any(
                        edit_index in assigned
                        for owner, assigned in assignments.items()
                        if owner != target_instance
                    ):
                        raise ValueError(
                            f"Candidate {edit_index} is assigned to another tool; clear that assignment before editing it."
                        )
                    context = (
                        candidates[edit_index].mask
                        if edit_index is not None
                        else assigned_candidate(by_tool[target_tool]).mask
                    )
                    if option == "new":
                        context = blank
                    label = (
                        f"candidate {edit_index}"
                        if edit_index is not None
                        else "new prompt"
                    )
                    points = collect_points(
                        image,
                        context,
                        list(click_prompt.points),
                        f"Tool {target_tool} — {canonical_tool_label(by_tool[target_tool]['Tool Name'], by_tool[target_tool]['Tool Class'], {})} — {label}",
                    )
                    if points is None:
                        continue
                    click_prompt.points = list(points)
                    active_prompts[target_instance] = click_prompt
                    if edit_index is not None:
                        click_prompts[edit_index] = click_prompt
                    prediction = spatial(points, click_prompt.seed)
                    if (
                        prediction.mask.shape != (image.height, image.width)
                        or prediction.mask.dtype != np.bool_
                    ):
                        raise ValueError("SAM3 must return a boolean full-image mask.")
                    derived = prediction
                    details = {
                        "prompt_kind": "points",
                        "points": points,
                        "polygon_seed": click_prompt.seed is not None,
                        "revised_candidate": edit_index,
                        "new_portion": click_prompt.separate,
                    }
                elif command in {"draw", "polygon"}:
                    if command == "polygon" and spatial is None:
                        raise ValueError(
                            "SAM3 polygon prompting is unavailable in this reviewer."
                        )
                    if draw is None:
                        raise ValueError(
                            "Polygon drawing is unavailable in this reviewer."
                        )
                    drawn = draw(
                        image,
                        assigned_candidate(by_tool[target_tool]).mask.copy(),
                        f"Tool {target_tool} — {canonical_tool_label(by_tool[target_tool]['Tool Name'], by_tool[target_tool]['Tool Class'], {})}",
                    )
                    if drawn is None:
                        continue
                    if (
                        drawn.shape != (image.height, image.width)
                        or drawn.dtype != np.bool_
                    ):
                        raise ValueError(
                            "Drawing must return a boolean full-image mask."
                        )
                    derived = Candidate(
                        drawn,
                        1.0,
                        f"manually drawn tool {target_tool}",
                        f"candidate-{len(candidates)}",
                    )
                    details = {"method": "polygon"}
                    if command == "polygon":
                        if not drawn.any():
                            raise ValueError(
                                "Draw a nonempty coarse polygon around the target tool first."
                            )
                        assert spatial is not None
                        click_prompt = ClickPrompt(target_instance, seed=drawn.copy())
                        active_prompts[target_instance] = click_prompt
                        prediction = spatial([], drawn)
                        if (
                            prediction.mask.shape != (image.height, image.width)
                            or prediction.mask.dtype != np.bool_
                        ):
                            raise ValueError(
                                "SAM3 must return a boolean full-image mask."
                            )
                        derived = replace(
                            prediction, identity=f"candidate-{len(candidates)}"
                        )
                        details = {"prompt_kind": "polygon_mask"}
                else:
                    bases = indices(parts[1])
                    if len(bases) != 1:
                        raise ValueError(
                            "Choose exactly one combined candidate as BASE_INDEX."
                        )
                    source_index = bases[0]
                    removed_tools = list(dict.fromkeys(parts[2].split(",")))
                    if target_tool in removed_tools:
                        raise ValueError("Cannot subtract the target tool from itself.")
                    remainder = candidates[source_index].mask.copy()
                    for tool_id in removed_tools:
                        if tool_id not in by_tool:
                            raise ValueError(f"Unknown Tool ID {tool_id} to subtract.")
                        remove = assigned_candidate(by_tool[tool_id]).mask
                        if not remove.any():
                            raise ValueError(
                                f"Assign or draw a correct visible mask for tool {tool_id} before subtracting it."
                            )
                        remainder &= ~remove
                    if np.array_equal(remainder, candidates[source_index].mask):
                        raise ValueError(
                            "The selected masks do not overlap the combined candidate; nothing was subtracted."
                        )
                    derived = replace(
                        candidates[source_index],
                        mask=remainder,
                        identity=f"candidate-{len(candidates)}",
                        prompt=f"candidate {source_index} minus tools {','.join(removed_tools)}",
                    )
                    details = {
                        "source_index": source_index,
                        "source_candidate_id": candidates[source_index].identity,
                        "subtracted_tool_ids": removed_tools,
                    }
                if not derived.mask.any():
                    raise ValueError(
                        "The resulting mask is empty; the tool was not assigned."
                    )
                edit_index = (
                    click_prompt.candidate_index
                    if command == "click" and click_prompt is not None
                    else None
                )
                keep_parts = (
                    command == "click"
                    and click_prompt is not None
                    and (edit_index is not None or click_prompt.separate)
                )
                if edit_index is None:
                    new_index = len(candidates)
                    derived = replace(derived, identity=f"candidate-{new_index}")
                    candidates.append(derived)
                else:
                    new_index = edit_index
                    candidates[new_index] = replace(
                        derived, identity=candidates[new_index].identity
                    )
                if keep_parts:
                    assignments[target_instance] = sorted(
                        set(assignments.get(target_instance, [])) | {new_index}
                    )
                else:
                    excluded.update(
                        set(assignments.get(target_instance, [])) & generated
                    )
                    assignments[target_instance] = [new_index]
                excluded.discard(new_index)
                ignored.discard(new_index)
                generated.add(new_index)
                if click_prompt is not None:
                    click_prompt.candidate_index = new_index
                    click_prompts[new_index] = click_prompt
                else:
                    active_prompts.pop(target_instance, None)
                cleaned.pop(target_instance, None)
                refinements.pop(target_instance, None)
                if states.get(target_instance) not in {"clear", "partial", "severe"}:
                    states[target_instance] = "clear"
                log(
                    key,
                    "in_progress",
                    action=command,
                    tool_id=target_tool,
                    instance_id=target_instance,
                    candidate_index=new_index,
                    **details,
                )
                record(f"{command}_assignment")
                print(
                    f"Tool {target_tool}: {'updated' if edit_index is not None else 'assigned new'} candidate {new_index}. Inspect its colored pixels before accepting."
                )
                if command == "subtract":
                    print(
                        f"Combined candidate {source_index} is unchanged. Exclude it with 'x {source_index}' when no longer needed."
                    )
            elif command in {"clean", "unclean"}:
                target, _, options = argument.partition(" ")
                cleanup_row = next(
                    (row for row in instances if row["Tool ID"] == target), None
                )
                if cleanup_row is None:
                    raise ValueError("Use clean TOOL_ID [OP SIZE] or unclean TOOL_ID.")
                identity = cleanup_row["Instance ID"]
                if command == "unclean":
                    if options.strip():
                        raise ValueError("Use unclean TOOL_ID.")
                    cleaned.pop(identity, None)
                    record("undo_cleanup")
                    continue
                operation, size = parse_cleanup(options)
                source = assigned_candidate(cleanup_row)
                if not source.mask.any():
                    raise ValueError("Assign a visible mask before cleanup.")
                protected = blank.copy()
                for other in instances:
                    if other["Instance ID"] != identity:
                        protected |= assigned_candidate(other).mask
                for index in ignored:
                    protected |= candidates[index].mask
                result = clean_mask(source.mask, operation, size, protected=protected)
                if not result.any():
                    raise ValueError(
                        "Cleanup would erase the whole tool; previous mask retained. Try a smaller size."
                    )
                added = int(np.count_nonzero(result & ~source.mask))
                removed_pixels = int(np.count_nonzero(source.mask & ~result))
                cleaned[identity] = replace(source, mask=result)
                log(
                    key,
                    "in_progress",
                    action="cleanup",
                    tool_id=target,
                    instance_id=identity,
                    operation=operation,
                    size=size,
                    added_pixels=added,
                    removed_pixels=removed_pixels,
                )
                record("cleanup_assignment")
                print(
                    f"Tool {target}: cleanup added {added} and removed {removed_pixels} pixels. Inspect the preview; 'unclean {target}' undoes all cleanup for this tool."
                )
            elif command in {"refine", "restore"}:
                target, _, prompt_override = argument.partition(" ")
                targets = (
                    instances
                    if target == "all"
                    else [row for row in instances if row["Tool ID"] == target]
                )
                if not targets:
                    raise ValueError(f"Use {command} TOOL_ID or {command} all.")
                if target == "all" and prompt_override:
                    raise ValueError(
                        "Prompt overrides apply to one Tool ID, not all tools."
                    )
                if command == "refine" and refine is None:
                    raise ValueError("Crop refinement is unavailable in this reviewer.")
                if (
                    command == "refine"
                    and target != "all"
                    and not assignments.get(targets[0]["Instance ID"])
                ):
                    raise ValueError(
                        f"Assign tool {target} a mask with 'g {target} INDEX' before refinement."
                    )
                for row in targets:
                    identity = row["Instance ID"]
                    if command == "restore":
                        cleaned.pop(identity, None)
                        refinements.pop(identity, None)
                        print(
                            f"Tool {row['Tool ID']}: restored original assigned mask."
                        )
                    elif assignments.get(identity):
                        assert refine is not None
                        prompt = prompt_override or canonical_tool_label(
                            row["Tool Name"], row["Tool Class"], {}
                        )
                        replacement, message = refine(assigned_candidate(row), prompt)
                        if replacement is not None:
                            if (
                                replacement.mask.shape != (image.height, image.width)
                                or not replacement.mask.any()
                            ):
                                raise ValueError(
                                    "Refinement must return a nonempty full-image mask."
                                )
                            cleaned.pop(identity, None)
                            refinements[identity] = replace(
                                replacement, identity=identity
                            )
                        print(f"Tool {row['Tool ID']}: {message}")
                        log(
                            key,
                            "in_progress",
                            action="crop_refinement",
                            tool_id=row["Tool ID"],
                            instance_id=identity,
                            prompt=prompt,
                            applied=replacement is not None,
                            message=message,
                        )
                    else:
                        print(
                            f"Tool {row['Tool ID']}: no visible mask assigned; skipped refinement."
                        )
                record(command)
            elif command in {"g", "u", "v"}:
                target, _, value = argument.partition(" ")
                target_row = next(
                    (row for row in instances if row["Tool ID"] == target), None
                )
                if target_row is None:
                    # Accept old commands while keeping Tool IDs the primary interface.
                    target_row = next(
                        (row for row in instances if row["Instance ID"] == target), None
                    )
                if target_row is None:
                    raise ValueError(
                        "Unknown Tool ID. Use an ID from the declared tools list."
                    )
                row = target_row
                identity = row["Instance ID"]
                if command == "g":
                    if not value.strip():
                        raise ValueError(
                            f"Provide a candidate number: g {row['Tool ID']} 0 (or 0,2 to group pieces)."
                        )
                    values = indices(value)
                    if any(not candidates[index].mask.any() for index in values):
                        raise ValueError("Cannot assign an empty mask as visible.")
                    if any(
                        set(values) & set(assigned)
                        for owner, assigned in assignments.items()
                        if owner != identity
                    ):
                        raise ValueError(
                            "Clear the other tool's assignment before reusing a candidate."
                        )
                    assignments[identity] = values
                    active_prompts.pop(identity, None)
                    cleaned.pop(identity, None)
                    refinements.pop(identity, None)
                    excluded.difference_update(values)
                    ignored.difference_update(values)
                    if states.get(identity) not in {"clear", "partial", "severe"}:
                        states[identity] = "clear"
                elif command == "u":
                    active_prompts.pop(identity, None)
                    cleaned.pop(identity, None)
                    refinements.pop(identity, None)
                    assignments.pop(identity, None)
                    states.pop(identity, None)
                else:
                    if value not in {
                        "clear",
                        "partial",
                        "severe",
                        "occluded",
                        "absent",
                    }:
                        raise ValueError("Invalid visibility state.")
                    states[identity] = value
                    if value in {"occluded", "absent"}:
                        active_prompts.pop(identity, None)
                        cleaned.pop(identity, None)
                        refinements.pop(identity, None)
                        assignments.pop(identity, None)
                record(
                    {
                        "g": "assign_fragments",
                        "u": "clear_assignment",
                        "v": "visibility",
                    }[command]
                )
            elif command == "remove":
                values = indices(argument)
                cleared_tools = []
                for row in instances:
                    identity = row["Instance ID"]
                    if set(values) & set(assignments.get(identity, [])):
                        assignments.pop(identity, None)
                        states.pop(identity, None)
                        cleaned.pop(identity, None)
                        refinements.pop(identity, None)
                        active_prompts.pop(identity, None)
                        cleared_tools.append(row["Tool ID"])
                for index in values:
                    click_prompts.pop(index, None)
                for owner, saved_prompt in list(active_prompts.items()):
                    if saved_prompt.candidate_index in values:
                        active_prompts.pop(owner)
                removed.update(values)
                excluded.difference_update(values)
                ignored.difference_update(values)
                generated.difference_update(values)
                for index in values:
                    # Leave an empty slot so candidate numbers and colors never shift.
                    candidates[index] = replace(candidates[index], mask=blank)
                record("remove_candidates")
                print(
                    f"Removed candidates {values}; remaining candidate numbers are unchanged."
                )
                if cleared_tools:
                    print(
                        f"Cleared assignments for tools {', '.join(cleared_tools)}. Assign new masks before accepting."
                    )
            elif command in {"x", "i"}:
                values = indices(argument)
                if any(
                    set(values) & set(assigned) for assigned in assignments.values()
                ):
                    raise ValueError(
                        "Clear the tool assignment before excluding its candidates."
                    )
                excluded.difference_update(values)
                ignored.difference_update(values)
                (ignored if command == "i" else excluded).update(values)
                record("ignore" if command == "i" else "exclude")
            elif command in {"a", "e"}:
                if command == "e":
                    active_prompts.clear()
                    click_prompts.clear()
                    cleaned.clear()
                    refinements.clear()
                    assignments.clear()
                    states = {row["Instance ID"]: "absent" for row in instances}
                    excluded.update(set(range(len(candidates))) - ignored - removed)
                validate_instance_metadata(instances)
                for row in instances:
                    identity = row["Instance ID"]
                    state = states.get(identity)
                    if state is None or (
                        state in {"clear", "partial", "severe"}
                        and not assignments.get(identity)
                    ):
                        if len(removed) == len(candidates):
                            raise ValueError(
                                f"Resolve tool {row['Tool ID']} before acceptance. No mask candidates "
                                f"are available; try 'p {row['Tool Class']}' or a more specific prompt. "
                                f"Use 'v {row['Tool ID']} absent' or 'v {row['Tool ID']} occluded' "
                                "only if the tool is absent or fully hidden."
                            )
                        raise ValueError(
                            f"Resolve tool {row['Tool ID']} before acceptance. Assign its mask with "
                            f"'g {row['Tool ID']} INDEX' (comma-separated indices combine pieces), "
                            f"or explicitly mark it 'v {row['Tool ID']} absent' or "
                            f"'v {row['Tool ID']} occluded' if appropriate."
                        )
                used = {index for values in assignments.values() for index in values}
                unresolved = (
                    set(range(len(candidates))) - used - excluded - ignored - removed
                )
                if unresolved:
                    raise ValueError(
                        f"Assign, exclude, or ignore candidates {sorted(unresolved)}."
                    )
                selected = [assigned_candidate(row) for row in instances]
                if mask_overlap(np.stack([part.mask for part in selected])).any():
                    raise ValueError(
                        "Instance masks overlap. Correct boundaries or use 'order TOOL_ID ...' "
                        "from front to back, inspect the result, then 'a'. "
                        "Use 'order impossible' to reject incompatible arrangements."
                    )
                record("resolved")
                return ImageReview(
                    "accepted",
                    instances,
                    selected,
                    [candidates[index] for index in sorted(ignored)],
                    list(depth_order),
                )
            else:
                raise ValueError("Invalid review command.")
        except ValueError as error:
            print(error)
