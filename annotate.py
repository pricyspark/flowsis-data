from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from annotations import (
    CAPTURE_SESSION_FIELD,
    IMAGE_ID_FIELD,
    IMAGE_OBJECT_FIELDS,
    VIDEO_MANIFEST_FIELDS,
    VIDEO_OBJECT_FIELDS,
    canonical_tool_label,
    extend_image_manifest,
    image_mask_path,
    load_taxonomy_aliases,
    normalize_tool_name,
    read_csv,
    stable_media_id,
    write_csv,
    write_image_manifests,
)
from masks import mask2xywh, save_mask_bundle


@dataclass(frozen=True)
class Candidate:
    mask: NDArray[np.bool_]
    score: float
    prompt: str
    identity: str
    visibility: str = "clear"


class ReviewLog:
    """Append-only decisions; the latest decision for a key wins on resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.decisions: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.decisions[str(record["key"])] = record

    def completed(self, key: str) -> bool:
        return self.decisions.get(key, {}).get("decision") in {
            "accepted",
            "rejected",
        }

    def append(self, key: str, decision: str, **values: Any) -> None:
        record = {"key": key, "decision": decision, **values}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            file.flush()
        self.decisions[key] = record


def _as_numpy(value: Any) -> NDArray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_candidates(
    output: Mapping[str, Any] | None,
    *,
    prompt: str,
    identity_prefix: str = "instance",
    keep_empty: bool = False,
) -> list[Candidate]:
    if output is None or output.get("masks") is None:
        return []
    masks = _as_numpy(output["masks"])
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected candidate masks [N,H,W], got {masks.shape}.")
    scores_value = output.get("scores")
    scores = (
        np.ones(len(masks), dtype=np.float32)
        if scores_value is None
        else _as_numpy(scores_value).reshape(-1)
    )
    object_ids = output.get("object_ids")
    identities = (
        [f"{identity_prefix}-{index}" for index in range(len(masks))]
        if object_ids is None
        else [str(value) for value in _as_numpy(object_ids).reshape(-1).tolist()]
    )
    if len(scores) != len(masks) or len(identities) != len(masks):
        raise ValueError("SAM3 masks, scores, and identities must align.")
    return [
        Candidate(mask.astype(np.bool_, copy=False), float(score), prompt, identity)
        for mask, score, identity in zip(masks, scores, identities, strict=True)
        if keep_empty or bool(mask.any())
    ]


def mask_iou(left: NDArray[np.bool_], right: NDArray[np.bool_]) -> float:
    intersection = np.count_nonzero(left & right)
    union = np.count_nonzero(left | right)
    return intersection / union if union else 0.0


def deduplicate_candidates(
    candidates: Iterable[Candidate], *, threshold: float = 0.9
) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.identity.startswith("instance-"), item.score),
        reverse=True,
    ):
        if any(
            mask_iou(candidate.mask, existing.mask) >= threshold for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def image_prompt(row: Mapping[str, str], style: str, aliases: Mapping[str, str]) -> str:
    tool_class = " ".join(row.get("Tool Class", "").strip().split())
    tool_name = normalize_tool_name(row.get("Tool Name", ""), aliases)
    if style == "tool-class":
        prompt = tool_class or tool_name
    elif style == "name-and-class":
        prompt = canonical_tool_label(tool_name, tool_class, aliases)
    else:
        raise ValueError(f"Unknown image prompt style {style!r}.")
    if not prompt:
        raise ValueError("An image row must have a Tool Class or Tool Name prompt.")
    return prompt


def show_candidates(
    image: Image.Image, candidates: Sequence[Candidate], title: str
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(image)
    colors = plt.get_cmap("tab20")
    boxes: list[list[int]] = []
    for index, candidate in enumerate(candidates):
        overlay = np.zeros((*candidate.mask.shape, 4), dtype=np.float32)
        overlay[candidate.mask] = (*colors(index % 20)[:3], 0.48)
        axis.imshow(overlay, interpolation="nearest")
        box = mask2xywh(candidate.mask)
        if box is not None:
            boxes.append(box)
            x, y, width, height = box
            axis.text(
                x,
                y,
                f"{index}: {candidate.prompt} {candidate.score:.2f}",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.7},
            )
            axis.add_patch(
                plt.Rectangle((x, y), width, height, fill=False, color=colors(index))
            )
    if boxes:
        left = min(box[0] for box in boxes)
        top = min(box[1] for box in boxes)
        right = max(box[0] + box[2] for box in boxes)
        bottom = max(box[1] + box[3] for box in boxes)
        padding = max(1, int(np.ceil(0.2 * max(right - left, bottom - top))))
        axis.set_xlim(
            max(0, left - padding) - 0.5,
            min(image.width, right + padding) - 0.5,
        )
        axis.set_ylim(
            min(image.height, bottom + padding) - 0.5,
            max(0, top - padding) - 0.5,
        )
    axis.set_title(title)
    axis.axis("off")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def _parse_indices(text: str, count: int) -> list[int]:
    if not text.strip():
        return list(range(count))
    indices = sorted({int(value) for value in text.split(",")})
    if any(index < 0 or index >= count for index in indices):
        raise ValueError(f"Candidate index must be between 0 and {count - 1}.")
    return indices


def review_candidates(
    image: Image.Image,
    candidates: Sequence[Candidate],
    *,
    key: str,
    auto_accept: bool,
    allow_occluded: bool = False,
    review_log: ReviewLog | None = None,
) -> tuple[str, list[Candidate], list[Candidate], str | None, str]:
    """Return decision, selected/ignored candidates, new prompt, and visibility."""
    import matplotlib.pyplot as plt

    if auto_accept:
        return "accepted", list(candidates), [], None, "clear"
    visibility = "clear"
    visibility_by_index = {index: "clear" for index in range(len(candidates))}
    ignored_indices: set[int] = set()
    while True:
        show_candidates(image, candidates, key)
        print(
            "a [0,2]=accept, e=accept empty, r=reject, s=skip, "
            "p TEXT=re-prompt, i [0,2]=mark ignore regions, "
            "v STATE [0,2]=set visibility, "
            + ("o [0,2]=retain occluded tracks, " if allow_occluded else "")
            + "q=quit"
        )
        response = input("review> ").strip()
        plt.close("all")
        command, _, argument = response.partition(" ")
        try:
            if command == "a":
                accepted_indices = set(_parse_indices(argument, len(candidates)))
                return (
                    "accepted",
                    [
                        replace(
                            candidates[index],
                            visibility=visibility_by_index[index],
                        )
                        for index in sorted(accepted_indices - ignored_indices)
                    ],
                    [candidates[index] for index in sorted(ignored_indices)],
                    None,
                    visibility,
                )
            if command == "e":
                return (
                    "accepted",
                    [],
                    [candidates[index] for index in sorted(ignored_indices)],
                    None,
                    visibility,
                )
            if command == "i":
                ignored_indices = set(_parse_indices(argument, len(candidates)))
                if review_log is not None:
                    review_log.append(
                        key,
                        "in_progress",
                        action="ignore",
                        indices=sorted(ignored_indices),
                    )
                continue
            if command == "o" and allow_occluded:
                selected = [
                    replace(candidates[index], visibility="occluded")
                    for index in _parse_indices(argument, len(candidates))
                ]
                return "accepted", selected, [], None, "occluded"
            if command == "r":
                return "rejected", [], [], None, visibility
            if command == "s":
                return "skipped", [], [], None, visibility
            if command == "p" and argument:
                if review_log is not None:
                    review_log.append(
                        key, "in_progress", action="reprompt", prompt=argument
                    )
                return "reprompt", [], [], argument, visibility
            if command == "v":
                state, _, indices_text = argument.partition(" ")
                if state not in {"clear", "partial", "severe"}:
                    raise ValueError("Visibility must be clear, partial, or severe.")
                indices = _parse_indices(indices_text, len(candidates))
                for index in indices:
                    visibility_by_index[index] = state
                visibility = state
                if review_log is not None:
                    review_log.append(
                        key,
                        "in_progress",
                        action="visibility",
                        visibility=state,
                        indices=indices,
                    )
                continue
            if command == "q":
                return "quit", [], [], None, visibility
        except ValueError as error:
            print(error)
            continue
        print("Invalid review command.")


def merge_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Union one prompt's tool fragments, keeping the strongest candidate's metadata."""
    if len(candidates) < 2:
        return candidates
    strongest = max(candidates, key=lambda candidate: candidate.score)
    return [
        replace(
            strongest,
            mask=np.logical_or.reduce([candidate.mask for candidate in candidates]),
        )
    ]


class Sam3ImageBackend:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        threshold: float,
        *,
        refine_crop: bool = True,
        single_tool: bool = False,
    ) -> None:
        from transformers import Sam3Model, Sam3Processor

        self.model = Sam3Model.from_pretrained(model_name).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(model_name)
        self.device = device
        self.threshold = threshold
        self.refine_crop = refine_crop
        self.single_tool = single_tool

    def predict(
        self, images: Sequence[Image.Image], prompts: Sequence[str]
    ) -> list[list[Candidate]]:
        candidates = self._predict(images, prompts)
        if self.single_tool:
            candidates = [merge_candidates(result) for result in candidates]
        if not self.refine_crop:
            return candidates

        crops: list[Image.Image] = []
        crop_prompts: list[str] = []
        regions: list[tuple[int, int, int, int, int]] = []
        for index, (image, prompt, initial) in enumerate(
            zip(images, prompts, candidates, strict=True)
        ):
            initial = deduplicate_candidates(initial)
            if len(initial) != 1:
                continue
            box = mask2xywh(initial[0].mask)
            if box is None:
                continue
            x, y, width, height = box
            padding = max(1, int(np.ceil(0.2 * max(width, height))))
            left, top = max(0, x - padding), max(0, y - padding)
            right = min(image.width, x + width + padding)
            bottom = min(image.height, y + height + padding)
            if (left, top, right, bottom) == (0, 0, image.width, image.height):
                continue
            crops.append(image.crop((left, top, right, bottom)))
            crop_prompts.append(prompt)
            regions.append((index, left, top, right, bottom))

        if not crops:
            return candidates
        refined = self._predict(crops, crop_prompts)
        if self.single_tool:
            refined = [merge_candidates(result) for result in refined]
        for (index, left, top, right, bottom), result in zip(
            regions, refined, strict=True
        ):
            result = deduplicate_candidates(result)
            if len(result) != 1:
                continue
            candidate = result[0]
            mask = candidate.mask
            image = images[index]
            if (
                (left > 0 and mask[:, 0].any())
                or (top > 0 and mask[0, :].any())
                or (right < image.width and mask[:, -1].any())
                or (bottom < image.height and mask[-1, :].any())
            ):
                # A mask touching an interior crop edge may cut off the tool.
                continue
            full_mask = np.zeros((image.height, image.width), dtype=np.bool_)
            full_mask[top:bottom, left:right] = mask
            original = deduplicate_candidates(candidates[index])[0]
            candidates[index] = [
                replace(candidate, mask=full_mask, identity=original.identity)
            ]
        return candidates

    @torch.inference_mode()
    def _predict(
        self, images: Sequence[Image.Image], prompts: Sequence[str]
    ) -> list[list[Candidate]]:
        inputs = self.processor(
            images=list(images), text=list(prompts), return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )
        return [
            extract_candidates(result, prompt=prompt)
            for result, prompt in zip(results, prompts, strict=True)
        ]


def _visibility_flags(visibility: str) -> dict[str, str]:
    severe = visibility == "severe"
    visible = visibility != "occluded"
    return {
        "Visibility": visibility,
        "Reviewed": "TRUE",
        "Annotation Complete": "TRUE",
        "Mask Valid": str(visible).upper(),
        "Class Valid": str(visible and not severe).upper(),
        "Base Valid": str(visible and not severe).upper(),
        "Visible": str(visible).upper(),
        "Annotation Confidence": "1.0",
    }


def _replace_scope(
    rows: list[dict[str, object]],
    replacements: list[dict[str, object]],
    *,
    scope: Mapping[str, object],
) -> list[dict[str, object]]:
    retained = [
        row
        for row in rows
        if not all(str(row.get(key, "")) == str(value) for key, value in scope.items())
    ]
    return retained + replacements


def _image_object_rows(
    source: Mapping[str, str],
    candidates: Sequence[Candidate],
    mask_path: Path,
    intended_prompt: str,
) -> list[dict[str, object]]:
    rows = []
    intended_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.prompt.casefold() == intended_prompt.casefold()
        and candidate.identity.startswith("instance-")
    ]
    intended_index = (
        0
        if len(candidates) == 1
        else (intended_indices[0] if intended_indices else None)
    )
    for index, candidate in enumerate(candidates):
        box = mask2xywh(candidate.mask)
        if box is None:
            continue
        rows.append(
            {
                IMAGE_ID_FIELD: source[IMAGE_ID_FIELD],
                "Instance ID": candidate.identity,
                "Tool ID": (
                    source.get("Tool ID", "") if index == intended_index else ""
                ),
                "Tool Class": (
                    source.get("Tool Class", "") if index == intended_index else ""
                ),
                "Tool Name": (
                    source.get("Tool Name", "")
                    if index == intended_index
                    else candidate.prompt
                ),
                "Confidence": (
                    source.get("Confidence", "") if index == intended_index else ""
                ),
                "BBox X": box[0],
                "BBox Y": box[1],
                "BBox Width": box[2],
                "BBox Height": box[3],
                "Mask Path": str(mask_path),
                "Mask Index": index,
                **_visibility_flags(candidate.visibility),
            }
        )
    return rows


def annotate_images(args: argparse.Namespace) -> None:
    manifest = extend_image_manifest(args.manifest)
    rows = read_csv(manifest)
    mask_paths = [image_mask_path(args.mask_dir, row["Filepath"]) for row in rows]
    if len(set(mask_paths)) != len(mask_paths):
        raise ValueError("Image filenames map to duplicate mask paths.")
    object_rows: list[dict[str, object]] = (
        list(read_csv(args.object_manifest)) if args.object_manifest.is_file() else []
    )
    sources: dict[str, dict[str, str]] = {}
    for row in rows:
        instances = [
            obj for obj in object_rows
            if obj[IMAGE_ID_FIELD] == row[IMAGE_ID_FIELD]
        ]
        if len(instances) == 1:
            sources[row[IMAGE_ID_FIELD]] = {
                **{key: str(value) for key, value in instances[0].items()}, **row
            }
    review = ReviewLog(args.review_log)
    for row in rows:
        if (
            not review.completed(row[IMAGE_ID_FIELD])
            and row[IMAGE_ID_FIELD] not in sources
        ):
            raise ValueError(
                f"{row['Filepath']} requires exactly one declared instance for this "
                "annotation workflow. Run basic.py to add instance metadata."
            )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backend = Sam3ImageBackend(
        args.model,
        device,
        args.threshold,
        refine_crop=args.refine_crop,
        single_tool=args.single_tool,
    )
    root = manifest.parent
    aliases = load_taxonomy_aliases(args.aliases_json)
    missing = [
        row["Filepath"] for row in rows if not (root / row["Filepath"]).is_file()
    ]
    if missing:
        print(
            f"Skipping {len(missing)} manifest images that are not present below {root}."
        )

    queue = [
        row for row in rows
        if not review.completed(row[IMAGE_ID_FIELD])
        and (root / row["Filepath"]).is_file()
    ]
    queue.sort(
        key=lambda row: (
            review.decisions.get(row[IMAGE_ID_FIELD], {}).get("decision") == "skipped"
        ) != args.skipped_first
    )
    for offset in range(0, len(queue), args.batch_size):
        pending = queue[offset : offset + args.batch_size]
        images = [Image.open(root / row["Filepath"]).convert("RGB") for row in pending]
        prompts = [
            image_prompt(sources[row[IMAGE_ID_FIELD]], args.image_prompt, aliases)
            for row in pending
        ]
        predicted = backend.predict(images, prompts)
        discovery_prompts = tuple(
            normalize_tool_name(prompt, aliases)
            for prompt in tuple(args.discovery_prompt or ())
            + tuple(args.canonical_prompt or ())
        )
        for discovery_prompt in discovery_prompts:
            discovered = backend.predict(images, [discovery_prompt] * len(images))
            for image_index, candidates in enumerate(discovered):
                predicted[image_index].extend(
                    Candidate(
                        candidate.mask,
                        candidate.score,
                        candidate.prompt,
                        f"{stable_media_id('prompt', discovery_prompt)}-{candidate.identity}",
                    )
                    for candidate in candidates
                )
        for row, image, initial in zip(pending, images, predicted, strict=True):
            key = row[IMAGE_ID_FIELD]
            active_prompt = prompts[pending.index(row)]
            candidates = deduplicate_candidates(initial)
            while True:
                decision, selected, ignored, replacement_prompt, visibility = (
                    review_candidates(
                        image,
                        candidates,
                        key=key,
                        auto_accept=args.auto_accept,
                        review_log=review,
                    )
                )
                if decision != "reprompt":
                    break
                assert replacement_prompt is not None
                replacement_prompt = normalize_tool_name(replacement_prompt, aliases)
                active_prompt = replacement_prompt
                candidates = deduplicate_candidates(
                    backend.predict([image], [replacement_prompt])[0]
                )
            if decision == "quit":
                return
            if decision == "skipped":
                review.append(key, decision, prompt=active_prompt)
                continue
            if decision == "accepted":
                source = sources[key]
                if len(selected) == 1:
                    selected = [replace(selected[0], identity=source["Instance ID"])]
                mask_path = image_mask_path(args.mask_dir, row["Filepath"])
                height, width = np.asarray(image).shape[:2]
                masks = (
                    np.stack([candidate.mask for candidate in selected])
                    if selected
                    else np.zeros((0, height, width), dtype=np.bool_)
                )
                save_mask_bundle(
                    mask_path,
                    masks,
                    [candidate.identity for candidate in selected],
                    ignore=(
                        np.logical_or.reduce([candidate.mask for candidate in ignored])
                        if ignored
                        else None
                    ),
                )
                normalized_row = {
                    **source,
                    "Tool Name": normalize_tool_name(source["Tool Name"], aliases),
                }
                replacements = _image_object_rows(
                    normalized_row,
                    selected,
                    mask_path,
                    active_prompt,
                )
                if not replacements:
                    # An accepted empty mask must not erase the capture's identity.
                    replacements = [{
                        field: source.get(field, "") if field in {
                            IMAGE_ID_FIELD, "Instance ID", "Tool ID", "Tool Class",
                            "Tool Name", "Confidence",
                        } else ""
                        for field in IMAGE_OBJECT_FIELDS
                    }]
                object_rows = _replace_scope(
                    object_rows,
                    replacements,
                    scope={IMAGE_ID_FIELD: key},
                )
                write_image_manifests(
                    manifest, rows, args.object_manifest, object_rows
                )
            review.append(key, decision, prompt=active_prompt)


def populate_video_manifest(args: argparse.Namespace) -> Path:
    existing = read_csv(args.manifest) if args.manifest.is_file() else []
    by_path = {row["Filepath"]: row for row in existing if row.get("Filepath")}
    for path in sorted(args.video_dir.glob("*.mp4")):
        relative = path.relative_to(args.manifest.parent).as_posix()
        if relative in by_path:
            continue
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not inspect video {path}.")
        try:
            by_path[relative] = {
                "Video ID": stable_media_id("video", relative),
                "Filepath": relative,
                "FPS": capture.get(cv2.CAP_PROP_FPS),
                "Frame Count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                "Width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "Height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                CAPTURE_SESSION_FIELD: args.capture_session_id,
                "Condition": args.condition,
            }
        finally:
            capture.release()
    return write_csv(args.manifest, by_path.values(), VIDEO_MANIFEST_FIELDS)


def _load_video_segment(
    path: Path, start: int, stop: int | None
) -> tuple[list[Image.Image], list[int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video {path}.")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[Image.Image] = []
    indices: list[int] = []
    index = start
    try:
        while stop is None or index < stop:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            indices.append(index)
            index += 1
    finally:
        capture.release()
    return frames, indices


class Sam3VideoBackend:
    def __init__(self, model_name: str, device: torch.device) -> None:
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self.model = Sam3VideoModel.from_pretrained(model_name).to(device).eval()
        self.processor = Sam3VideoProcessor.from_pretrained(model_name)
        self.device = device

    @torch.inference_mode()
    def predict(
        self, frames: Sequence[Image.Image], prompt: str
    ) -> dict[int, list[Candidate]]:
        session = self.processor.init_video_session(
            video=list(frames),
            inference_device=self.device,
            processing_device=self.device,
            video_storage_device="cpu",
        )
        session = self.processor.add_text_prompt(inference_session=session, text=prompt)
        results: dict[int, list[Candidate]] = {}
        for output in self.model.propagate_in_video_iterator(
            inference_session=session, show_progress_bar=True
        ):
            processed = self.processor.postprocess_outputs(session, output)
            results[int(output.frame_idx)] = extract_candidates(
                processed,
                prompt=prompt,
                identity_prefix="track",
                keep_empty=True,
            )
        return results


def annotate_videos(args: argparse.Namespace) -> None:
    populate_video_manifest(args)
    manifest_rows = read_csv(args.manifest)
    object_rows: list[dict[str, object]] = (
        list(read_csv(args.object_manifest)) if args.object_manifest.is_file() else []
    )
    review = ReviewLog(args.review_log)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    backend = Sam3VideoBackend(args.model, device)
    aliases = load_taxonomy_aliases(args.aliases_json)
    root = args.manifest.parent
    for video in manifest_rows:
        if args.video_id and video["Video ID"] != args.video_id:
            continue
        path = root / video["Filepath"]
        frames, frame_indices = _load_video_segment(
            path, args.start_frame, args.end_frame
        )
        if not frames:
            raise ValueError(f"Selected video segment is empty: {path}")
        active_prompt = normalize_tool_name(args.prompt, aliases)
        outputs = backend.predict(frames, active_prompt)
        for local_index, (frame, frame_index) in enumerate(
            zip(frames, frame_indices, strict=True)
        ):
            key = f"{video['Video ID']}:{frame_index}"
            if review.completed(key):
                continue
            candidates = deduplicate_candidates(outputs.get(local_index, []))
            while True:
                decision, selected, ignored, replacement, visibility = (
                    review_candidates(
                        frame,
                        candidates,
                        key=key,
                        auto_accept=args.auto_accept,
                        allow_occluded=True,
                        review_log=review,
                    )
                )
                if decision != "reprompt":
                    break
                assert replacement is not None
                active_prompt = normalize_tool_name(replacement, aliases)
                outputs = backend.predict(frames, active_prompt)
                candidates = deduplicate_candidates(outputs.get(local_index, []))
            if decision == "quit":
                return
            if decision == "skipped":
                review.append(key, "skipped", prompt=active_prompt)
                continue
            if decision == "accepted":
                mask_path = args.mask_dir / video["Video ID"] / f"{frame_index}.npz"
                height, width = np.asarray(frame).shape[:2]
                visible_candidates = [
                    candidate
                    for candidate in selected
                    if candidate.visibility != "occluded" and candidate.mask.any()
                ]
                masks = (
                    np.stack([candidate.mask for candidate in visible_candidates])
                    if visible_candidates
                    else np.zeros((0, height, width), dtype=np.bool_)
                )
                save_mask_bundle(
                    mask_path,
                    masks,
                    [candidate.identity for candidate in visible_candidates],
                    ignore=(
                        np.logical_or.reduce([candidate.mask for candidate in ignored])
                        if ignored
                        else None
                    ),
                )
                bundle_indices = {
                    candidate.identity: index
                    for index, candidate in enumerate(visible_candidates)
                }
                replacements = []
                for candidate in selected:
                    box = mask2xywh(candidate.mask)
                    if box is None and candidate.visibility != "occluded":
                        continue
                    if box is None:
                        box = (0, 0, 0, 0)
                    replacements.append(
                        {
                            "Video ID": video["Video ID"],
                            "Frame Index": frame_index,
                            "Track ID": candidate.identity,
                            "Tool ID": args.tool_id if len(selected) == 1 else "",
                            "Tool Class": args.tool_class,
                            "Tool Name": candidate.prompt,
                            "BBox X": box[0],
                            "BBox Y": box[1],
                            "BBox Width": box[2],
                            "BBox Height": box[3],
                            "Mask Path": str(mask_path),
                            "Mask Index": bundle_indices.get(candidate.identity, -1),
                            "Preferred Track": "FALSE",
                            **_visibility_flags(candidate.visibility),
                        }
                    )
                object_rows = _replace_scope(
                    object_rows,
                    replacements,
                    scope={
                        "Video ID": video["Video ID"],
                        "Frame Index": frame_index,
                    },
                )
                write_csv(args.object_manifest, object_rows, VIDEO_OBJECT_FIELDS)
            review.append(key, decision, prompt=active_prompt)


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="facebook/sam3")
    parser.add_argument("--device")
    parser.add_argument("--auto-accept", action="store_true")
    parser.add_argument("--aliases-json", type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and review SAM3 annotations."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    images = commands.add_parser("images")
    _shared(images)
    images.add_argument(
        "--manifest", type=Path, default=Path("image_manifest.csv")
    )
    images.add_argument(
        "--object-manifest",
        type=Path,
        default=Path("image_object_manifest.csv"),
    )
    images.add_argument("--mask-dir", type=Path, default=Path("masks/images"))
    images.add_argument(
        "--review-log", type=Path, default=Path("image_review.jsonl")
    )
    images.add_argument(
        "--skipped-first",
        action="store_true",
        help="Review previously skipped images first (default: skipped images last).",
    )
    images.add_argument("--batch-size", type=int, default=4)
    images.add_argument("--threshold", type=float, default=0.5)
    images.add_argument(
        "--single-tool",
        action="store_true",
        help=(
            "For images containing one tool, merge masks from each prompt into "
            "one object in both inference passes (useful for open scissors)."
        ),
    )
    images.add_argument(
        "--refine-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refine single-tool masks with a second SAM3 pass on a padded crop "
            "(default: enabled). Keep the original prediction if refinement is "
            "ambiguous or touches an interior crop edge."
        ),
    )
    images.add_argument(
        "--image-prompt",
        choices=("tool-class", "name-and-class"),
        default="tool-class",
        help=(
            "Prompt SAM3 with Tool Class (default), or combine Tool Name and "
            "Tool Class."
        ),
    )
    images.add_argument(
        "--discovery-prompt",
        action="append",
        help="Broad prompt to discover additional instruments in multi-tool images.",
    )
    images.add_argument(
        "--canonical-prompt",
        action="append",
        help="Canonical tool-name prompt to add to multi-tool discovery.",
    )

    videos = commands.add_parser("videos")
    _shared(videos)
    videos.add_argument(
        "--manifest", type=Path, default=Path("video_manifest.csv")
    )
    videos.add_argument("--video-dir", type=Path, default=Path("videos"))
    videos.add_argument(
        "--object-manifest",
        type=Path,
        default=Path("video_object_manifest.csv"),
    )
    videos.add_argument("--mask-dir", type=Path, default=Path("masks/videos"))
    videos.add_argument(
        "--review-log", type=Path, default=Path("video_review.jsonl")
    )
    videos.add_argument("--video-id")
    videos.add_argument("--start-frame", type=int, default=0)
    videos.add_argument("--end-frame", type=int)
    videos.add_argument("--prompt", required=True)
    videos.add_argument("--tool-id", default="")
    videos.add_argument("--tool-class", default="")
    videos.add_argument("--capture-session-id", default="")
    videos.add_argument("--condition", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "images":
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive.")
        annotate_images(args)
    else:
        if args.start_frame < 0 or (
            args.end_frame is not None and args.end_frame <= args.start_frame
        ):
            raise ValueError("Invalid video frame range.")
        annotate_videos(args)


if __name__ == "__main__":
    main()
