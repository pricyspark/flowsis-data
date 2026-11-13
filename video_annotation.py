"""Frame review, bounded correction propagation, and the video CLI."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

from annotations import (
    atomic_write_text,
    canonical_tool_label,
    load_taxonomy_aliases,
    normalize_tool_name,
    read_csv,
    write_csv,
)
from image_review import Candidate
from image_refinement import refine_tool, tool_crop
from masks import load_mask_bundle, save_mask_bundle
from video_backend import Sam3VideoBackend
from video_data import (
    FRAME_FIELDS,
    Mask,
    VideoLog,
    file_hash,
    prepare_videos,
    read_frame,
    read_mask,
    resolve_path,
    save_frame_mask,
    select_videos,
    validate_video_frames,
)
from video_ui import (
    collect_clicks,
    render_preview,
    save_contact_sheet,
    select_region,
    show_frame,
)


def frame_ranges(indices: set[int]) -> str:
    """Format sorted, inclusive runs without implying that gaps were processed."""
    if not indices:
        return "none"
    ordered = sorted(indices)
    first = last = ordered[0]
    ranges = []
    for index in ordered[1:]:
        if index == last + 1:
            last = index
            continue
        ranges.append(str(first) if first == last else f"{first}–{last}")
        first = last = index
    ranges.append(str(first) if first == last else f"{first}–{last}")
    return ", ".join(ranges)


def merge_region(
    original: Mask, replacement: Mask, region: tuple[int, int, int, int] | None
) -> Mask:
    if replacement.shape != original.shape:
        raise ValueError("Replacement mask dimensions differ from the frame.")
    if region is None:
        return replacement.copy()
    left, top, right, bottom = region
    if not (
        0 <= left < right <= original.shape[1]
        and 0 <= top < bottom <= original.shape[0]
    ):
        raise ValueError("Region must be inside the frame.")
    result = original.copy()
    result[top:bottom, left:right] = replacement[top:bottom, left:right]
    return result


class VideoReviewer:
    def __init__(
        self,
        args: argparse.Namespace,
        video: dict[str, str],
        all_frames: list[dict[str, str]],
        backend: Sam3VideoBackend,
    ):
        if args.chunk_size < 2 or args.context_frames < 0 or args.start_frame < 0:
            raise ValueError(
                "chunk-size must be at least 2; context/start frame must be nonnegative."
            )
        if not 64 <= args.preview_size <= 2048:
            raise ValueError("preview-size must be between 64 and 2048 pixels.")
        self.args = args
        self.video = video
        self.root = args.manifest.parent.resolve()
        self.all_frames = all_frames
        self.frames = sorted(
            (row for row in all_frames if row["Video ID"] == video["Video ID"]),
            key=lambda row: int(row["Frame Index"]),
        )
        validate_video_frames(self.root, video, self.frames, check_files=False)
        self.backend = backend
        self.log = VideoLog(args.review_log)
        self.aliases = load_taxonomy_aliases(args.aliases_json)
        self.inspected: dict[int, str] = {}
        self.zoom = True
        self.stay = args.stay_on_video or bool(args.video_id) or args.video is not None
        self.click_history: dict[
            tuple[int, tuple[int, int, int, int] | None], list[tuple[float, float, int]]
        ] = {}

    def flush(self) -> None:
        write_csv(self.args.frame_manifest, self.all_frames, FRAME_FIELDS)

    def is_finished(self) -> bool:
        if any(
            row.get("Review Status") not in {"accepted", "rejected"}
            for row in self.frames
        ):
            return False
        return all(
            row.get("Review Status") == "rejected"
            or self.log.acceptance_matches(self.root, self.video, row)
            for row in self.frames
        )

    def editable(self, indices: list[int]) -> None:
        protected = [
            index
            for index in indices
            if self.frames[index].get("Review Status") in {"accepted", "rejected"}
        ]
        if protected:
            raise ValueError(
                f"Frames {protected[:10]} are reviewed. Explicitly reopen the desired range first."
            )

    def reopen(self, start: int, stop: int) -> None:
        self.check_range(start, stop)
        for row in self.frames[start : stop + 1]:
            row.update(
                {
                    "Review Status": "pending",
                    "Reviewed": "FALSE",
                    "Annotation Complete": "FALSE",
                    "Mask Valid": "FALSE",
                    "Class Valid": "FALSE",
                    "Base Valid": "FALSE",
                    "Annotation Confidence": "",
                }
            )
            self.log.append(row["Frame ID"], "pending", action="reopen")
        self.flush()

    def remove_anchor(self, index: int) -> None:
        self.check_range(index, index)
        row = self.frames[index]
        path = row.get("Anchor Path")
        if not path:
            print(f"Frame {index} has no saved anchor.")
            return
        resolve_path(self.root, path).unlink(missing_ok=True)
        row["Anchor Path"] = ""
        self.flush()
        self.log.append(
            row["Frame ID"],
            row.get("Review Status") or "pending",
            action="remove_anchor",
            anchor_path=path,
            extraction_id=self.video["Extraction ID"],
        )
        print(f"Removed anchor at frame {index}; its mask and review status are unchanged.")

    def set_anchor(self, index: int) -> None:
        self.check_range(index, index)
        row = self.frames[index]
        mask = self.mask(index)
        if not mask.any():
            raise ValueError(f"Frame {index} needs a nonempty mask before becoming an anchor.")
        path = (
            self.args.mask_dir
            / "anchors"
            / self.video["Video ID"]
            / f"{index:06d}.npz"
        )
        save_mask_bundle(path, mask[None], [f"tool-{self.video['Tool ID']}"])
        row["Anchor Path"] = path.resolve().relative_to(self.root).as_posix()
        self.flush()
        self.log.append(
            row["Frame ID"],
            row.get("Review Status") or "pending",
            action="set_anchor",
            anchor_path=row["Anchor Path"],
            extraction_id=self.video["Extraction ID"],
        )
        print(f"Set anchor at frame {index}; its mask and review status are unchanged.")

    def check_range(self, start: int, stop: int) -> None:
        if not 0 <= start <= stop < len(self.frames):
            raise ValueError(
                f"Use an inclusive frame range within 0..{len(self.frames)-1}."
            )

    def mask(self, index: int) -> Mask:
        return read_mask(
            self.root,
            self.frames[index],
            (int(self.video["Height"]), int(self.video["Width"])),
        )

    def save(
        self,
        index: int,
        mask: Mask,
        decision: str,
        *,
        visibility: str = "",
        anchor: bool = False,
        action: str = "review",
        **details: object,
    ) -> None:
        self.editable([index])
        row = self.frames[index]
        save_frame_mask(
            self.root,
            row,
            self.video,
            self.args.mask_dir,
            mask,
            decision=decision,
            visibility=visibility,
            anchor=anchor,
        )
        self.flush()
        self.log.append(
            row["Frame ID"],
            decision,
            action=action,
            extraction_id=self.video["Extraction ID"],
            visibility=visibility,
            mask_path=row["Mask Path"],
            mask_sha256=(
                file_hash(resolve_path(self.root, row["Mask Path"]))
                if decision == "accepted"
                else ""
            ),
            frame_sha256=row["Frame SHA256"],
            **details,
        )

    def preview(self, start: int, stop: int) -> Path:
        self.check_range(start, stop)
        self.backend.release()
        directory = (
            self.args.preview_dir
            / self.video["Video ID"]
            / f"{start:06d}-{stop:06d}-{uuid4().hex[:8]}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        pages = 0
        for first in range(start, stop + 1, 12):
            indices = list(range(first, min(stop + 1, first + 12)))
            items: list[tuple[Image.Image, str]] = []
            inspected: dict[int, str] = {}
            page = f"{indices[0]:06d}-{indices[-1]:06d}.png"
            for index in indices:
                row = self.frames[index]
                path = row.get("Mask Path")
                digest = file_hash(resolve_path(self.root, path)) if path else ""
                mask = self.mask(index)
                has_pixels = bool(mask.any())
                with read_frame(self.root, row) as image:
                    bounds = (
                        tool_crop(mask, padding=0.4)
                        if self.zoom and mask.any()
                        else (0, 0, image.width, image.height)
                    )
                    thumbnail = render_preview(
                        image, mask, bounds, self.args.preview_size
                    )
                del mask
                items.append(
                    (
                        thumbnail,
                        f"Frame {index} | {float(row['Timestamp Seconds']):.3f}s | "
                        f"{row.get('Review Status')} | "
                        f"{'mask' if has_pixels else 'EMPTY MASK' if path else 'NO MASK'}",
                    )
                )
                if path:
                    inspected[index] = digest
                records.append(
                    {
                        "frame_index": index,
                        "frame_sha256": row["Frame SHA256"],
                        "mask_path": path or "",
                        "mask_sha256": digest,
                        "bounds": bounds,
                        "page": page,
                    }
                )
            try:
                save_contact_sheet(items, directory / page, self.args.preview_size)
            finally:
                for thumbnail, _ in items:
                    thumbnail.close()
                items.clear()
            atomic_write_text(
                directory / "manifest.json",
                json.dumps(
                    {
                        "video_id": self.video["Video ID"],
                        "extraction_id": self.video["Extraction ID"],
                        "preview_size": self.args.preview_size,
                        "frames": records,
                    },
                    indent=2,
                ),
            )
            self.inspected.update(inspected)
            pages += 1
            print(f"Saved preview sheet {page}.")
        print(f"Saved {pages} contact sheets to {directory.resolve()}.")
        print("Inspect the saved sheets before accepting masks.")
        return directory

    def accept_range(self, start: int, stop: int) -> None:
        self.check_range(start, stop)
        skipped = {
            index
            for index in range(start, stop + 1)
            if self.frames[index].get("Review Status") == "accepted"
            and self.log.acceptance_matches(self.root, self.video, self.frames[index])
        }
        indices = [index for index in range(start, stop + 1) if index not in skipped]
        reaccept = {
            index
            for index in indices
            if self.frames[index].get("Review Status") == "accepted"
        }
        self.editable([index for index in indices if index not in reaccept])
        for index in indices:
            row = self.frames[index]
            path = row.get("Mask Path")
            if not path or self.inspected.get(index) != file_hash(
                resolve_path(self.root, path)
            ):
                raise ValueError(
                    f"Preview the current mask for frame {index} before range acceptance."
                )
            if not self.mask(index).any():
                raise ValueError(
                    f"Frame {index} has no visible mask; resolve absent/occluded explicitly."
                )
        for index in indices:
            if index in reaccept:
                self.reopen(index, index)
            self.save(
                index,
                self.mask(index),
                "accepted",
                visibility=self.frames[index].get("Visibility") or "clear",
            )
        print(f"Newly accepted frames: {frame_ranges(set(indices) - reaccept)}.")
        if reaccept:
            print(f"Reaccepted changed or unverified frames: {frame_ranges(reaccept)}.")
        if skipped:
            print(f"Already accepted frames preserved: {frame_ranges(skipped)}.")

    def accept_masked(self) -> None:
        """Explicitly approve all unreviewed visible masks without requiring previews."""
        count = 0
        reaccepted = 0
        for index, row in enumerate(self.frames):
            if (
                row.get("Review Status") == "rejected"
                or not row.get("Mask Path")
            ):
                continue
            accepted = row.get("Review Status") == "accepted"
            if accepted and self.log.acceptance_matches(self.root, self.video, row):
                continue
            mask = self.mask(index)
            if not mask.any():
                continue
            if accepted:
                self.reopen(index, index)
                reaccepted += 1
            self.save(
                index,
                mask,
                "accepted",
                visibility=row.get("Visibility") or "clear",
                action="accept_all_masked",
            )
            count += 1
        print(f"Accepted {count} masked frames in the current video.")
        if reaccepted:
            print(f"Reaccepted {reaccepted} changed or unverified frames.")

    def propagate(
        self,
        anchor_index: int,
        start: int,
        stop: int,
        direction: str,
        *,
        preserve_start: bool = False,
        seed_mask: Mask | None = None,
        on_prediction: Callable[[int], None] | None = None,
    ) -> None:
        self.check_range(start, stop)
        if not start <= anchor_index <= stop or direction not in {
            "forward",
            "backward",
            "both",
        }:
            raise ValueError(
                "Select forward/backward/both and a range containing the current "
                f"frame ({anchor_index}). Use jump FRAME to choose the starting mask."
            )
        target_start = anchor_index if direction == "forward" else start
        target_stop = anchor_index if direction == "backward" else stop
        accepted_anchor = self.frames[anchor_index].get("Review Status") == "accepted"
        self.editable(
            [
                index
                for index in range(target_start, target_stop + 1)
                if index != anchor_index or not accepted_anchor
            ]
        )
        seed = self.mask(anchor_index) if seed_mask is None else seed_mask
        if seed.shape != (int(self.video["Height"]), int(self.video["Width"])):
            raise ValueError("Starting anchor must match the full-frame dimensions.")
        if not seed.any():
            raise ValueError(
                f"Frame {anchor_index} has no visible mask to use as an anchor. "
                "Jump to a masked frame or create a mask here before propagation."
            )
        existing = {
            index
            for index in range(target_start, target_stop + 1)
            if self.frames[index].get("Mask Path")
        }
        reused = {anchor_index} | {
            index
            for index in range(target_start, target_stop + 1)
            if self.frames[index].get("Anchor Path")
        }
        # Persist the explicitly selected anchor independently of pending propagation output.
        if not accepted_anchor and not preserve_start:
            self.save(anchor_index, seed, "pending", anchor=True, action="anchor")
        written: set[int] = {anchor_index} if accepted_anchor or preserve_start else set()
        for reverse in (
            [False, True] if direction == "both" else [direction == "backward"]
        ):
            current, carry = anchor_index, seed
            terminal = target_start if reverse else target_stop
            while True:
                first = (
                    max(terminal, current - self.args.chunk_size + 1)
                    if reverse
                    else current
                )
                last = (
                    current
                    if reverse
                    else min(terminal, current + self.args.chunk_size - 1)
                )
                window_start = max(0, first - self.args.context_frames)
                window_stop = min(len(self.frames) - 1, last + self.args.context_frames)
                images = [
                    read_frame(self.root, row)
                    for row in self.frames[window_start : window_stop + 1]
                ]
                anchors = {}
                for index in range(window_start, window_stop + 1):
                    path = self.frames[index].get("Anchor Path")
                    if path and self.frames[index].get("Review Status") != "rejected":
                        bundle = load_mask_bundle(resolve_path(self.root, path))
                        if bundle.masks.shape != (1, images[0].height, images[0].width):
                            raise ValueError(
                                "Correction anchor has invalid mask dimensions."
                            )
                        anchors[index - window_start] = bundle.masks[0]
                anchors[current - window_start] = carry
                next_index = first if reverse else last
                seen = {current}
                boundary = carry if next_index == current else None
                for local_index, prediction in self.backend.track(
                    images,
                    anchors,
                    current - window_start,
                    next_index - window_start,
                    reverse,
                ):
                    index = local_index + window_start
                    if not first <= index <= last:
                        continue
                    seen.add(index)
                    if index == next_index:
                        boundary = prediction
                    if index not in written:
                        self.save(
                            index,
                            prediction,
                            "pending",
                            action="propagate",
                            anchor_frame=anchor_index,
                            frame_range=[target_start, target_stop],
                            direction=direction,
                        )
                        written.add(index)
                        if on_prediction is not None:
                            on_prediction(index)
                # Save each prediction immediately; do not retain a chunk of
                # full-resolution masks or lose it if inference is interrupted.
                del images, anchors
                missing = set(range(first, last + 1)) - seen
                if missing:
                    raise ValueError(
                        f"Tracker did not return frame {min(missing)}; correct it before continuing."
                    )
                if next_index == terminal:
                    break
                if boundary is None or not boundary.any():
                    raise ValueError(
                        f"Tracking lost the tool at frame {next_index}. Add a correction there."
                    )
                carry = boundary
                current = next_index
        predicted = written - reused
        print(f"Propagation completed: frames {target_start}–{target_stop} ({direction}).")
        print(f"New pending masks: {frame_ranges(predicted - existing)}.")
        print(f"Regenerated pending masks: {frame_ranges(predicted & existing)}.")
        print(f"Reused anchor masks: {frame_ranges(reused)}.")
        if accepted_anchor:
            print(f"Preserved accepted starting mask: {anchor_index} (unchanged).")
        print("Review pending masks before acceptance.")
        if self.args.auto_preview:
            self.preview(target_start, target_stop)

    def run(self) -> bool:
        start = self.args.start_frame
        stop = (
            self.args.end_frame
            if self.args.end_frame is not None
            else len(self.frames) - 1
        )
        self.check_range(start, stop)
        current = next(
            (
                index
                for index in range(start, stop + 1)
                if self.frames[index].get("Review Status")
                not in {"accepted", "rejected"}
            ),
            None,
        )
        if current is None:
            print(f"{self.video['Filepath']}: selected frames are already reviewed.")
            if self.args.auto_accept:
                return True
            current = start
        prompt = self.args.prompt or canonical_tool_label(
            self.video["Tool Name"], self.video["Tool Class"], self.aliases
        )
        region: tuple[int, int, int, int] | None = None
        candidates: list[Candidate] = []
        visited: set[int] = set()
        if not self.args.auto_accept and (self.stay or not self.is_finished()):
            print(
                "Commands: p, g, click, prop, preview, accept, jump, q. "
                "See README.md for details."
            )
        while current is not None:
            if not self.stay and self.is_finished():
                self.backend.release()
                print(f"Finished video {self.video['Filepath']}; moving to the next video.")
                return True
            row = self.frames[current]
            image = read_frame(self.root, row)
            mask = self.mask(current)
            if self.args.auto_accept:
                if not mask.any():
                    raise ValueError(
                        "--auto-accept requires existing nonempty predictions; missing masks are unresolved."
                    )
                self.save(
                    current, mask, "accepted", visibility="clear", action="auto_accept"
                )
                command, argument = "next", ""
            else:
                status = row.get("Review Status") or "pending"
                if status == "accepted" and not self.log.acceptance_matches(
                    self.root, self.video, row
                ):
                    status += " (changed/unverified)"
                show_frame(
                    image,
                    mask,
                    f"{self.video['Tool Name']} ({self.video['Tool ID']}) | "
                    f"frame {current} @ {row['Timestamp Seconds']}s | {status} | "
                    f"anchor {'yes' if row.get('Anchor Path') else 'no'} | region {region}",
                    candidates,
                    self.zoom,
                )
                if row.get("Mask Path"):
                    self.inspected[current] = file_hash(
                        resolve_path(self.root, row["Mask Path"])
                    )
                command, _, argument = input("video> ").strip().partition(" ")
            try:
                if command == "q":
                    return False
                if command == "next-video":
                    self.backend.release()
                    print(f"Leaving {self.video['Filepath']}; saved masks are retained.")
                    return True
                if command == "stay":
                    if argument not in {"", "on", "off"}:
                        raise ValueError("Use stay, stay on, or stay off.")
                    self.stay = argument != "off"
                    print(f"Stay on the current video: {'on' if self.stay else 'off'}.")
                    continue
                if command == "jump":
                    index = int(argument)
                    self.check_range(index, index)
                    current, region, candidates = index, None, []
                    continue
                if command == "view":
                    if argument not in {"crop", "full"}:
                        raise ValueError("Use view crop or view full.")
                    self.zoom = argument == "crop"
                    continue
                if command == "autopreview":
                    if argument not in {"on", "off"}:
                        raise ValueError("Use autopreview on or autopreview off.")
                    self.args.auto_preview = argument == "on"
                    print(f"Automatic saved previews: {argument}.")
                    continue
                if command in {"preview", "accept"}:
                    if command == "accept" and not argument:
                        self.accept_masked()
                        continue
                    first, last = map(int, argument.split())
                    (self.preview if command == "preview" else self.accept_range)(
                        first, last
                    )
                    continue
                if command == "reopen":
                    first, last = map(int, argument.split())
                    self.reopen(first, last)
                    continue
                if command == "unanchor":
                    self.remove_anchor(int(argument) if argument else current)
                    continue
                if command == "anchor":
                    self.set_anchor(int(argument) if argument else current)
                    continue
                if command == "whole":
                    region = None
                    continue
                if command == "region":
                    if argument:
                        values = [int(value) for value in argument.split()]
                        if len(values) != 4:
                            raise ValueError("Region requires L T R B coordinates.")
                        selected_region = (values[0], values[1], values[2], values[3])
                        merge_region(mask, mask, selected_region)
                        region = selected_region
                    else:
                        region = select_region(image)
                    continue
                if command in {
                    "p",
                    "g",
                    "click",
                    "box",
                    "a",
                    "v",
                    "absent",
                    "occluded",
                    "r",
                }:
                    self.editable([current])
                if command == "p":
                    prompt = normalize_tool_name(argument or prompt, self.aliases)
                    view = image.crop(region) if region else image
                    parts = self.backend.text(view, prompt)
                    candidates = []
                    for part in parts:
                        full = np.zeros((image.height, image.width), dtype=np.bool_)
                        if region:
                            left, top, right, bottom = region
                            full[top:bottom, left:right] = part.mask
                        else:
                            full = part.mask
                        candidates.append(
                            Candidate(full, part.score, prompt, part.identity)
                        )
                    self.log.append(
                        row["Frame ID"],
                        "pending",
                        action="text_prompt",
                        prompt=prompt,
                        region=region,
                    )
                    continue
                if command == "g":
                    selected = sorted({int(value) for value in argument.split(",")})
                    if not selected or any(
                        index < 0 or index >= len(candidates) for index in selected
                    ):
                        raise ValueError("Select valid candidate indices.")
                    replacement = np.logical_or.reduce(
                        [candidates[index].mask for index in selected]
                    )
                    refinement = ""
                    if self.args.refine_crop and replacement.any():
                        view = image.crop(region) if region else image
                        local = (
                            replacement[region[1] : region[3], region[0] : region[2]]
                            if region
                            else replacement
                        )
                        refined, refinement = refine_tool(
                            view,
                            Candidate(local, 1.0, prompt, f"tool-{self.video['Tool ID']}"),
                            prompt,
                            self.backend.text,
                        )
                        print(refinement)
                        if refined is not None:
                            if region:
                                replacement = np.zeros_like(replacement)
                                replacement[
                                    region[1] : region[3], region[0] : region[2]
                                ] = refined.mask
                            else:
                                replacement = refined.mask
                    self.save(
                        current,
                        merge_region(mask, replacement, region),
                        "pending",
                        anchor=True,
                        action="group_fragments",
                        candidate_indices=selected,
                        refinement=refinement,
                        prompt=prompt,
                        region=region,
                    )
                    candidates = []
                    continue
                if command in {"click", "box"}:
                    view = image.crop(region) if region else image
                    view_mask = (
                        mask[region[1] : region[3], region[0] : region[2]]
                        if region
                        else mask
                    )
                    points = (
                        collect_clicks(
                            view,
                            view_mask,
                            self.click_history.get((current, region), []),
                        )
                        if command == "click"
                        else []
                    )
                    box = select_region(view) if command == "box" else None
                    if points is None or (not points and box is None):
                        continue
                    replacement = self.backend.clicks(view, points, box)
                    full = np.zeros_like(mask)
                    if region:
                        full[region[1] : region[3], region[0] : region[2]] = replacement
                    else:
                        full = replacement
                    self.save(
                        current,
                        merge_region(mask, full, region),
                        "pending",
                        anchor=True,
                        action="click_correction",
                        points=points,
                        box=box,
                        region=region,
                    )
                    if command == "click":
                        self.click_history[(current, region)] = points
                    candidates = []
                    continue
                if command in {"prop", "propr"}:
                    parts = argument.split()
                    if len(parts) not in {2, 3} or parts[0] not in {
                        "forward",
                        "backward",
                        "both",
                    }:
                        raise ValueError(
                            "Use prop forward|backward|both COUNT or START END."
                        )
                    direction = parts[0]
                    if len(parts) == 2:
                        count = int(parts[1])
                        if count <= 0:
                            raise ValueError("Propagation frame count must be positive.")
                        first = (
                            max(0, current - count)
                            if direction in {"backward", "both"}
                            else current
                        )
                        last = (
                            min(len(self.frames) - 1, current + count)
                            if direction in {"forward", "both"}
                            else current
                        )
                    else:
                        first, last = map(int, parts[1:])
                    self.propagate(current, first, last, direction)
                    continue
                if command == "v":
                    if argument not in {"clear", "partial", "severe"}:
                        raise ValueError(
                            "Visible state must be clear, partial, or severe."
                        )
                    row["Visibility"] = argument
                    self.flush()
                    continue
                if command in {"a", "absent", "occluded"}:
                    state = row.get("Visibility") or "clear"
                    if command != "a":
                        state = command
                        mask = np.zeros_like(mask)
                    self.save(
                        current,
                        mask,
                        "accepted",
                        visibility=state,
                        anchor=bool(mask.any()),
                    )
                elif command == "r":
                    row.update(
                        {
                            "Review Status": "rejected",
                            "Reviewed": "FALSE",
                            "Annotation Complete": "FALSE",
                        }
                    )
                    self.flush()
                    self.log.append(row["Frame ID"], "rejected", action="reject")
                elif command not in {"next", "s"}:
                    raise ValueError("Unknown command.")
                elif not self.args.auto_accept:
                    self.log.append(
                        row["Frame ID"],
                        row.get("Review Status") or "pending",
                        action="skip",
                    )
                visited.add(current)
                current = next(
                    (
                        index
                        for index in range(start, stop + 1)
                        if index not in visited
                        and self.frames[index].get("Review Status")
                        not in {"accepted", "rejected"}
                    ),
                    None,
                )
                if current is None and self.stay and not self.args.auto_accept:
                    current = start
                region, candidates = None, []
            except (ValueError, OSError) as error:
                print(error)
        if self.is_finished():
            self.backend.release()
            print(f"Finished video {self.video['Filepath']}; moving to the next video.")
        return True


def list_videos(args: argparse.Namespace) -> None:
    videos = read_csv(args.manifest)
    frames = read_csv(args.frame_manifest) if args.frame_manifest.is_file() else []
    accepted = Counter(
        row["Video ID"] for row in frames if row.get("Review Status") == "accepted"
    )
    print("Video  Tool  Accepted/Frames  File")
    for number, video in select_videos(args, videos):
        progress = f"{accepted[video['Video ID']]}/{video.get('Frame Count') or '?'}"
        print(
            f"{number:>5}  {video['Tool ID']:>4}  {progress:>15}  {video['Filepath']}"
        )


def annotate_videos(args: argparse.Namespace) -> None:
    videos = read_csv(args.manifest)
    all_frames = read_csv(args.frame_manifest)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    backend = Sam3VideoBackend(args.model, device, args.threshold)
    for number, video in select_videos(args, videos):
        root = args.manifest.parent.resolve()
        if file_hash(resolve_path(root, video["Filepath"])) != video["Source SHA256"]:
            raise ValueError("Source MP4 changed; reextract before annotating.")
        print(f"Opening {video['Filepath']} (Video {number}, Tool {video['Tool ID']}).")
        if not VideoReviewer(args, video, all_frames, backend).run():
            return
    print("No more selected videos.")


def verify_videos(args: argparse.Namespace) -> None:
    videos = read_csv(args.manifest)
    frames = read_csv(args.frame_manifest)
    known = {video["Video ID"] for video in videos}
    if len(known) != len(videos):
        raise ValueError("Video IDs must be unique.")
    if args.video_id and args.video_id not in known:
        raise ValueError("Unknown Video ID.")
    if any(row["Video ID"] not in known for row in frames):
        raise ValueError("Frame manifest refers to an unknown video.")
    log = VideoLog(args.review_log)
    for _, video in select_videos(args, videos):
        selected = [row for row in frames if row["Video ID"] == video["Video ID"]]
        validate_video_frames(args.manifest.parent.resolve(), video, selected)
        for row in selected:
            if (
                row.get("Review Status") == "accepted"
                and log.decisions.get(row["Frame ID"]) != "accepted"
            ):
                raise ValueError(
                    f"Review log disagrees with accepted frame {row['Frame ID']}."
                )
            if row.get("Review Status") == "accepted":
                log.validate_acceptance(args.manifest.parent.resolve(), video, row)
        print(
            f"Verified {video['Filepath']}: {len(selected)} frame files and their annotations."
        )


def add_video_commands(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    prepare = commands.add_parser(
        "prepare-videos", help="Extract single-tool MP4s into individual PNG frames."
    )
    review = commands.add_parser(
        "videos",
        help="Review frame masks with mouse corrections and bounded propagation.",
    )
    verify = commands.add_parser(
        "verify-videos", help="Validate extracted frames and video annotations."
    )
    export = commands.add_parser(
        "export-video-training",
        help="Export reviewed frames for the existing flowsis image training reader.",
    )
    listing = commands.add_parser(
        "list-videos", help="List permanent video numbers and source paths."
    )
    for parser in (prepare, review, verify, export, listing):
        parser.add_argument("--manifest", type=Path, default=Path("video_manifest.csv"))
        parser.add_argument(
            "--frame-manifest", type=Path, default=Path("video_frame_manifest.csv")
        )
        parser.add_argument(
            "--review-log", type=Path, default=Path("video_review.jsonl")
        )
        selection = parser.add_mutually_exclusive_group()
        selection.add_argument(
            "--video", type=int, help="Permanent video number; see list-videos."
        )
        selection.add_argument(
            "--video-id", help="Existing hash-based video ID (also supported)."
        )
    prepare.add_argument("--video-dir", type=Path, default=Path("videos"))
    prepare.add_argument("--frame-dir", type=Path, default=Path("frames/videos"))
    prepare.add_argument(
        "--image-object-manifest", type=Path, default=Path("image_object_manifest.csv")
    )
    prepare.add_argument(
        "--tool-id",
        help="Explicit known physical tool ID when videos are not in ID subfolders.",
    )
    prepare.add_argument("--capture-session-id", default="")
    prepare.add_argument("--condition", default="")
    prepare.add_argument(
        "--reextract",
        action="store_true",
        help="Invalidate existing masks/review and replace extracted frames.",
    )
    review.add_argument("--model", default="facebook/sam3")
    review.add_argument("--device")
    review.add_argument("--threshold", type=float, default=0.5)
    review.add_argument(
        "--refine-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refine selected text proposals on a padded crop (default: enabled).",
    )
    review.add_argument("--mask-dir", type=Path, default=Path("masks/videos"))
    review.add_argument("--aliases-json", type=Path)
    review.add_argument(
        "--prompt",
        help="Optional text-prompt override; tool identity remains unchanged.",
    )
    review.add_argument("--start-frame", type=int, default=0)
    review.add_argument("--end-frame", type=int, help="Inclusive final frame index.")
    review.add_argument(
        "--stay-on-video",
        action="store_true",
        help=(
            "Keep the reviewer open after completion; "
            "explicit video selection also enables this."
        ),
    )
    review.add_argument("--chunk-size", type=int, default=16)
    review.add_argument("--context-frames", type=int, default=2)
    review.add_argument(
        "--auto-preview",
        action="store_true",
        help="Save contact sheets after propagation without waiting for inspection.",
    )
    review.add_argument("--preview-dir", type=Path, default=Path("previews/videos"))
    review.add_argument(
        "--preview-size",
        type=int,
        default=768,
        help="Maximum thumbnail dimension in saved contact sheets (64–2048 pixels).",
    )
    review.add_argument(
        "--auto-accept",
        action="store_true",
        help="Explicitly accept existing nonempty pending masks without human review.",
    )
    export.add_argument(
        "--replace",
        action="store_true",
        help="Replace a previous training export after annotation changes.",
    )
    export.add_argument("--output-dir", type=Path, default=Path("video_training"))
