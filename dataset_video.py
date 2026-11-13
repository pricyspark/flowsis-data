"""Read reviewed video runs and partition them without copying media assets."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import numpy as np

from annotations import parse_bool, read_csv
from dataset_schema import ImageRecord, ObjectRecord
from masks import load_mask_bundle, mask2xywh
from video_data import VideoLog, accepted_frame_run, resolve_path


def build_video_rows(root: Path) -> tuple[list[ImageRecord], dict[str, int]]:
    manifest = root / "video_manifest.csv"
    if not manifest.exists():
        return [], {}
    videos = read_csv(manifest)
    if not videos:
        return [], {}
    frames = read_csv(root / "video_frame_manifest.csv")
    log = VideoLog(root / "video_review.jsonl")
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for frame in frames:
        by_video[frame["Video ID"]].append(frame)
    video_ids = [video["Video ID"] for video in videos]
    if len(set(video_ids)) != len(video_ids) or set(by_video) - set(video_ids):
        raise ValueError(
            "Video/frame manifests have duplicate or unknown video identities."
        )
    rows: list[ImageRecord] = []
    skipped: Counter[str] = Counter()
    for video in videos:
        video_id = video["Video ID"]
        print(f"Validating accepted frame run for {video_id}...", flush=True)
        selected = accepted_frame_run(root, video, by_video[video_id], log)
        skipped["video_outside_first_accepted_run"] += len(by_video[video_id]) - len(
            selected
        )
        for frame in selected:
            mask_path = resolve_path(root, frame["Mask Path"])
            bundle = load_mask_bundle(mask_path)
            visibility = frame["Visibility"]
            objects: list[ObjectRecord] = []
            if visibility != "absent":
                positive = bool(len(bundle.masks))
                box = mask2xywh(bundle.masks[0]) if positive else None
                confidence = float(frame["Annotation Confidence"])
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError(
                        f"Invalid annotation confidence: {frame['Frame ID']}"
                    )
                objects.append(
                    ObjectRecord(
                        instance_id=f"tool-{video['Tool ID']}",
                        track_id=f"tool-{video['Tool ID']}",
                        area=float(np.count_nonzero(bundle.masks)) if positive else 0.0,
                        bbox=box or [0, 0, 0, 0],
                        category=-1,
                        tool_name=video["Tool Name"],
                        tool_class=video["Tool Class"],
                        tool_id=video["Tool ID"],
                        mask_path=str(mask_path),
                        mask_index=int(frame["Mask Index"]),
                        annotation_confidence=confidence,
                        visibility=visibility,
                        mask_valid=parse_bool(frame["Mask Valid"]),
                        class_valid=parse_bool(frame["Class Valid"]),
                        base_valid=parse_bool(frame["Base Valid"]),
                        visible=parse_bool(frame["Visible"]),
                        preferred_track=False,
                    )
                )
            rows.append(
                ImageRecord(
                    image_path=str(resolve_path(root, frame["Filepath"])),
                    video_path="",  # Saved PNG is authoritative; do not decode the MP4.
                    frame_index=int(frame["Frame Index"]),
                    image_id=frame["Frame ID"],
                    height=int(video["Height"]),
                    width=int(video["Width"]),
                    tool_ids=[video["Tool ID"]],
                    sample_id="",
                    capture_session_id=video["Capture Session ID"],
                    source_type="video",
                    video_id=video_id,
                    sequence_id=video_id,
                    timestamp_seconds=float(frame["Timestamp Seconds"]),
                    mask_bundle_path=str(mask_path),
                    base_eligible=(
                        visibility in {"clear", "partial", "absent"}
                        and not (bundle.ignore is not None and bundle.ignore.any())
                    ),
                    objects=objects,
                )
            )
    return rows, dict(skipped)


def interval_sizes(available: int, sizes: dict[str, float]) -> dict[str, int]:
    """Round interval lengths while preserving the total available frame count."""
    counts = {
        split: max(0, math.floor(available * size))
        for split, size in sizes.items()
        if size > 0
    }
    remainder = max(0, available - sum(counts.values()))
    for split in sorted(
        counts, key=lambda split: available * sizes[split] - counts[split], reverse=True
    )[:remainder]:
        counts[split] += 1
    return counts


def split_video_sections(
    rows: list[ImageRecord],
    sizes: dict[str, float],
    seed: int,
    *,
    sections: int,
    gap: int,
    min_frames: int,
    short_video_policy: Literal["train", "error"] = "error",
) -> tuple[list[ImageRecord], list[str], list[dict[str, str | int]]]:
    if sections < 1 or gap < 0 or min_frames < 2:
        raise ValueError(
            "Sections must be positive, gap nonnegative, and intervals >= 2 frames."
        )
    if any(
        not math.isfinite(size) or size < 0 for size in sizes.values()
    ) or not math.isclose(sum(sizes.values()), 1):
        raise ValueError("Split sizes must be nonnegative and sum to one.")
    active = [split for split, size in sizes.items() if size > 0]
    by_video: dict[str, list[ImageRecord]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    result, assignments, intervals = [], [], []
    generator = random.Random(seed)
    for video_id, frames in sorted(by_video.items()):
        frames.sort(key=lambda row: row["frame_index"])
        if any(
            right["frame_index"] != left["frame_index"] + 1
            for left, right in zip(frames, frames[1:])
        ):
            raise ValueError(f"Section splitting requires a contiguous run: {video_id}")
        smallest = len(frames) // sections - gap * (len(active) - 1)
        if short_video_policy == "train" and any(
            count < min_frames for count in interval_sizes(smallest, sizes).values()
        ):
            sequence_id = f"{video_id}:train-only"
            for row in frames:
                row["sequence_id"] = sequence_id
            result.extend(frames)
            assignments.extend(["train"] * len(frames))
            intervals.append(
                {
                    "video_id": video_id,
                    "sequence_id": sequence_id,
                    "split": "train",
                    "start_frame": frames[0]["frame_index"],
                    "end_frame": frames[-1]["frame_index"],
                    "frames": len(frames),
                    "reason": "too_short_for_section_holdouts",
                }
            )
            continue
        initial_rotation = generator.randrange(len(active))
        for section in range(sections):
            start, stop = (
                len(frames) * section // sections,
                len(frames) * (section + 1) // sections,
            )
            available = stop - start - gap * (len(active) - 1)
            counts = interval_sizes(available, sizes)
            if any(count < min_frames for count in counts.values()):
                raise ValueError(
                    f"Video {video_id} section {section + 1} cannot provide {min_frames} frames per split "
                    f"after gaps (counts: {counts}). Reduce --video-sections, --split-gap, "
                    "or --min-interval-frames, or use whole-video grouping."
                )
            rotation = (initial_rotation + section) % len(active)
            order = active[rotation:] + active[:rotation]
            cursor = start
            for position, split in enumerate(order):
                interval = frames[cursor : cursor + counts[split]]
                sequence_id = f"{video_id}:section-{section}:{split}"
                for row in interval:
                    row["sequence_id"] = sequence_id
                result.extend(interval)
                assignments.extend([split] * len(interval))
                intervals.append(
                    {
                        "video_id": video_id,
                        "sequence_id": sequence_id,
                        "split": split,
                        "start_frame": interval[0]["frame_index"],
                        "end_frame": interval[-1]["frame_index"],
                        "frames": len(interval),
                    }
                )
                cursor += counts[split] + (gap if position < len(order) - 1 else 0)
    return result, assignments, intervals


def snippet_counts(
    rows: list[ImageRecord], lengths: tuple[int, ...], strides: tuple[int, ...]
) -> dict[str, int]:
    """Count windows fully inside each interval, including a positive visible mask."""
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for row in rows:
        if row["video_id"]:
            groups[row["sequence_id"]].append(row)
    counts = {
        f"length={length},stride={stride}": 0
        for length in lengths
        for stride in strides
    }
    for frames in groups.values():
        indexed = {row["frame_index"]: row for row in frames}
        for length in lengths:
            for stride in strides:
                span = (length - 1) * stride + 1
                count = sum(
                    all(index in indexed for index in range(start, start + span))
                    and any(
                        any(obj["mask_valid"] for obj in indexed[index]["objects"])
                        for index in range(start, start + span, stride)
                    )
                    for start in indexed
                )
                counts[f"length={length},stride={stride}"] += count
    return counts
