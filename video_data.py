"""Single-tool video intake, timestamp-preserving extraction, and frame storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TypeAlias
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from annotations import (
    CAPTURE_SESSION_FIELD,
    VIDEO_MANIFEST_FIELDS as VIDEO_FIELDS,
    VIDEO_FRAME_FIELDS as FRAME_FIELDS,
    parse_bool,
    read_csv,
    stable_media_id,
    tool_metadata,
    write_csv,
)
from masks import load_mask_bundle, mask2xywh, save_mask_bundle

Mask: TypeAlias = NDArray[np.bool_]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a relative path inside the repository: {value}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves the repository: {value}")
    return resolved


class VideoLog:
    def __init__(self, path: Path):
        self.path = path
        self.decisions: dict[str, str] = {}
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.decisions[str(record["key"])] = str(record["decision"])

    def append(self, key: str, decision: str, **details: object) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"key": key, "decision": decision, **details}) + "\n"
            )
        self.decisions[key] = decision


def prepare_videos(args: argparse.Namespace) -> None:
    import av

    root = args.manifest.parent.resolve()
    videos = read_csv(args.manifest) if args.manifest.is_file() else []
    frames = read_csv(args.frame_manifest) if args.frame_manifest.is_file() else []
    log = VideoLog(args.review_log)
    paths = sorted(
        path
        for path in args.video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not paths:
        raise ValueError(f"No MP4 videos found under {args.video_dir}.")
    for source in paths:
        relative = source.resolve().relative_to(root).as_posix()
        existing = next((row for row in videos if row["Filepath"] == relative), None)
        video_id = (
            existing["Video ID"] if existing else stable_media_id("video", relative)
        )
        if args.video_id and video_id != args.video_id:
            continue
        parts = source.relative_to(args.video_dir).parts
        tool_id = args.tool_id or (parts[0] if len(parts) > 1 else "")
        if not tool_id:
            raise ValueError("Put videos under videos/<Tool ID>/ or supply --tool-id.")
        metadata = tool_metadata(args.image_object_manifest, tool_id)
        if existing and any(
            existing.get(field) != metadata[field]
            for field in ("Tool ID", "Tool Class", "Tool Name")
        ):
            raise ValueError(
                f"{relative}: video identity disagrees with image metadata."
            )
        digest = file_hash(source)
        if existing and existing.get("Source SHA256") != digest and not args.reextract:
            raise ValueError(
                f"{relative} changed. Use --reextract to invalidate annotations and extract again."
            )
        video = existing or dict.fromkeys(VIDEO_FIELDS, "")
        if not existing:
            video.update(
                {
                    **metadata,
                    "Video ID": video_id,
                    "Filepath": relative,
                    CAPTURE_SESSION_FIELD: args.capture_session_id
                    or f"video-session-{video_id}",
                    "Condition": args.condition,
                }
            )
            videos.append(video)
        video["Manufacturer"] = metadata["Manufacturer"]
        old_frames = [row for row in frames if row["Video ID"] == video_id]
        previous_filepaths = {row["Filepath"] for row in old_frames}
        if args.reextract:
            for row in old_frames:
                log.append(row["Frame ID"], "pending", action="reextract")
                for field in ("Mask Path", "Anchor Path"):
                    if row.get(field):
                        resolve_path(root, row[field]).unlink(missing_ok=True)
            frames = [row for row in frames if row["Video ID"] != video_id]
            old_frames = []
        video.update(
            {
                "Source SHA256": digest,
                "Extraction ID": (
                    uuid4().hex
                    if args.reextract
                    else video.get("Extraction ID") or uuid4().hex
                ),
                "Extraction Complete": "FALSE",
            }
        )
        write_csv(args.manifest, videos, VIDEO_FIELDS)
        write_csv(args.frame_manifest, frames, FRAME_FIELDS)
        by_index = {int(row["Frame Index"]): row for row in old_frames}
        frame_dir = args.frame_dir / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        count, previous_time = 0, -math.inf
        with av.open(str(source)) as container:
            if len(container.streams.video) != 1:
                raise ValueError(f"Expected exactly one video stream: {relative}")
            stream = container.streams.video[0]
            reported_count = stream.frames
            video["FPS"] = (
                str(float(stream.average_rate)) if stream.average_rate else ""
            )
            for index, decoded in enumerate(container.decode(stream)):
                timestamp = decoded.time
                if (
                    timestamp is None
                    or not math.isfinite(timestamp)
                    or timestamp < previous_time
                    or decoded.is_corrupt
                ):
                    raise ValueError(
                        f"Invalid timestamp or corrupt frame {index} in {relative}."
                    )
                previous_time = timestamp
                image = decoded.to_image().convert("RGB")
                if index == 0:
                    video["Width"], video["Height"] = str(image.width), str(
                        image.height
                    )
                elif image.size != (int(video["Width"]), int(video["Height"])):
                    raise ValueError(
                        "Video frame dimensions changed within the stream."
                    )
                destination = frame_dir / f"{index:06d}.png"
                filepath = destination.resolve().relative_to(root).as_posix()
                row = by_index.get(index)
                if row and (
                    row["Filepath"] != filepath
                    or float(row["Timestamp Seconds"]) != timestamp
                ):
                    raise ValueError("Extraction settings changed; use --reextract.")
                if destination.is_file() and row:
                    if file_hash(destination) != row["Frame SHA256"]:
                        raise ValueError(f"Frame {filepath} changed; use --reextract.")
                else:
                    temporary = destination.with_suffix(".tmp.png")
                    try:
                        image.save(temporary, format="PNG")
                        if row and file_hash(temporary) != row["Frame SHA256"]:
                            raise ValueError(
                                "Recreated frame differs from its original; use --reextract."
                            )
                        temporary.replace(destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                if row is None:
                    row = dict.fromkeys(FRAME_FIELDS, "")
                    row.update(
                        {
                            "Video ID": video_id,
                            "Frame Index": str(index),
                            "Timestamp Seconds": repr(timestamp),
                            "Filepath": filepath,
                            "Frame ID": f"{video_id}:{index}",
                            "Frame SHA256": file_hash(destination),
                            "Review Status": "pending",
                        }
                    )
                    frames.append(row)
                count += 1
                if count % 100 == 0:
                    write_csv(args.frame_manifest, frames, FRAME_FIELDS)
                    print(f"Extracted {count} frames from {relative}.")
        if reported_count and count != reported_count:
            raise ValueError(
                f"Decoded {count} frames, but the MP4 declares {reported_count}; extraction is incomplete."
            )
        if not count:
            raise ValueError(f"Video contains no decoded frames: {relative}")
        if any(index >= count for index in by_index) or file_hash(source) != digest:
            raise ValueError(
                "Video changed or extraction is incomplete; use --reextract."
            )
        if args.reextract:
            current_filepaths = {
                row["Filepath"] for row in frames if row["Video ID"] == video_id
            }
            for filepath in previous_filepaths - current_filepaths:
                path = resolve_path(root, filepath)
                if path.is_relative_to(frame_dir.resolve()):
                    path.unlink(missing_ok=True)
        video.update({"Frame Count": str(count), "Extraction Complete": "TRUE"})
        write_csv(args.frame_manifest, frames, FRAME_FIELDS)
        write_csv(args.manifest, videos, VIDEO_FIELDS)
        print(f"Prepared {relative}: {count} independently loadable PNG frames.")


def read_frame(root: Path, row: dict[str, str]) -> Image.Image:
    path = resolve_path(root, row["Filepath"])
    if file_hash(path) != row["Frame SHA256"]:
        raise ValueError(
            f"Frame {row['Frame ID']} changed; reextract before annotating."
        )
    with Image.open(path) as image:
        return image.convert("RGB")


def read_mask(root: Path, row: dict[str, str], shape: tuple[int, int]) -> Mask:
    if not row.get("Mask Path"):
        return np.zeros(shape, dtype=np.bool_)
    bundle = load_mask_bundle(resolve_path(root, row["Mask Path"]))
    if bundle.masks.shape[1:] != shape or len(bundle.masks) > 1:
        raise ValueError(
            "Single-tool frame bundle has invalid dimensions or mask count."
        )
    return (
        bundle.masks[0].copy() if len(bundle.masks) else np.zeros(shape, dtype=np.bool_)
    )


def save_frame_mask(
    root: Path,
    row: dict[str, str],
    video: dict[str, str],
    mask_dir: Path,
    mask: Mask,
    *,
    decision: str,
    visibility: str = "",
    anchor: bool = False,
) -> None:
    shape = (int(video["Height"]), int(video["Width"]))
    if mask.shape != shape:
        raise ValueError("Mask must match the extracted frame dimensions.")
    if decision == "accepted":
        positive = visibility in {"clear", "partial", "severe"}
        if visibility not in {
            "clear",
            "partial",
            "severe",
            "occluded",
            "absent",
        } or positive != bool(mask.any()):
            raise ValueError(
                "Review a visible mask or explicitly mark the frame absent/occluded."
            )
    path = mask_dir / video["Video ID"] / f"{int(row['Frame Index']):06d}.npz"
    masks = mask[None] if mask.any() else np.zeros((0, *shape), dtype=np.bool_)
    identity = f"tool-{video['Tool ID']}"
    save_mask_bundle(path, masks, [identity] if mask.any() else [])
    box = mask2xywh(mask)
    row.update(
        {
            "Mask Path": path.resolve().relative_to(root).as_posix(),
            "Mask Index": "0" if mask.any() else "-1",
            "Visibility": visibility,
            "Review Status": decision,
            "BBox X": str(box[0]) if box else "",
            "BBox Y": str(box[1]) if box else "",
            "BBox Width": str(box[2]) if box else "",
            "BBox Height": str(box[3]) if box else "",
        }
    )
    reviewed = decision == "accepted"
    class_valid = reviewed and bool(mask.any()) and visibility != "severe"
    row.update(
        {
            "Reviewed": str(reviewed).upper(),
            "Annotation Complete": str(reviewed).upper(),
            "Mask Valid": str(reviewed and bool(mask.any())).upper(),
            "Class Valid": str(class_valid).upper(),
            "Base Valid": str(class_valid).upper(),
            "Visible": str(bool(mask.any())).upper(),
            "Annotation Confidence": "1.0" if reviewed else "",
        }
    )
    if anchor and mask.any():
        path = (
            mask_dir
            / "anchors"
            / video["Video ID"]
            / f"{int(row['Frame Index']):06d}.npz"
        )
        save_mask_bundle(path, mask[None], [identity])
        row["Anchor Path"] = path.resolve().relative_to(root).as_posix()
    elif reviewed and not mask.any():
        row["Anchor Path"] = ""


def validate_video_frames(
    root: Path,
    video: dict[str, str],
    frames: list[dict[str, str]],
    *,
    check_files: bool = True,
) -> None:
    for field in (
        "Video ID",
        "Tool ID",
        "Tool Class",
        "Tool Name",
        "Confidence",
        CAPTURE_SESSION_FIELD,
        "Source SHA256",
        "Extraction ID",
    ):
        if not video.get(field, "").strip():
            raise ValueError(f"Video metadata requires {field}.")
    if not 0 <= float(video["Confidence"]) <= 1:
        raise ValueError("Video identification confidence must be from 0 to 1.")
    if not parse_bool(video.get("Extraction Complete")):
        raise ValueError("Video extraction is incomplete; run prepare-videos first.")
    count = int(video["Frame Count"])
    if count <= 0 or int(video["Width"]) <= 0 or int(video["Height"]) <= 0:
        raise ValueError(
            "Prepared video must have positive frame count and dimensions."
        )
    timestamps = [
        float(row["Timestamp Seconds"])
        for row in sorted(frames, key=lambda row: int(row["Frame Index"]))
    ]
    if any(not math.isfinite(value) for value in timestamps) or timestamps != sorted(
        timestamps
    ):
        raise ValueError("Frame timestamps must be finite and in presentation order.")
    if len(frames) != count or sorted(
        int(row["Frame Index"]) for row in frames
    ) != list(range(count)):
        raise ValueError("Frame manifest must contain each decoded frame exactly once.")
    for row in frames:
        if row["Video ID"] != video["Video ID"]:
            raise ValueError("Frame refers to a different video.")
        if row["Frame ID"] != f"{video['Video ID']}:{int(row['Frame Index'])}":
            raise ValueError("Frame ID disagrees with video and frame index.")
        if check_files:
            image = read_frame(root, row)
            if image.size != (int(video["Width"]), int(video["Height"])):
                raise ValueError(
                    "Extracted frame dimensions disagree with the video manifest."
                )
        if row.get("Review Status") == "accepted":
            mask = read_mask(root, row, (int(video["Height"]), int(video["Width"])))
            positive = row.get("Visibility") in {"clear", "partial", "severe"}
            if not parse_bool(row.get("Reviewed")) or not parse_bool(
                row.get("Annotation Complete")
            ):
                raise ValueError("Accepted frame has incomplete annotation flags.")
            if positive != bool(mask.any()) or not row.get("Mask Path"):
                raise ValueError("Accepted frame visibility disagrees with its mask.")
            bundle = load_mask_bundle(resolve_path(root, row["Mask Path"]))
            if not mask.any() and row["Mask Index"] != "-1":
                raise ValueError(
                    "Nonvisible frame must not reference a positive mask index."
                )
            if row["Visibility"] not in {
                "clear",
                "partial",
                "severe",
                "occluded",
                "absent",
            }:
                raise ValueError("Accepted frame has invalid visibility.")
            class_valid = positive and row["Visibility"] != "severe"
            if (
                parse_bool(row.get("Visible")) != positive
                or parse_bool(row.get("Mask Valid")) != positive
                or parse_bool(row.get("Class Valid")) != class_valid
                or parse_bool(row.get("Base Valid")) != class_valid
            ):
                raise ValueError(
                    "Accepted frame annotation flags disagree with visibility."
                )
            recorded_box = (
                [
                    int(row[field])
                    for field in ("BBox X", "BBox Y", "BBox Width", "BBox Height")
                ]
                if positive
                else None
            )
            if mask2xywh(mask) != recorded_box:
                raise ValueError("Frame bounding box disagrees with the mask.")
            if mask.any() and (
                bundle.identities != (f"tool-{video['Tool ID']}",)
                or row["Mask Index"] != "0"
            ):
                raise ValueError(
                    "Accepted frame has an incorrect tool identity or mask index."
                )
