"""Expose extracted video frames to the existing flowsis image dataset builder."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from annotations import CAPTURE_SESSION_FIELD, IMAGE_OBJECT_FIELDS, read_csv, write_csv
from video_data import VideoLog, file_hash, resolve_path, validate_video_frames


def snapshot_file(source: Path, destination: Path, replace: bool) -> None:
    """Hard-link immutable files; atomic source replacement cannot change an export."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() and file_hash(destination) == file_hash(source):
            return
        if not replace:
            raise ValueError(
                f"Export file already exists with different contents: {destination}. Use --replace or a new output directory."
            )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.unlink(missing_ok=True)
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def export_video_training(args: argparse.Namespace) -> None:
    root = args.manifest.parent.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    videos = read_csv(args.manifest)
    frames = read_csv(args.frame_manifest)
    log = VideoLog(args.review_log)
    images: list[dict[str, str]] = []
    objects: list[dict[str, str]] = []
    accepted: list[str] = []
    for video in videos:
        if args.video_id and video["Video ID"] != args.video_id:
            continue
        selected = [row for row in frames if row["Video ID"] == video["Video ID"]]
        validate_video_frames(root, video, selected)
        for row in selected:
            if (
                row.get("Review Status") != "accepted"
                or log.decisions.get(row["Frame ID"]) != "accepted"
            ):
                continue
            # Fully hidden tools and class-invalid severe masks are excluded from this base training export.
            if row["Visibility"] in {"occluded", "severe"}:
                continue
            name = f"{int(row['Frame Index']):06d}"
            image_path = Path("images") / video["Video ID"] / f"{name}.png"
            mask_path = Path("masks/images") / video["Video ID"] / f"{name}.npz"
            snapshot_file(
                resolve_path(root, row["Filepath"]), output / image_path, args.replace
            )
            snapshot_file(
                resolve_path(root, row["Mask Path"]), output / mask_path, args.replace
            )
            images.append(
                {
                    "Image ID": row["Frame ID"],
                    "Tool IDs": video["Tool ID"],
                    "Filepath": image_path.as_posix(),
                    "Sample ID": "",
                    "Angle": "",
                    CAPTURE_SESSION_FIELD: video[CAPTURE_SESSION_FIELD],
                    "Video ID": video["Video ID"],
                    "Frame Index": row["Frame Index"],
                    "Timestamp Seconds": row["Timestamp Seconds"],
                }
            )
            obj = dict.fromkeys(IMAGE_OBJECT_FIELDS, "")
            obj.update(
                {
                    field: video.get(field, "")
                    for field in (
                        "Tool ID", "Tool Class", "Tool Name", "Manufacturer", "Confidence"
                    )
                }
            )
            obj.update(
                {
                    "Image ID": row["Frame ID"],
                    "Instance ID": f"tool-{video['Tool ID']}",
                    "Filepath": image_path.as_posix(),
                }
            )
            if row["Visibility"] != "absent":
                obj.update(
                    {field: row[field] for field in IMAGE_OBJECT_FIELDS if field in row}
                )
                obj["Mask Path"] = mask_path.as_posix()
                obj["Filepath"] = image_path.as_posix()
                obj["Visibility"] = row["Visibility"]
            objects.append(obj)
            accepted.append(row["Frame ID"])
    if not images:
        raise ValueError("No accepted, base-valid video frames to export.")
    fields = (
        "Tool IDs",
        "Sample ID",
        "Angle",
        "Filepath",
        "Image ID",
        CAPTURE_SESSION_FIELD,
        "Video ID",
        "Frame Index",
        "Timestamp Seconds",
    )
    write_csv(output / "image_manifest.csv", images, fields)
    write_csv(output / "image_object_manifest.csv", objects, IMAGE_OBJECT_FIELDS)
    from annotations import atomic_write_text
    import json

    atomic_write_text(
        output / "image_review.jsonl",
        "".join(
            json.dumps({"key": key, "decision": "accepted"}) + "\n" for key in accepted
        ),
    )
    print(
        f"Exported {len(images)} independently loadable frames to {output}; snapshot files reuse storage where possible."
    )
