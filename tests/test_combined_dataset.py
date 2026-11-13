"""Integration coverage for shared media, accepted runs, and temporal split boundaries."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from datasets import load_from_disk
from PIL import Image

from annotations import read_csv, write_csv
from build_dataset import build_dataset, build_rows
from video_data import FRAME_FIELDS, VIDEO_FIELDS, VideoLog, file_hash, save_frame_mask
from video_export import export_video_training
from test_build_dataset import source  # Shared still-image fixture.


@pytest.fixture
def combined(source):
    videos, frames = [], []
    log = VideoLog(source / "video_review.jsonl")
    for number in range(2):
        video = dict.fromkeys(VIDEO_FIELDS, "")
        video.update(
            {
                "Video ID": f"v{number}",
                "Tool ID": f"video-tool-{number}",
                "Tool Class": "scissors",
                "Tool Name": "iris",
                "Confidence": "1",
                "Capture Session ID": f"session-video-{number}",
                "Source SHA256": "source",
                "Extraction ID": "generation",
                "Extraction Complete": "TRUE",
                "Frame Count": "90",
                "Width": "12",
                "Height": "10",
                "FPS": "10",
                "Filepath": f"videos/{number}.mp4",
            }
        )
        videos.append(video)
        for index in range(90):
            path = source / f"frames/videos/v{number}/{index:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 10), (index, 0, 0)).save(path)
            row = dict.fromkeys(FRAME_FIELDS, "")
            row.update(
                {
                    "Video ID": video["Video ID"],
                    "Frame ID": f"v{number}:{index}",
                    "Frame Index": str(index),
                    "Timestamp Seconds": str(index / 10),
                    "Filepath": path.relative_to(source).as_posix(),
                    "Frame SHA256": file_hash(path),
                    "Review Status": "pending",
                }
            )
            if 2 <= index <= 81 or index == 85:
                visibility = {5: "occluded", 6: "severe", 7: "absent"}.get(
                    index, "clear"
                )
                mask = np.zeros((10, 12), dtype=bool)
                if visibility not in {"absent", "occluded"}:
                    mask[2:5, 3:7] = True
                save_frame_mask(
                    source,
                    row,
                    video,
                    source / "masks/videos",
                    mask,
                    decision="accepted",
                    visibility=visibility,
                )
                log.append(
                    row["Frame ID"],
                    "accepted",
                    mask_sha256=file_hash(source / row["Mask Path"]),
                    mask_path=row["Mask Path"],
                    visibility=visibility,
                    extraction_id=video["Extraction ID"],
                    frame_sha256=row["Frame SHA256"],
                )
            frames.append(row)
    write_csv(source / "video_manifest.csv", videos, VIDEO_FIELDS)
    write_csv(source / "video_frame_manifest.csv", frames, FRAME_FIELDS)
    return source


def test_combined_export_references_assets_and_first_run(combined):
    output = combined / "export"
    build_dataset(combined, output, validation_size=0, test_size=0)
    data = load_from_disk(output)["train"]
    videos = [row for row in data if row["video_id"]]
    assert len(data) == 166
    assert len(videos) == 160
    assert {row["frame_index"] for row in videos} == set(range(2, 82))
    assert not list(output.rglob("*.png"))
    assert not list(output.rglob("*.npz"))
    for row in videos:
        assert (output / row["image_path"]).is_file()
        assert (output / row["mask_bundle_path"]).is_file()
        assert row["timestamp_seconds"] == row["frame_index"] / 10
        assert bool(row["tool_ids"])
        assert row["base_eligible"] == (row["frame_index"] not in {5, 6})
        if row["frame_index"] == 7:
            assert row["objects"] == []
        if row["frame_index"] == 5:
            assert row["objects"][0]["track_id"].startswith("tool-")
            assert not row["objects"][0]["mask_valid"]
    report = json.loads((output / "build_report.json").read_text())
    assert report["video_runs"]["v0"] == {
        "start_frame": 2,
        "end_frame": 81,
        "frames": 80,
    }
    assert report["skipped"]["video_outside_first_accepted_run"] == 20


def test_section_splits_preserve_intervals_and_report_gaps(combined):
    result = build_dataset(
        combined,
        combined / "sections",
        video_sections=2,
        validation_size=0.25,
        test_size=0.25,
        split_gap=2,
    )
    seen = set()
    for split, subset in result.items():
        groups = {}
        for row in subset:
            assert row["image_id"] not in seen
            seen.add(row["image_id"])
            if row["video_id"]:
                groups.setdefault(row["sequence_id"], []).append(row["frame_index"])
        for indices in groups.values():
            assert indices == list(range(min(indices), max(indices) + 1))
            assert len(indices) >= 8
    report = json.loads((combined / "sections/build_report.json").read_text())
    assert report["section_gap_frames_excluded"] == 16
    assert len(report["video_intervals"]) == 12
    assert sum(len(subset) for subset in result.values()) == 150
    assert report["splits"]["validation"]["snippets"]["length=8,stride=1"] > 0
    assert report["splits"]["validation"]["snippets"]["length=8,stride=3"] == 0
    with pytest.raises(ValueError, match="Reduce --video-sections"):
        build_dataset(
            combined, combined / "short", video_sections=8, short_video_policy="error"
        )


def test_changed_selected_assets_rejected_but_later_runs_ignored(combined):
    frames = read_csv(combined / "video_frame_manifest.csv")
    excluded = next(row for row in frames if row["Frame ID"] == "v0:85")
    (combined / excluded["Filepath"]).write_bytes(b"excluded asset is irrelevant")
    rows, _, _ = build_rows(combined, include_videos=True)
    assert len(rows) == 166
    selected = next(row for row in frames if row["Frame ID"] == "v0:2")
    (combined / selected["Filepath"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        build_rows(combined, include_videos=True)


def test_image_style_export_uses_the_same_first_run(combined):
    output = combined / "snapshot"
    export_video_training(
        argparse.Namespace(
            manifest=combined / "video_manifest.csv",
            frame_manifest=combined / "video_frame_manifest.csv",
            review_log=combined / "video_review.jsonl",
            output_dir=output,
            video=None,
            video_id=None,
            replace=False,
        )
    )
    exported = read_csv(output / "image_manifest.csv")
    assert len(exported) == 156  # Two severe/occluded exclusions per video.
    assert all(2 <= int(row["Frame Index"]) <= 81 for row in exported)
    # The snapshot also remains consumable by the local builder, retaining ordering.
    rows, _, _ = build_rows(output)
    assert len(rows) == 156
    assert all(row["video_id"] and row["frame_index"] >= 2 for row in rows)


def test_short_runs_can_be_kept_in_training_with_explicit_report(combined):
    result = build_dataset(combined, combined / "short-train", video_sections=8)
    assert all(
        not row["video_id"]
        for name, split in result.items()
        if name != "train"
        for row in split
    )
    assert sum(bool(row["video_id"]) for row in result["train"]) == 160
    report = json.loads((combined / "short-train/build_report.json").read_text())
    assert all(
        interval["reason"] == "too_short_for_section_holdouts"
        for interval in report["video_intervals"]
    )


def test_session_splits_keep_still_and_video_captures_together(combined):
    from build_dataset import split_rows

    rows, _, _ = build_rows(combined, include_videos=True)
    rows[0]["capture_session_id"] = "session-video-0"
    assigned = split_rows(
        rows,
        {"train": 0.5, "validation": 0.25, "test": 0.25},
        seed=42,
        split_by="session",
    )
    assert {
        split
        for row, split in zip(rows, assigned, strict=True)
        if row["capture_session_id"] == "session-video-0"
    } == {assigned[0]}
