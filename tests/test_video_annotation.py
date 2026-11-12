import json
import sys
import weakref
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import annotate
import batch_propagate_videos
import video_annotation
from annotations import IMAGE_OBJECT_FIELDS, read_csv, write_csv
from image_review import Candidate
from masks import load_mask_bundle, save_mask_bundle
from video_backend import Sam3VideoBackend
from video_data import (
    FRAME_FIELDS,
    VideoLog,
    file_hash,
    prepare_videos,
    save_frame_mask,
    validate_video_frames,
)
from video_export import export_video_training


def write_video(path, times=(0, 1, 3, 4)):
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index, timestamp in enumerate(times):
            pixels = np.full((24, 32, 3), (index * 30) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def parse(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["annotate.py", *arguments, "--data-dir", "."])
    return annotate.parse_args()


def prepared(tmp_path, monkeypatch, times=(0, 1, 3, 4)):
    monkeypatch.chdir(tmp_path)
    write_video(Path("videos/12/tool.mp4"), times)
    write_csv(
        "image_object_manifest.csv",
        [
            {
                "Image ID": "image-1",
                "Instance ID": "tool-12",
                "Tool ID": "12",
                "Tool Class": "scissors",
                "Tool Name": "iris",
                "Manufacturer": "Example Instruments",
                "Confidence": "0.9",
            }
        ],
        IMAGE_OBJECT_FIELDS,
    )
    args = parse(monkeypatch, "prepare-videos")
    prepare_videos(args)
    video = read_csv("video_manifest.csv")[0]
    assert video["Manufacturer"] == "Example Instruments"
    return video, read_csv("video_frame_manifest.csv")


def mask():
    result = np.zeros((24, 32), dtype=np.bool_)
    result[4:12, 7:15] = True
    return result


def seed_frame(root, video, frames, index, decision="pending", anchor=True):
    save_frame_mask(
        root,
        frames[index],
        video,
        Path("masks/videos"),
        mask(),
        decision=decision,
        visibility="clear" if decision == "accepted" else "",
        anchor=anchor,
    )
    write_csv("video_frame_manifest.csv", frames, FRAME_FIELDS)
    row = frames[index]
    VideoLog(Path("video_review.jsonl")).append(
        row["Frame ID"],
        decision,
        mask_sha256=file_hash(Path(row["Mask Path"])) if decision == "accepted" else "",
        mask_path=row["Mask Path"],
        visibility=row["Visibility"],
        extraction_id=video["Extraction ID"],
        frame_sha256=row["Frame SHA256"],
    )


class Tracker:
    def __init__(self):
        self.calls = []
        self.releases = 0

    def release(self):
        self.releases += 1

    def track(self, images, anchors, start, stop, reverse):
        self.calls.append((len(images), sorted(anchors), start, stop, reverse))
        indices = range(start, stop - 1, -1) if reverse else range(start, stop + 1)
        for index in indices:
            nearest = min(anchors, key=lambda value: abs(value - index))
            yield index, anchors[nearest].copy()

    def text(self, image, prompt):
        value = np.ones((image.height, image.width), dtype=np.bool_)
        return [Candidate(value, 0.9, prompt, "candidate-0")]

    def clicks(self, image, points, box=None):
        return np.ones((image.height, image.width), dtype=np.bool_)


def test_batch_uses_last_anchor_and_preserves_earlier_and_reviewed_masks(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch, tuple(range(7)))
    for index, decision in [
        (0, "accepted"), (1, "pending"), (3, "pending"),
        (4, "pending"), (5, "accepted"),
    ]:
        seed_frame(tmp_path, video, frames, index, decision, anchor=index in (0, 3))
    for index in (3, 4):
        # Preserve the saved anchor at 3, but make its current frame mask differ:
        # propagation must use the actual anchor bundle, not this stale mask.
        save_frame_mask(
            tmp_path,
            frames[index],
            video,
            Path("masks/videos"),
            ~mask(),
            decision="pending",
        )
    write_csv("video_frame_manifest.csv", frames, FRAME_FIELDS)
    original = {
        index: (row.copy(), Path(row["Mask Path"]).read_bytes())
        for index, row in enumerate(frames)
        if row.get("Mask Path") and index != 4
    }
    backend = Tracker()
    monkeypatch.setattr(
        batch_propagate_videos, "Sam3VideoBackend", lambda *args: backend
    )

    def no_interaction(*args):
        pytest.fail("Batch propagation must not enter the interactive reviewer")

    monkeypatch.setattr(video_annotation.VideoReviewer, "run", no_interaction)
    args = parse(
        monkeypatch, "videos", "--device", "cpu",
        "--auto-preview", "--preview-size", "64",
    )
    report = batch_propagate_videos.run_batch(args)
    assert report[0]["status"] == "processed"
    assert report[0]["start_anchor_frame"] == 3
    assert report[0]["new_mask_frames"] == "6"
    assert report[0]["regenerated_mask_frames"] == "4"
    assert Path(report[0]["preview_directory"]).is_dir()
    assert json.loads(Path("batch_propagation_report.json").read_text()) == report
    saved = read_csv("video_frame_manifest.csv")
    for index, (row, contents) in original.items():
        assert saved[index] == row
        assert Path(row["Mask Path"]).read_bytes() == contents
    assert not saved[2]["Mask Path"]
    for index in (4, 6):
        assert saved[index]["Review Status"] == "pending"
        np.testing.assert_array_equal(
            load_mask_bundle(saved[index]["Mask Path"]).masks[0], mask()
        )
    calls = len(backend.calls)
    assert args.auto_preview is True
    second = batch_propagate_videos.run_batch(args)
    assert second[0]["status"] == "processed"
    assert second[0]["new_mask_frames"] == "none"
    assert second[0]["regenerated_mask_frames"] == "4, 6"
    assert len(backend.calls) > calls


def test_batch_can_start_from_nonzero_anchor_without_frame_zero_mask(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 2)
    monkeypatch.setattr(
        batch_propagate_videos, "Sam3VideoBackend", lambda *args: Tracker()
    )
    report = batch_propagate_videos.run_batch(
        parse(monkeypatch, "videos", "--device", "cpu")
    )
    assert report[0]["start_anchor_frame"] == 2
    assert report[0]["new_mask_frames"] == "3"
    saved = read_csv("video_frame_manifest.csv")
    assert not saved[0]["Mask Path"] and not saved[1]["Mask Path"]
    assert saved[3]["Mask Path"]


@pytest.mark.parametrize("failure", ["loss", "error"])
def test_batch_keeps_partial_progress_and_continues_to_next_video(
    tmp_path, monkeypatch, failure
):
    video, frames = prepared(tmp_path, monkeypatch, tuple(range(6)))
    seed_frame(tmp_path, video, frames, 0)
    write_video(Path("videos/12/second.mp4"), tuple(range(6)))
    prepare_videos(parse(monkeypatch, "prepare-videos"))
    videos = read_csv("video_manifest.csv")
    frames = read_csv("video_frame_manifest.csv")
    second = videos[1]
    second_start = next(row for row in frames if row["Video ID"] == second["Video ID"])
    save_frame_mask(
        tmp_path,
        second_start,
        second,
        Path("masks/videos"),
        mask(),
        decision="pending",
        anchor=True,
    )
    write_csv("video_frame_manifest.csv", frames, FRAME_FIELDS)

    class FailingTracker(Tracker):
        def track(self, images, anchors, start, stop, reverse):
            yield 0, mask()
            yield 1, mask()
            if failure == "error":
                raise RuntimeError("interrupted inference")
            yield 2, np.zeros_like(mask())

    backends = [FailingTracker(), Tracker()]
    pending = iter(backends)
    monkeypatch.setattr(
        batch_propagate_videos, "Sam3VideoBackend", lambda *args: next(pending)
    )
    args = parse(
        monkeypatch, "videos", "--device", "cpu", "--chunk-size", "3",
        "--auto-preview", "--preview-size", "64",
    )
    report = batch_propagate_videos.run_batch(args)
    assert report[0]["status"] == ("stopped" if failure == "loss" else "error")
    assert Path(report[0]["preview_directory"]).is_dir()
    assert report[1]["status"] == "processed"
    saved = read_csv("video_frame_manifest.csv")
    first_rows = [row for row in saved if row["Video ID"] == video["Video ID"]]
    assert first_rows[1]["Mask Path"]
    assert not first_rows[3]["Mask Path"]
    assert all(
        row["Mask Path"] for row in saved if row["Video ID"] == second["Video ID"]
    )
    assert all(row["Review Status"] == "pending" for row in saved)
    assert all(backend.releases >= 1 for backend in backends)


def test_batch_skips_video_without_any_anchor(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0, anchor=False)
    original = Path("video_frame_manifest.csv").read_bytes()
    backend = Tracker()
    monkeypatch.setattr(
        batch_propagate_videos, "Sam3VideoBackend", lambda *args: backend
    )
    report = batch_propagate_videos.run_batch(
        parse(monkeypatch, "videos", "--device", "cpu")
    )
    assert report[0]["status"] == "skipped"
    assert "No saved anchor" in report[0]["message"]
    assert not backend.calls
    assert Path("video_frame_manifest.csv").read_bytes() == original


def test_extraction_uses_decoded_timestamps_and_resumes_without_changing_annotations(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch)
    assert len(frames) == int(video["Frame Count"]) == 4
    assert [float(row["Timestamp Seconds"]) for row in frames] == pytest.approx(
        [0, 0.1, 0.3, 0.4]
    )
    assert [int(row["Frame Index"]) for row in frames] == list(range(4))
    assert all(Path(row["Filepath"]).suffix == ".png" for row in frames)
    seed_frame(tmp_path, video, frames, 0, "accepted")
    before = Path("video_frame_manifest.csv").read_bytes()
    generation = video["Extraction ID"]
    prepare_videos(parse(monkeypatch, "prepare-videos"))
    assert Path("video_frame_manifest.csv").read_bytes() == before
    assert read_csv("video_manifest.csv")[0]["Extraction ID"] == generation
    validate_video_frames(tmp_path, video, read_csv("video_frame_manifest.csv"))
    # Missing files can be recreated exactly from the unchanged source without invalidating masks.
    Path(frames[0]["Filepath"]).unlink()
    prepare_videos(parse(monkeypatch, "prepare-videos"))
    assert Path("video_frame_manifest.csv").read_bytes() == before


def test_changed_video_requires_explicit_reextraction_and_invalidates_masks(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0, "accepted")
    old_mask = Path(frames[0]["Mask Path"])
    old_tail = Path(frames[-1]["Filepath"])
    write_video(Path("videos/12/tool.mp4"), (0, 1))
    with pytest.raises(ValueError, match="--reextract"):
        prepare_videos(parse(monkeypatch, "prepare-videos"))
    prepare_videos(parse(monkeypatch, "prepare-videos", "--reextract"))
    updated = read_csv("video_frame_manifest.csv")
    assert len(updated) == 2
    assert all(
        not row["Mask Path"] and row["Review Status"] == "pending" for row in updated
    )
    assert not old_mask.exists() and not old_tail.exists()
    assert read_csv("video_manifest.csv")[0]["Extraction ID"] != video["Extraction ID"]
    assert (
        VideoLog(Path("video_review.jsonl")).decisions[frames[0]["Frame ID"]]
        == "pending"
    )


def test_corrupted_frame_and_unknown_or_conflicting_identity_are_rejected(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch)
    Path(frames[0]["Filepath"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="--reextract"):
        prepare_videos(parse(monkeypatch, "prepare-videos"))
    with pytest.raises(ValueError, match="Unknown Tool ID"):
        prepare_videos(parse(monkeypatch, "prepare-videos", "--tool-id", "99"))
    objects = read_csv("image_object_manifest.csv")
    objects.append({**objects[0], "Tool Name": "different"})
    write_csv("image_object_manifest.csv", objects, IMAGE_OBJECT_FIELDS)
    with pytest.raises(ValueError, match="conflicting"):
        prepare_videos(parse(monkeypatch, "prepare-videos"))


def test_both_directions_are_bounded_and_preserve_other_correction_anchors(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch, tuple(range(12)))
    seed_frame(tmp_path, video, frames, 6)
    seed_frame(tmp_path, video, frames, 3)
    seed_frame(tmp_path, video, frames, 0, "accepted")
    seed_frame(tmp_path, video, frames, 11, "accepted")
    outside = {
        index: Path(frames[index]["Mask Path"]).read_bytes() for index in (0, 11)
    }
    backend = Tracker()
    args = parse(monkeypatch, "videos", "--chunk-size", "3", "--context-frames", "1")
    reviewer = video_annotation.VideoReviewer(args, video, frames, backend)
    reviewer.propagate(6, 2, 9, "both")
    assert all(
        frames[index]["Review Status"] == "pending" and frames[index]["Mask Path"]
        for index in range(2, 10)
    )
    assert not frames[1]["Mask Path"] and not frames[10]["Mask Path"]
    assert all(
        Path(frames[index]["Mask Path"]).read_bytes() == content
        for index, content in outside.items()
    )
    assert all(call[0] <= 5 for call in backend.calls)
    assert {call[-1] for call in backend.calls} == {True, False}
    records = [
        json.loads(line) for line in Path("video_review.jsonl").read_text().splitlines()
    ]
    anchor_events = [
        record
        for record in records
        if record.get("action") == "propagate"
        and record["key"] == frames[6]["Frame ID"]
    ]
    assert len(anchor_events) == 1
    assert load_mask_bundle(frames[6]["Mask Path"]).identities == ("tool-12",)
    assert any(len(call[1]) > 1 for call in backend.calls)


def test_rejected_context_anchor_does_not_influence_propagation(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch, tuple(range(8)))
    seed_frame(tmp_path, video, frames, 1, "rejected")
    seed_frame(tmp_path, video, frames, 3)
    rejected = frames[1].copy()
    backend = Tracker()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--chunk-size", "3", "--context-frames", "2"),
        video,
        frames,
        backend,
    )
    reviewer.propagate(3, 3, 5, "forward")
    # The window starts at frame 1: local index 0 is the rejected reference,
    # while local index 2 is the valid starting anchor at frame 3.
    assert backend.calls[0][1] == [2]
    assert frames[1] == rejected


def test_propagation_requires_reopening_accepted_frames_and_range_acceptance_requires_preview(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 1)
    seed_frame(tmp_path, video, frames, 2, "accepted")
    backend = Tracker()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, backend
    )
    with pytest.raises(ValueError, match="reopen"):
        reviewer.propagate(1, 0, 3, "both")
    assert not backend.calls
    reviewer.reopen(2, 2)
    reviewer.propagate(1, 0, 3, "both")
    with pytest.raises(ValueError, match="Preview"):
        reviewer.accept_range(0, 3)
    reviewer.preview(0, 3)
    reviewer.accept_range(0, 3)
    assert all(row["Review Status"] == "accepted" for row in frames)


def test_range_acceptance_skips_accepted_frames_and_is_repeatable(
    tmp_path, monkeypatch, capsys
):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0, "accepted")
    for index in (1, 2, 3):
        seed_frame(tmp_path, video, frames, index)
    original = frames[0].copy()
    mask_bytes = Path(frames[0]["Mask Path"]).read_bytes()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--preview-size", "64"), video, frames, Tracker()
    )
    # Only the pending portion needs a current preview.
    reviewer.preview(1, 3)
    reviewer.accept_range(0, 3)
    assert all(row["Review Status"] == "accepted" for row in frames)
    assert frames[0] == original
    assert Path(frames[0]["Mask Path"]).read_bytes() == mask_bytes
    log_bytes = Path("video_review.jsonl").read_bytes()
    reviewer.inspected.clear()
    reviewer.accept_range(0, 3)
    assert Path("video_review.jsonl").read_bytes() == log_bytes
    output = capsys.readouterr().out
    assert "Newly accepted frames: 1–3." in output
    assert "Already accepted frames preserved: 0." in output
    assert "Newly accepted frames: none." in output
    assert "Already accepted frames preserved: 0–3." in output


def test_range_acceptance_still_blocks_rejected_frames(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0, "accepted")
    seed_frame(tmp_path, video, frames, 1, "rejected")
    seed_frame(tmp_path, video, frames, 2)
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--preview-size", "64"), video, frames, Tracker()
    )
    reviewer.preview(2, 2)
    with pytest.raises(ValueError, match="reopen"):
        reviewer.accept_range(0, 2)
    assert frames[1]["Review Status"] == "rejected"
    assert frames[2]["Review Status"] == "pending"


@pytest.mark.parametrize("change", ["mask", "visibility", "legacy"])
def test_changed_or_unverified_accepted_mask_is_previewed_and_reaccepted(
    tmp_path, monkeypatch, capsys, change
):
    video, frames = prepared(tmp_path, monkeypatch)
    for index in (0, 1):
        seed_frame(tmp_path, video, frames, index, "accepted")
    seed_frame(tmp_path, video, frames, 2)
    unchanged = frames[0].copy()
    unchanged_bytes = Path(frames[0]["Mask Path"]).read_bytes()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--preview-size", "64"), video, frames, Tracker()
    )
    if change == "mask":
        save_frame_mask(
            tmp_path,
            frames[1],
            video,
            Path("masks/videos"),
            ~mask(),
            decision="accepted",
            visibility="clear",
        )
    elif change == "visibility":
        frames[1]["Visibility"] = "partial"
    else:
        reviewer.log.append(frames[1]["Frame ID"], "accepted")
    reviewer.log = VideoLog(Path("video_review.jsonl"))
    assert not reviewer.log.acceptance_matches(tmp_path, video, frames[1])
    with pytest.raises(ValueError, match="Preview"):
        reviewer.accept_range(0, 2)
    assert frames[2]["Review Status"] == "pending"
    reviewer.preview(1, 2)
    reviewer.accept_range(0, 2)
    assert frames[0] == unchanged
    assert Path(frames[0]["Mask Path"]).read_bytes() == unchanged_bytes
    log = VideoLog(Path("video_review.jsonl"))
    assert all(log.acceptance_matches(tmp_path, video, row) for row in frames[:3])
    assert "Reaccepted changed or unverified frames: 1." in capsys.readouterr().out
    reviewer.inspected.clear()
    reviewer.accept_range(0, 2)


def test_bare_accept_reaccepts_changed_masks_without_preview(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0, "accepted")
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    save_frame_mask(
        tmp_path,
        frames[0],
        video,
        Path("masks/videos"),
        ~mask(),
        decision="accepted",
        visibility="clear",
    )
    reviewer.accept_masked()
    assert reviewer.log.acceptance_matches(tmp_path, video, frames[0])
    np.testing.assert_array_equal(
        load_mask_bundle(frames[0]["Mask Path"]).masks[0], ~mask()
    )


@pytest.mark.parametrize("command", ["verify-videos", "export-video-training"])
def test_changed_accepted_mask_cannot_verify_or_export(tmp_path, monkeypatch, command):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0, "accepted")
    save_frame_mask(
        tmp_path,
        frames[0],
        video,
        Path("masks/videos"),
        ~mask(),
        decision="accepted",
        visibility="clear",
    )
    write_csv("video_frame_manifest.csv", frames, FRAME_FIELDS)
    args = parse(monkeypatch, command)
    run = (
        video_annotation.verify_videos
        if command == "verify-videos"
        else export_video_training
    )
    with pytest.raises(ValueError, match="changed since acceptance"):
        run(args)


@pytest.mark.parametrize("direction,start,stop", [("forward", 2, 5), ("backward", 0, 2)])
def test_accepted_boundary_is_a_read_only_propagation_anchor(
    tmp_path, monkeypatch, capsys, direction, start, stop
):
    video, frames = prepared(tmp_path, monkeypatch, tuple(range(6)))
    seed_frame(tmp_path, video, frames, 2, "accepted", anchor=False)
    before = frames[2].copy()
    mask_bytes = Path(frames[2]["Mask Path"]).read_bytes()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    reviewer.propagate(2, start, stop, direction)
    assert "Preserved accepted starting mask: 2 (unchanged)." in capsys.readouterr().out
    assert frames[2] == before
    assert Path(frames[2]["Mask Path"]).read_bytes() == mask_bytes
    log = VideoLog(Path("video_review.jsonl"))
    assert log.decisions[frames[2]["Frame ID"]] == "accepted"
    for index in range(start, stop + 1):
        if index != 2:
            assert frames[index]["Review Status"] == "pending"
            np.testing.assert_array_equal(
                load_mask_bundle(frames[index]["Mask Path"]).masks[0], mask()
            )


def test_propagation_reports_new_regenerated_and_reused_frame_ranges(
    tmp_path, monkeypatch, capsys
):
    video, frames = prepared(tmp_path, monkeypatch, tuple(range(8)))
    for index in (2, 5):
        seed_frame(tmp_path, video, frames, index)
    stale = np.zeros_like(mask())
    stale[0:2, 0:2] = True
    for index in (3, 6):
        save_frame_mask(
            tmp_path,
            frames[index],
            video,
            Path("masks/videos"),
            stale,
            decision="pending",
        )
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--chunk-size", "3"), video, frames, Tracker()
    )
    reviewer.propagate(2, 0, 7, "both")
    output = capsys.readouterr().out
    assert "Propagation completed: frames 0–7 (both)." in output
    assert "New pending masks: 0–1, 4, 7." in output
    assert "Regenerated pending masks: 3, 6." in output
    assert "Reused anchor masks: 2, 5." in output
    assert "Preserved accepted" not in output
    for index in (3, 6):
        np.testing.assert_array_equal(
            load_mask_bundle(frames[index]["Mask Path"]).masks[0], mask()
        )


def test_continue_from_accepted_range_through_terminal_commands(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    for index in (0, 1):
        seed_frame(tmp_path, video, frames, index, "accepted", anchor=False)
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    commands = iter(
        ["jump 1", "prop forward 1 3", "preview 2 3", "accept 2 3"]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert reviewer.run() is True
    assert all(row["Review Status"] == "accepted" for row in frames)


@pytest.mark.parametrize("decision", ["pending", "accepted"])
def test_remove_anchor_preserves_mask_and_review_status(tmp_path, monkeypatch, decision):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 1, decision)
    original = frames[1].copy()
    anchor_path = Path(original["Anchor Path"])
    mask_bytes = Path(original["Mask Path"]).read_bytes()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    reviewer.remove_anchor(1)
    assert frames[1] == {**original, "Anchor Path": ""}
    assert Path(original["Mask Path"]).read_bytes() == mask_bytes
    assert not anchor_path.exists()
    assert read_csv("video_frame_manifest.csv")[1]["Anchor Path"] == ""
    event = json.loads(Path("video_review.jsonl").read_text().splitlines()[-1])
    assert event["decision"] == decision
    assert event["action"] == "remove_anchor"
    assert event["anchor_path"] == original["Anchor Path"]
    if decision == "accepted":
        assert VideoLog(Path("video_review.jsonl")).acceptance_matches(
            tmp_path, video, frames[1]
        )
    reviewer.remove_anchor(1)
    validate_video_frames(tmp_path, video, frames)


@pytest.mark.parametrize("decision", ["pending", "accepted"])
def test_existing_mask_can_become_anchor_without_changing_review(
    tmp_path, monkeypatch, decision
):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 1, decision, anchor=False)
    original = frames[1].copy()
    mask_bytes = Path(original["Mask Path"]).read_bytes()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    reviewer.set_anchor(1)
    assert {**frames[1], "Anchor Path": ""} == original
    assert Path(original["Mask Path"]).read_bytes() == mask_bytes
    np.testing.assert_array_equal(
        load_mask_bundle(frames[1]["Anchor Path"]).masks[0], mask()
    )
    event = json.loads(Path("video_review.jsonl").read_text().splitlines()[-1])
    assert event["decision"] == decision
    assert event["action"] == "set_anchor"
    if decision == "accepted":
        assert VideoLog(Path("video_review.jsonl")).acceptance_matches(
            tmp_path, video, frames[1]
        )
    validate_video_frames(tmp_path, video, frames)


def test_two_sided_anchors_are_passed_together_for_interior_repair(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    for index in (0, 2):
        seed_frame(tmp_path, video, frames, index, "accepted", anchor=False)
    backend = Tracker()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--context-frames", "1"), video, frames, backend
    )
    original = {index: Path(frames[index]["Mask Path"]).read_bytes() for index in (0, 2)}
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    commands = iter(["anchor 0", "jump 2", "anchor", "prop backward 1", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert reviewer.run() is False
    assert backend.calls[0][1] == [0, 2]
    assert frames[1]["Review Status"] == "pending"
    assert frames[1]["Mask Path"]
    for index in (0, 2):
        assert frames[index]["Review Status"] == "accepted"
        assert Path(frames[index]["Mask Path"]).read_bytes() == original[index]


def test_cannot_anchor_a_frame_without_visible_mask(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    with pytest.raises(ValueError, match="nonempty"):
        reviewer.set_anchor(1)
    assert not frames[1]["Anchor Path"]


@pytest.mark.parametrize("command", ["unanchor 1", "jump 1\nunanchor"])
def test_unanchor_allows_backward_propagation_to_replace_bad_correction(
    tmp_path, monkeypatch, command
):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 1)
    seed_frame(tmp_path, video, frames, 3)
    bad = ~mask()
    save_frame_mask(
        tmp_path,
        frames[1],
        video,
        Path("masks/videos"),
        bad,
        decision="pending",
        anchor=True,
    )
    backend = Tracker()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, backend
    )
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    commands = iter([*command.splitlines(), "jump 3", "prop backward 3", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert reviewer.run() is False
    assert not frames[1]["Anchor Path"]
    np.testing.assert_array_equal(
        load_mask_bundle(frames[1]["Mask Path"]).masks[0], mask()
    )
    assert all(1 not in call[1] for call in backend.calls)


def test_bare_accept_approves_only_pending_visible_masks_in_current_video(
    tmp_path, monkeypatch, capsys
):
    video, frames = prepared(tmp_path, monkeypatch, tuple(range(6)))
    for index, decision in [
        (0, "accepted"),
        (1, "pending"),
        (2, "rejected"),
        (4, "pending"),
    ]:
        seed_frame(tmp_path, video, frames, index, decision)
    frames[1]["Visibility"] = "partial"
    frames[4]["Visibility"] = "severe"
    save_frame_mask(
        tmp_path,
        frames[3],
        video,
        Path("masks/videos"),
        np.zeros((24, 32), dtype=np.bool_),
        decision="pending",
    )
    other = {**frames[1], "Video ID": "other-video", "Frame ID": "other-video:1"}
    all_frames = [*frames, other]
    unchanged = {index: frames[index].copy() for index in (0, 2, 3, 5)}
    other_before = other.copy()
    anchors = {
        index: Path(frames[index]["Anchor Path"]).read_bytes() for index in (1, 4)
    }
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, all_frames, Tracker()
    )
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    commands = iter(["jump 5", "accept", "accept", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert reviewer.run() is False
    for index in (1, 4):
        assert frames[index]["Review Status"] == "accepted"
        assert frames[index]["Reviewed"] == "TRUE"
        assert frames[index]["Annotation Complete"] == "TRUE"
        assert Path(frames[index]["Anchor Path"]).read_bytes() == anchors[index]
    assert frames[1]["Visibility"] == "partial"
    assert frames[4]["Visibility"] == "severe"
    assert frames[4]["Class Valid"] == "FALSE"
    for index, row in unchanged.items():
        assert frames[index] == row
    assert other == other_before
    events = [
        json.loads(line)
        for line in Path("video_review.jsonl").read_text().splitlines()
    ]
    approved = [event for event in events if event.get("action") == "accept_all_masked"]
    assert {event["key"] for event in approved} == {
        frames[1]["Frame ID"],
        frames[4]["Frame ID"],
    }
    assert len(approved) == 2
    validate_video_frames(tmp_path, video, frames)
    output = capsys.readouterr().out
    assert "Accepted 2 masked frames" in output
    assert "Accepted 0 masked frames" in output


def test_fully_reviewed_video_can_be_opened_and_reopened(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    for index in range(len(frames)):
        seed_frame(tmp_path, video, frames, index, "accepted")
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--stay-on-video"), video, frames, Tracker()
    )
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    commands = iter(["reopen 1 2", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert reviewer.run() is False
    assert [row["Review Status"] for row in frames] == [
        "accepted",
        "pending",
        "pending",
        "accepted",
    ]


def test_default_video_queue_skips_finished_video(tmp_path, monkeypatch, capsys):
    video, frames = prepared(tmp_path, monkeypatch)
    for index in range(len(frames)):
        seed_frame(tmp_path, video, frames, index, "accepted")
    write_video(Path("videos/12/second.mp4"))
    prepare_videos(parse(monkeypatch, "prepare-videos"))
    backend = Tracker()
    monkeypatch.setattr(video_annotation, "Sam3VideoBackend", lambda *args: backend)
    displayed = []
    monkeypatch.setattr(
        video_annotation, "show_frame", lambda *args: displayed.append(args)
    )
    monkeypatch.setattr("builtins.input", lambda _: "q")
    video_annotation.annotate_videos(parse(monkeypatch, "videos"))
    assert len(displayed) == 1
    assert backend.releases == 1
    output = capsys.readouterr().out
    assert "Finished video videos/12/tool.mp4; moving to the next video." in output
    assert "Opening videos/12/second.mp4" in output


def test_explicit_video_id_opens_finished_video_and_stays(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    for index in range(len(frames)):
        seed_frame(tmp_path, video, frames, index, "accepted")
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--video-id", video["Video ID"]),
        video,
        frames,
        Tracker(),
    )
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    monkeypatch.setattr("builtins.input", lambda _: "q")
    assert reviewer.is_finished()
    assert reviewer.run() is False


@pytest.mark.parametrize("last_command,result", [("q", False), ("stay off", True)])
def test_stay_command_controls_advancing_after_final_acceptance(
    tmp_path, monkeypatch, last_command, result
):
    video, frames = prepared(tmp_path, monkeypatch)
    for index in range(len(frames)):
        seed_frame(tmp_path, video, frames, index)
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    commands = iter(["stay", "accept", last_command])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert reviewer.run() is result
    assert reviewer.is_finished()


def test_next_video_leaves_pending_masks_unchanged(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0)
    original = [row.copy() for row in frames]
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    monkeypatch.setattr("builtins.input", lambda _: "next-video")
    assert reviewer.run() is True
    assert not reviewer.is_finished()
    assert frames == original


@pytest.mark.parametrize("change", ["mask", "legacy"])
def test_modified_or_unverified_video_is_not_skipped_as_finished(
    tmp_path, monkeypatch, change
):
    video, frames = prepared(tmp_path, monkeypatch)
    for index in range(len(frames)):
        seed_frame(tmp_path, video, frames, index, "accepted")
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    assert reviewer.is_finished()
    if change == "mask":
        save_frame_mask(
            tmp_path,
            frames[0],
            video,
            Path("masks/videos"),
            ~mask(),
            decision="accepted",
            visibility="clear",
        )
    else:
        reviewer.log.append(frames[0]["Frame ID"], "accepted")
    assert not reviewer.is_finished()
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    monkeypatch.setattr("builtins.input", lambda _: "q")
    assert reviewer.run() is False


@pytest.mark.parametrize(
    "command,expected",
    [
        ("prop forward 2", (3, 3, 5, "forward")),
        ("prop backward 2", (3, 1, 3, "backward")),
        ("prop both 2", (3, 1, 5, "both")),
        ("prop forward 100", (3, 3, 7, "forward")),
        ("prop backward 100", (3, 0, 3, "backward")),
        ("prop both 100", (3, 0, 7, "both")),
        ("prop forward 3 5", (3, 3, 5, "forward")),
        ("propr forward 2", (3, 3, 5, "forward")),
    ],
)
def test_propagation_count_shorthand(tmp_path, monkeypatch, command, expected):
    video, frames = prepared(tmp_path, monkeypatch, tuple(range(8)))
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    calls = []
    monkeypatch.setattr(reviewer, "propagate", lambda *args: calls.append(args))
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    commands = iter(["jump 3", command, "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert reviewer.run() is False
    assert calls == [expected]


@pytest.mark.parametrize("count", [0, -1])
def test_propagation_count_must_be_positive(tmp_path, monkeypatch, capsys, count):
    video, frames = prepared(tmp_path, monkeypatch)
    backend = Tracker()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, backend
    )
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    commands = iter([f"prop forward {count}", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert reviewer.run() is False
    assert "frame count must be positive" in capsys.readouterr().out
    assert not backend.calls


def test_video_displays_zoom_to_mask_and_can_show_full_frame(monkeypatch):
    import matplotlib.pyplot as plt
    from image_refinement import tool_crop
    from video_ui import show_frame

    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    monkeypatch.setattr(plt, "pause", lambda *args: None)
    image = Image.new("RGB", (32, 24))
    left, top, right, bottom = tool_crop(mask(), padding=0.4)
    try:
        show_frame(image, mask(), "test", [])
        axis = plt.gcf().axes[0]
        np.testing.assert_allclose(axis.get_xlim(), (left - 0.5, right - 0.5))
        np.testing.assert_allclose(axis.get_ylim(), (bottom - 0.5, top - 0.5))
        show_frame(image, mask(), "test", [], zoom=False)
        np.testing.assert_allclose(plt.gcf().axes[0].get_xlim(), (-0.5, 31.5))
    finally:
        plt.close("all")


def test_saved_previews_stream_frames_into_contact_sheets(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    video, frames = prepared(tmp_path, monkeypatch, tuple(range(25)))
    seed_frame(tmp_path, video, frames, 0)
    backend = Tracker()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--preview-size", "64"), video, frames, backend
    )
    live = []
    read = video_annotation.read_frame

    def read_one(root, row):
        image = read(root, row)
        live.append(weakref.ref(image))
        assert sum(reference() is not None for reference in live) <= 2
        return image

    def no_gui(*args, **kwargs):
        pytest.fail("Saved previews must not open GUI windows")

    monkeypatch.setattr(video_annotation, "read_frame", read_one)
    monkeypatch.setattr(plt, "show", no_gui)
    directory = reviewer.preview(0, 24)
    assert backend.releases == 1
    assert sorted(path.name for path in directory.glob("*.png")) == [
        "000000-000011.png",
        "000012-000023.png",
        "000024-000024.png",
    ]
    with Image.open(directory / "000000-000011.png") as sheet:
        assert sheet.size == (256, 312)
    snapshot = json.loads((directory / "manifest.json").read_text())
    assert snapshot["video_id"] == video["Video ID"]
    assert snapshot["extraction_id"] == video["Extraction ID"]
    assert [row["frame_index"] for row in snapshot["frames"]] == list(range(25))
    assert snapshot["frames"][0]["mask_sha256"] == file_hash(
        Path(frames[0]["Mask Path"])
    )
    assert reviewer.inspected == {0: snapshot["frames"][0]["mask_sha256"]}
    assert all(row["Review Status"] == "pending" for row in frames)


def test_auto_preview_saves_sheets_after_propagation(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0)
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--auto-preview", "--preview-size", "64"),
        video,
        frames,
        Tracker(),
    )
    reviewer.propagate(0, 0, 3, "forward")
    assert len(list(Path("previews/videos").rglob("*.png"))) == 1
    assert set(reviewer.inspected) == set(range(4))
    assert all(row["Review Status"] == "pending" for row in frames)
    # The saved snapshot qualifies for explicit range acceptance, but changing
    # a mask invalidates that snapshot exactly as an interactive preview did.
    reviewer.save(1, ~mask(), "pending")
    with pytest.raises(ValueError, match="Preview"):
        reviewer.accept_range(0, 3)


def test_frame_display_passes_only_bounded_raster_to_matplotlib(monkeypatch):
    import matplotlib.pyplot as plt
    from video_ui import show_frame

    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    monkeypatch.setattr(plt, "pause", lambda *args: None)
    image = Image.new("RGB", (2160, 2160), "white")
    mask = np.zeros((2160, 2160), dtype=np.bool_)
    mask[500:1500, 500:1500] = True
    try:
        show_frame(image, mask, "test", [], zoom=False)
        rendered = plt.gcf().axes[0].images
        assert len(rendered) == 1
        assert rendered[0].get_array().shape == (1024, 1024, 3)
        assert rendered[0].get_array().dtype == np.uint8
        np.testing.assert_allclose(plt.gcf().axes[0].get_xlim(), (-0.5, 2159.5))
    finally:
        plt.close("all")
        image.close()


def test_click_region_preserves_pixels_outside_it_and_records_coordinates(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0)
    before = mask()
    args = parse(monkeypatch, "videos")
    reviewer = video_annotation.VideoReviewer(args, video, frames, Tracker())
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    monkeypatch.setattr(
        video_annotation,
        "collect_clicks",
        lambda image, mask, initial_points: [(1.0, 1.0, 1), (2.0, 2.0, 0)],
    )
    commands = iter(["region 1 1 5 5", "click", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert reviewer.run() is False
    result = load_mask_bundle(frames[0]["Mask Path"]).masks[0]
    expected = before.copy()
    expected[1:5, 1:5] = True
    np.testing.assert_array_equal(result, expected)
    records = [
        json.loads(line) for line in Path("video_review.jsonl").read_text().splitlines()
    ]
    correction = next(
        record for record in records if record.get("action") == "click_correction"
    )
    assert correction["region"] == [1, 1, 5, 5]
    assert correction["points"] == [[1.0, 1.0, 1], [2.0, 2.0, 0]]
    assert frames[0]["Review Status"] == "pending"


def test_empty_prediction_is_unresolved_and_absent_is_an_explicit_empty_bundle(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch)
    root = tmp_path
    with pytest.raises(ValueError, match="explicitly"):
        save_frame_mask(
            root,
            frames[0],
            video,
            Path("masks/videos"),
            np.zeros((24, 32), bool),
            decision="accepted",
            visibility="clear",
        )
    save_frame_mask(
        root,
        frames[0],
        video,
        Path("masks/videos"),
        np.zeros((24, 32), bool),
        decision="accepted",
        visibility="absent",
    )
    assert load_mask_bundle(frames[0]["Mask Path"]).masks.shape == (0, 24, 32)
    assert (
        frames[0]["Visible"] == "FALSE" and frames[0]["Annotation Complete"] == "TRUE"
    )
    validate_video_frames(root, video, frames)


def test_export_uses_saved_frames_and_preserves_an_immutable_snapshot(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0, "accepted")
    seed_frame(tmp_path, video, frames, 1, "accepted")
    save_frame_mask(
        tmp_path,
        frames[2],
        video,
        Path("masks/videos"),
        np.zeros((24, 32), bool),
        decision="accepted",
        visibility="absent",
    )
    write_csv("video_frame_manifest.csv", frames, FRAME_FIELDS)
    VideoLog(Path("video_review.jsonl")).append(frames[2]["Frame ID"], "accepted")
    args = parse(monkeypatch, "export-video-training")
    export_video_training(args)
    images = read_csv("video_training/image_manifest.csv")
    objects = read_csv("video_training/image_object_manifest.csv")
    assert len(images) == len(objects) == 3
    assert all(row["Manufacturer"] == "Example Instruments" for row in objects)
    assert all(
        row["Capture Session ID"] == video["Capture Session ID"] for row in images
    )
    assert objects[-1]["Reviewed"] == ""
    saved_mask = Path("video_training") / objects[0]["Mask Path"]
    before = file_hash(saved_mask)
    changed = np.zeros((24, 32), bool)
    changed[12:20, 16:24] = True
    save_frame_mask(
        tmp_path,
        frames[0],
        video,
        Path("masks/videos"),
        changed,
        decision="accepted",
        visibility="clear",
    )
    assert (
        file_hash(saved_mask) == before
    )  # Atomic source replacement leaves the snapshot intact.
    assert not (Path("video_training") / images[0]["Filepath"]).is_symlink()


def test_tracker_adapter_initializes_all_anchors_and_uses_the_requested_direction():
    backend = Sam3VideoBackend("unused", torch.device("cpu"))
    calls = []
    seeds = []
    released = []
    session = SimpleNamespace(
        processed_frames={0: object()},
        reset_inference_session=lambda: released.append(True),
    )

    class Processor:
        def init_video_session(self, **kwargs):
            return session

        def add_inputs_to_inference_session(self, session, **kwargs):
            seeds.append(kwargs["frame_idx"])

        def post_process_masks(self, masks, original_sizes, binarize):
            return [torch.ones((1, 1, 24, 32), dtype=torch.bool)]

    class Model:
        def __call__(self, **kwargs):
            return SimpleNamespace(pred_masks=torch.ones((1, 1, 24, 32)))

        def propagate_in_video_iterator(self, **kwargs):
            calls.append(kwargs)
            yield SimpleNamespace(frame_idx=1, pred_masks=torch.ones((1, 1, 24, 32)))
            yield SimpleNamespace(frame_idx=0, pred_masks=torch.ones((1, 1, 24, 32)))

    backend.tracker, backend.video_processor = Model(), Processor()
    result = dict(
        backend.track(
            [Image.new("RGB", (32, 24))] * 2, {0: mask(), 1: mask()}, 1, 0, True
        )
    )
    assert seeds == [0, 1]
    assert calls[0]["start_frame_idx"] == 1
    assert calls[0]["max_frame_num_to_track"] == 1
    assert calls[0]["reverse"] is True
    np.testing.assert_array_equal(result[0], mask())
    assert released == [True]
    assert session.processed_frames is None


@pytest.mark.parametrize("fail", [False, True])
def test_tracking_session_is_released_on_close_or_error(monkeypatch, fail):
    backend = Sam3VideoBackend("unused", torch.device("cpu"))
    released = []
    session = SimpleNamespace(
        processed_frames={0: object()},
        reset_inference_session=lambda: released.append(True),
    )

    class Processor:
        def add_inputs_to_inference_session(self, *args, **kwargs):
            pass

    class Model:
        def __call__(self, **kwargs):
            pass

        def propagate_in_video_iterator(self, **kwargs):
            yield SimpleNamespace(frame_idx=0)
            raise RuntimeError("interrupted inference")

    backend.tracker, backend.video_processor = Model(), Processor()
    monkeypatch.setattr(backend, "_session", lambda images: session)
    predictions = backend._track_window(
        [Image.new("RGB", (32, 24))] * 2, {0: mask()}, 0, 1, False
    )
    next(predictions)
    assert not released
    if fail:
        with pytest.raises(RuntimeError, match="interrupted"):
            next(predictions)
    else:
        predictions.close()
    assert released == [True]
    assert session.processed_frames is None


def test_propagation_saves_predictions_before_inference_finishes(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0)

    class InterruptedTracker(Tracker):
        def track(self, images, anchors, start, stop, reverse):
            yield 0, mask()
            yield 1, mask()
            # These writes must happen before requesting the next prediction.
            saved = read_csv("video_frame_manifest.csv")[1]
            np.testing.assert_array_equal(
                load_mask_bundle(saved["Mask Path"]).masks[0], mask()
            )
            assert saved["Review Status"] == "pending"
            raise RuntimeError("interrupted inference")

    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, InterruptedTracker()
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        reviewer.propagate(0, 0, 3, "forward")
    assert frames[1]["Mask Path"]
    assert not frames[2]["Mask Path"]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("jump", [6, 50])
def test_tracking_expands_crop_and_reframes_reliable_anchor(monkeypatch, reverse, jump):
    backend = Sam3VideoBackend("unused", torch.device("cpu"))
    images, truth = [], []
    for index in range(4):
        pixels = np.zeros((80, 200, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(200)
        pixels[:, :, 1] = np.arange(80)[:, None]
        pixels[:, :, 2] = index
        images.append(Image.fromarray(pixels))
        value = np.zeros((80, 200), dtype=np.bool_)
        x = 20 + jump * index
        value[30:40, x : x + 10] = True
        truth.append(value)
    if reverse:
        truth.reverse()
    start, stop = (3, 0) if reverse else (0, 3)
    calls = []

    def track_window(crops, anchors, first, last, backward):
        left, top = np.asarray(crops[0])[0, 0, :2].astype(int)
        width, height = crops[0].size
        calls.append((left, top, width, height, first))
        # A retry must seed from a correctly reframed, unclipped mask.
        np.testing.assert_array_equal(
            anchors[first], truth[first][top : top + height, left : left + width]
        )
        indices = range(first, last - 1, -1) if backward else range(first, last + 1)
        for index in indices:
            yield index, truth[index][top : top + height, left : left + width]

    monkeypatch.setattr(backend, "_track_window", track_window)
    outputs = dict(backend.track(images, {start: truth[start]}, start, stop, reverse))
    assert len(calls) > 1
    assert calls[-1][2] > calls[0][2]
    assert set(outputs) == set(range(4))
    for index, expected in enumerate(truth):
        np.testing.assert_array_equal(outputs[index], expected)


def test_video_group_refines_selected_proposal_and_saves_full_frame_anchor(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch)
    backend = Tracker()
    sizes = []

    def text(image, prompt):
        sizes.append(image.size)
        value = (
            mask()
            if image.size == (32, 24)
            else np.zeros((image.height, image.width), dtype=np.bool_)
        )
        if image.size != (32, 24):
            value[2:10, 2:10] = True
            value[3, 3] = False
        return [Candidate(value, 0.9, prompt, "candidate-0")]

    backend.text = text
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, backend
    )
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    commands = iter(["p", "g 0", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    assert reviewer.run() is False
    assert sizes == [(32, 24), (12, 12)]
    expected = mask()
    expected[5, 8] = False
    for field in ("Mask Path", "Anchor Path"):
        np.testing.assert_array_equal(
            load_mask_bundle(frames[0][field]).masks[0], expected
        )


def test_tracking_retries_missing_mask_until_full_frame_then_leaves_it_empty(monkeypatch):
    backend = Sam3VideoBackend("unused", torch.device("cpu"))
    images = [Image.new("RGB", (32, 24))] * 2
    sizes = []

    def track_window(crops, anchors, start, stop, reverse):
        sizes.append(crops[0].size)
        yield start, anchors[start]
        yield stop, np.zeros((crops[0].height, crops[0].width), dtype=np.bool_)

    monkeypatch.setattr(backend, "_track_window", track_window)
    outputs = dict(backend.track(images, {0: mask()}, 0, 1, False))
    assert sizes[0] != (32, 24)
    assert sizes[-1] == (32, 24)
    np.testing.assert_array_equal(outputs[0], mask())
    assert outputs[1].shape == (24, 32)
    assert not outputs[1].any()


def test_mouse_editor_collects_positive_negative_clicks_and_undo(monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
    from video_ui import collect_clicks

    def interact(*, block):
        figure = plt.gcf()
        figure.canvas.draw()
        axis = figure.axes[0]
        for x, y, button in [
            (3, 4, MouseButton.LEFT),
            (8, 9, MouseButton.RIGHT),
            (12, 13, MouseButton.LEFT),
        ]:
            screen_x, screen_y = axis.transData.transform((x, y))
            event = MouseEvent(
                "button_press_event", figure.canvas, screen_x, screen_y, button=button
            )
            figure.canvas.callbacks.process("button_press_event", event)
        figure.canvas.callbacks.process(
            "key_press_event",
            KeyEvent("key_press_event", figure.canvas, key="backspace"),
        )
        figure.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", figure.canvas, key="enter")
        )

    monkeypatch.setattr(plt, "show", interact)
    points = collect_clicks(Image.new("RGB", (32, 24)), mask(), [])
    assert points is not None
    assert len(points) == 2
    assert [point[2] for point in points] == [1, 0]
    assert points[0][:2] == pytest.approx((3, 4), abs=0.1)
    assert points[1][:2] == pytest.approx((8, 9), abs=0.1)
