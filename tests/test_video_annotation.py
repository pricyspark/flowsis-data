import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import annotate
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
                "Confidence": "0.9",
            }
        ],
        IMAGE_OBJECT_FIELDS,
    )
    args = parse(monkeypatch, "prepare-videos")
    prepare_videos(args)
    return read_csv("video_manifest.csv")[0], read_csv("video_frame_manifest.csv")


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
    VideoLog(Path("video_review.jsonl")).append(frames[index]["Frame ID"], decision)


class Tracker:
    def __init__(self):
        self.calls = []

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
    monkeypatch.setattr(video_annotation, "preview_frames", lambda items: None)
    reviewer.preview(0, 3)
    reviewer.accept_range(0, 3)
    assert all(row["Review Status"] == "accepted" for row in frames)


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

    class Processor:
        def init_video_session(self, **kwargs):
            return object()

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
