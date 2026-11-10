import json
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

import video_convert
from square_crop import MarkerDetectionError, crop_marked_square


def marked_frame():
    image = np.full((512, 512, 3), 180, dtype=np.uint8)
    marker = np.zeros((100, 100), dtype=np.bool_)
    marker[:29, :] = True
    marker[:, :29] = True
    for y, x, shape in (
        (20, 20, marker),
        (20, 392, np.fliplr(marker)),
        (392, 20, np.flipud(marker)),
        (392, 392, np.flip(marker)),
    ):
        image[y : y + 100, x : x + 100][shape] = (255, 85, 0)
    return image


def write_video(path, images, times=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    times = range(len(images)) if times is None else times
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 512
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "12"}
        for image, pts in zip(images, times, strict=True):
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = pts, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def conversion(monkeypatch):
    monkeypatch.setattr(video_convert, "review_orientation", lambda image, source: 1)
    monkeypatch.setattr(video_convert, "CROP_SIZE", 64)
    monkeypatch.setattr(
        video_convert, "crop_marked_square", lambda image: crop_marked_square(image, 64)
    )


def test_conversion_preserves_every_frame_and_variable_timing(tmp_path, conversion):
    source, destination = tmp_path / "raw.mov", tmp_path / "output.mp4"
    images = [marked_frame() for _ in range(4)]
    for index, image in enumerate(images):
        # An asymmetric interior feature verifies accepted rotation in the encoded output.
        image[150:230, 150:230] = (20 + index * 30, 20, 240)
    write_video(source, images, (5, 6, 8, 9))
    assert video_convert.save_cropped_video(source, destination) == 4
    with av.open(str(destination)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        frames = list(container.decode(video=0))
    assert len(frames) == 4
    assert [frame.time for frame in frames] == pytest.approx([0, 0.1, 0.3, 0.4])
    assert all((frame.width, frame.height) == (64, 64) for frame in frames)
    pixels = frames[0].to_ndarray(format="bgr24")
    red = pixels[..., 2].astype(float) - pixels[..., 0]
    assert red[:32, 32:].mean() > red[:32, :32].mean() + 10
    assert not list(tmp_path.glob(".*.convert-tmp.mp4"))


def test_missing_marker_discards_entire_video(tmp_path, conversion):
    source, destination = tmp_path / "raw.mp4", tmp_path / "out.mp4"
    bad = marked_frame()
    bad[:256, :256] = 180
    write_video(source, [marked_frame(), bad, marked_frame()])
    with pytest.raises(MarkerDetectionError, match="frame 2 at"):
        video_convert.save_cropped_video(source, destination)
    assert not destination.exists()
    assert source.exists()
    assert not list(tmp_path.glob(".*.convert-tmp.mp4"))
    failure = tmp_path / "conversion-failures/out/frame-000002.png"
    metadata = json.loads(failure.with_suffix(".json").read_text())
    with av.open(str(source)) as container:
        decoded = list(container.decode(video=0))[1]
        expected = video_convert.rotate_frame(decoded.to_ndarray(format="bgr24"), 1)
    np.testing.assert_array_equal(cv2.imread(str(failure)), expected)
    assert metadata["source"] == str(source.resolve())
    assert metadata["frame_number"] == 2
    assert metadata["frame_index"] == 1
    assert metadata["timestamp_seconds"] == pytest.approx(0.1)
    assert metadata["source_pts"] == decoded.pts
    assert metadata["source_time_base"] == str(decoded.time_base)
    assert metadata["rotation_degrees_clockwise"] == 90
    assert metadata["error"]


def test_diagnostic_write_failure_preserves_detection_error(
    tmp_path, monkeypatch, conversion
):
    source, destination = tmp_path / "raw.mp4", tmp_path / "out.mp4"
    write_video(source, [np.full((512, 512, 3), 180, dtype=np.uint8)])
    monkeypatch.setattr(cv2, "imwrite", lambda path, pixels: False)
    with pytest.raises(MarkerDetectionError) as caught:
        video_convert.save_cropped_video(source, destination)
    assert "no blue marker found" in str(caught.value)
    assert "could not save failed frame" in str(caught.value)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.convert-tmp.mp4"))


def test_default_output_is_2160_square(tmp_path, monkeypatch):
    source, destination = tmp_path / "raw.mov", tmp_path / "out.mp4"
    write_video(source, [marked_frame()])
    monkeypatch.setattr(video_convert, "review_orientation", lambda image, source: 0)
    assert video_convert.save_cropped_video(source, destination) == 1
    with av.open(str(destination)) as container:
        frame = next(container.decode(video=0))
        assert (frame.width, frame.height) == (2160, 2160)


def test_batch_continues_after_failure_and_force_removes_old_output(
    tmp_path, conversion, capsys
):
    raw, output = tmp_path / "raw", tmp_path / "output"
    write_video(raw / "12" / "bad.mov", [np.zeros((512, 512, 3), np.uint8)])
    write_video(raw / "12" / "good.mov", [marked_frame()])
    old = output / "12" / "bad.mp4"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old output")
    assert video_convert.main(True, raw, output) == 1
    assert not old.exists()
    assert (output / "12" / "good.mp4").exists()
    warning = capsys.readouterr().out
    assert "skipping entire video" in warning
    failure = output / "12/conversion-failures/bad/frame-000001.png"
    assert failure.is_file()
    assert f"failed frame saved to {failure}" in warning
    assert json.loads(failure.with_suffix(".json").read_text())["frame_number"] == 1
    assert not list(output.rglob(".*.convert-tmp.mp4"))


def test_existing_output_skipped_and_quit_preserves_it(tmp_path, monkeypatch, conversion):
    raw, output = tmp_path / "raw", tmp_path / "output"
    write_video(raw / "12" / "video.mp4", [marked_frame()])
    destination = output / "12" / "video.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old output")

    def quit_review(image, source):
        return None

    monkeypatch.setattr(video_convert, "review_orientation", quit_review)
    assert video_convert.main(False, raw, output) == 0
    assert video_convert.main(True, raw, output) == 0
    assert destination.read_bytes() == b"old output"
    assert not list(output.rglob(".*.convert-tmp.mp4"))


def test_colliding_names_rejected_before_conversion(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "same.mov").touch()
    (raw / "same.mp4").touch()
    with pytest.raises(ValueError, match="share an output path"):
        video_convert.main(False, raw, tmp_path / "output")


def test_orientation_repeats_until_accept_and_handles_quit(monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent

    def show(**kwargs):
        figure = plt.gcf()
        for key in ("r", "r", "left", "a"):
            event = KeyEvent("key_press_event", figure.canvas, key)
            figure.canvas.callbacks.process("key_press_event", event)

    monkeypatch.setattr(plt, "show", show)
    image = np.zeros((20, 30, 3), np.uint8)
    assert video_convert.review_orientation(image, Path("video.mov")) == 1

    def close(**kwargs):
        plt.close(plt.gcf())

    monkeypatch.setattr(plt, "show", close)
    assert video_convert.review_orientation(image, Path("video.mov")) is None


def test_interruption_cleans_up_temporary_output(tmp_path, monkeypatch, conversion):
    source, destination = tmp_path / "raw.mp4", tmp_path / "out.mp4"
    write_video(source, [marked_frame()])

    def interrupt(image):
        raise KeyboardInterrupt

    monkeypatch.setattr(video_convert, "crop_marked_square", interrupt)
    with pytest.raises(KeyboardInterrupt):
        video_convert.save_cropped_video(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.convert-tmp.mp4"))
