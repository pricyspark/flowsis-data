"""Review orientation, then normalize every video frame using the blue markers."""

import argparse
import json
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile

import av
import cv2
import numpy as np

from marker_tracking import MarkerTracker
from square_crop import (
    CROP_SIZE,
    MarkerDetectionError,
    find_markers,
    warp_quad_to_square,
)

ROOT = Path(__file__).resolve().parent / "data"
SOURCE_DIR = ROOT / "raw-videos"
OUTPUT_DIR = ROOT / "videos"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mts", ".m2ts", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review and convert raw videos to marker-stabilized 2160x2160 MP4s."
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--force", action="store_true", help="reprocess and replace existing MP4s"
    )
    return parser.parse_args()


def rotate_frame(image: np.ndarray, turns: int) -> np.ndarray:
    """Rotate clockwise in quarter turns, without interpolation."""
    return np.ascontiguousarray(np.rot90(image, -turns % 4))


def save_failure_frame(
    image: np.ndarray,
    path: Path,
    metadata: dict[str, str | int | float | list[list[float]]],
) -> None:
    """Save the exact oriented detector input, plus context for reproducing failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def review_orientation(image: np.ndarray, source: Path) -> int | None:
    """Return the accepted clockwise quarter turns, or None to quit the batch."""
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent

    turns = 0
    accepted = False
    # Disable shortcuts such as 'q' closing unrelated figures or 'l' changing axes.
    with plt.rc_context({key: [] for key in plt.rcParams if key.startswith("keymap.")}):
        figure, axes = plt.subplots(figsize=(10, 8))
        artist = axes.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        (outline,) = axes.plot([], [], color="lime", linewidth=1)
        axes.set_axis_off()

        def redraw() -> None:
            oriented = rotate_frame(image, turns)
            pixels = cv2.cvtColor(oriented, cv2.COLOR_BGR2RGB)
            height, width = pixels.shape[:2]
            artist.set_data(pixels)
            artist.set_extent((-0.5, width - 0.5, height - 0.5, -0.5))
            axes.set_xlim(-0.5, width - 0.5)
            axes.set_ylim(height - 0.5, -0.5)
            try:
                matches = find_markers(oriented)
                points = np.array([match.point for match in matches])[[0, 1, 3, 2, 0]]
                outline.set_data(points[:, 0], points[:, 1])
                status = "Check the green outline follows the marker inner corners."
            except MarkerDetectionError as error:
                outline.set_data([], [])
                status = f"Marker detection failed: {error}"
            axes.set_title(
                f"{source.name} — {turns * 90}° clockwise\n"
                "r / right: rotate clockwise | l / left: counterclockwise\n"
                "a / enter: accept | q / escape / close: quit batch\n"
                f"{status}",
                fontsize=10,
            )
            figure.canvas.draw_idle()

        def on_key(event: KeyEvent) -> None:
            nonlocal turns, accepted
            if event.key in {"r", "right", "l", "left"}:
                turns = (turns + (1 if event.key in {"r", "right"} else -1)) % 4
                redraw()
            elif event.key in {"a", "enter"}:
                accepted = True
                plt.close(figure)
            elif event.key in {"q", "escape"}:
                plt.close(figure)

        figure.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        try:
            plt.show(block=True)
        finally:
            plt.close(figure)
    return turns if accepted else None


def save_cropped_video(source: Path, destination: Path) -> int | None:
    """Publish only a complete video; return its frame count or None on user quit."""
    temporary: Path | None = None
    try:
        with av.open(str(source)) as container:
            if len(container.streams.video) != 1:
                raise ValueError("expected exactly one video stream")
            stream = container.streams.video[0]
            frames = container.decode(stream)
            first = next(frames, None)
            if first is None:
                raise ValueError("video contains no decoded frames")
            if first.is_corrupt or first.pts is None or first.time_base is None:
                raise ValueError("frame 1 is corrupt or has no presentation timestamp")
            turns = review_orientation(first.to_ndarray(format="bgr24"), source)
            if turns is None:
                return None

            time_base = Fraction(first.time_base)
            start_time = first.pts * time_base
            previous_time: Fraction | None = None
            tracker = MarkerTracker()
            rate = Fraction(stream.average_rate or stream.guessed_rate or 30)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                prefix=f".{destination.stem}.",
                suffix=".convert-tmp.mp4",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)

            with av.open(
                str(temporary), mode="w", options={"movflags": "+faststart"}
            ) as output:
                encoded = output.add_stream("libx264", rate=rate)
                encoded.width = encoded.height = CROP_SIZE
                encoded.pix_fmt = "yuv420p"
                encoded.time_base = encoded.codec_context.time_base = time_base
                encoded.codec_context.gop_size = max(1, round(rate))
                encoded.options = {"crf": "18", "preset": "medium"}
                count = 0
                decoded = first
                while decoded is not None:
                    count += 1
                    if (
                        decoded.is_corrupt
                        or decoded.pts is None
                        or decoded.time_base is None
                    ):
                        raise ValueError(f"frame {count} is corrupt or has no timestamp")
                    timestamp = decoded.pts * decoded.time_base
                    if previous_time is not None and timestamp <= previous_time:
                        raise ValueError(f"frame {count} has a non-increasing timestamp")
                    previous_time = timestamp
                    pixels = rotate_frame(decoded.to_ndarray(format="bgr24"), turns)
                    try:
                        corners = tracker.find_corners(
                            pixels, float(timestamp - start_time)
                        )
                        cropped = warp_quad_to_square(pixels, corners, size=CROP_SIZE)
                    except MarkerDetectionError as error:
                        failure_path = (
                            destination.parent
                            / "conversion-failures"
                            / destination.stem
                            / f"frame-{count:06d}.png"
                        )
                        try:
                            save_failure_frame(
                                pixels,
                                failure_path,
                                {
                                    "source": str(source.resolve()),
                                    "frame_number": count,
                                    "frame_index": count - 1,
                                    "timestamp_seconds": float(timestamp - start_time),
                                    "source_pts": decoded.pts,
                                    "source_time_base": str(decoded.time_base),
                                    "rotation_degrees_clockwise": turns * 90,
                                    "error": str(error),
                                    "search_bounds": (
                                        tracker.search_bounds.tolist()
                                        if tracker.search_bounds is not None
                                        else []
                                    ),
                                },
                            )
                            diagnostic = f"failed frame saved to {failure_path}"
                        except (OSError, cv2.error) as save_error:
                            diagnostic = f"could not save failed frame: {save_error}"
                        raise MarkerDetectionError(
                            f"frame {count} at {float(timestamp - start_time):.3f}s: "
                            f"{error}; {diagnostic}"
                        ) from error
                    frame = av.VideoFrame.from_ndarray(cropped, format="bgr24")
                    frame.time_base = time_base
                    frame.pts = round((timestamp - start_time) / time_base)
                    frame.duration = round(
                        decoded.duration * decoded.time_base / time_base
                    )
                    for packet in encoded.encode(frame):
                        output.mux(packet)
                    if count % 100 == 0:
                        print(f"  {source.name}: {count} frames processed", flush=True)
                    decoded = next(frames, None)

                if stream.frames and count != stream.frames:
                    raise ValueError(
                        f"decoded {count} frames, but the source reports {stream.frames}"
                    )
                for packet in encoded.encode():
                    output.mux(packet)
            temporary.replace(destination)
            return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(
    force: bool = False,
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> int:
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if (
        source_dir == output_dir
        or source_dir in output_dir.parents
        or output_dir in source_dir.parents
    ):
        raise ValueError(
            "source and output directories must be separate, non-nested directories"
        )
    sources = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not sources:
        print(f"No raw videos found under {source_dir}.")
        return 1
    destinations = [
        (output_dir / source.relative_to(source_dir)).with_suffix(".mp4")
        for source in sources
    ]
    if len(set(destinations)) != len(destinations):
        raise ValueError("raw videos share an output path; rename files with the same stem")

    failures: list[tuple[Path, str]] = []
    for source, destination in zip(sources, destinations, strict=True):
        if destination.exists() and not force:
            print(f"Skipping existing file: {destination}")
            continue
        try:
            count = save_cropped_video(source, destination)
        except (av.error.FFmpegError, OSError, ValueError) as error:
            removal = ""
            if force and destination.exists():
                destination.unlink()
                removal = "; existing MP4 removed"
            reason = f"{error}{removal}"
            failures.append((source.relative_to(source_dir), reason))
            print(f"Warning: skipping entire video {source}: {reason}", flush=True)
            continue
        if count is None:
            print("Conversion stopped by user.")
            break
        print(f"{source} -> {destination} ({count} frames)")

    if failures:
        print(f"\nWarning: {len(failures)} video(s) could not be converted:")
        for source, reason in failures:
            print(f"- {source}: {reason}")
    return int(bool(failures))


if __name__ == "__main__":
    args = parse_args()
    try:
        raise SystemExit(main(args.force, args.source_dir, args.output_dir))
    except KeyboardInterrupt:
        print("\nConversion interrupted; unfinished output discarded.")
        raise SystemExit(130)
