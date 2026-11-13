"""Unattended forward propagation from each video's last saved anchor.

Run: python batch_propagate_videos.py --chunk-size 16 --context-frames 1 --auto-preview
Uses the options from `annotate.py videos`; never opens the interactive reviewer
or accepts masks. Earlier frames and reviewed frames are preserved; pending
predictions after the last anchor are regenerated.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from annotate import parse_args
from annotations import atomic_write_text, read_csv
from masks import load_mask_bundle
from video_annotation import VideoReviewer, frame_ranges
from video_backend import Sam3VideoBackend
from video_data import file_hash, resolve_path, select_videos


def run_batch(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.start_frame != 0 or args.end_frame is not None or args.auto_accept:
        raise ValueError(
            "Batch mode uses the last anchor without accepting masks; "
            "omit --start-frame, --end-frame, and --auto-accept."
        )
    if args.chunk_size < 2 or args.context_frames < 0:
        raise ValueError(
            "chunk-size must be at least 2; context-frames must be nonnegative."
        )
    if not 64 <= args.preview_size <= 2048:
        raise ValueError("preview-size must be between 64 and 2048 pixels.")
    make_previews = args.auto_preview
    args = argparse.Namespace(**vars(args))
    args.auto_preview = False  # Save even partial results after a video stops.
    root = args.manifest.parent.resolve()
    videos = read_csv(args.manifest)
    all_frames = read_csv(args.frame_manifest)
    selected = select_videos(args, videos)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    report: list[dict[str, object]] = []
    report_path = root / "batch_propagation_report.json"
    for number, video in selected:
        rows = sorted(
            (row for row in all_frames if row["Video ID"] == video["Video ID"]),
            key=lambda row: int(row["Frame Index"]),
        )
        before = {index for index, row in enumerate(rows) if row.get("Mask Path")}
        result: dict[str, object] = {
            "video_id": video["Video ID"],
            "video_number": number,
            "filepath": video["Filepath"],
            "status": "skipped",
            "message": "No unreviewed frames after the last anchor.",
        }
        backend = Sam3VideoBackend(args.model, device, args.threshold)
        reviewer = None
        written: set[int] = set()
        anchor_index = None
        print(f"\nBatch video {number}: {video['Filepath']}", flush=True)
        try:
            anchors = [
                index for index, row in enumerate(rows)
                if row.get("Anchor Path") and row.get("Review Status") != "rejected"
            ]
            if not anchors:
                result["message"] = (
                    "No saved anchor; create one in the annotator first."
                )
            else:
                anchor_index = max(anchors)
                result["start_anchor_frame"] = anchor_index
                bundle = load_mask_bundle(
                    resolve_path(root, rows[anchor_index]["Anchor Path"])
                )
                if (
                    bundle.masks.shape
                    != (1, int(video["Height"]), int(video["Width"]))
                    or not bundle.masks[0].any()
                ):
                    raise ValueError(
                        f"Anchor at frame {anchor_index} is empty or has invalid dimensions."
                    )
                pending = {
                    index for index in range(anchor_index + 1, len(rows))
                    if rows[index].get("Review Status") not in {"accepted", "rejected"}
                }
                if pending:
                    if (
                        file_hash(resolve_path(root, video["Filepath"]))
                        != video["Source SHA256"]
                    ):
                        raise ValueError(
                            "Source MP4 changed; reextract before propagating."
                        )
                    reviewer = VideoReviewer(args, video, all_frames, backend)
                    print(
                        f"Starting from saved anchor {anchor_index}; "
                        "pending predictions after it will be regenerated.",
                        flush=True,
                    )
                    result.update(
                        status="processed",
                        message=f"Propagated forwards from anchor {anchor_index}.",
                    )
                    while pending:
                        first = min(pending)
                        seed = first - 1
                        seed_mask = (
                            bundle.masks[0]
                            if seed == anchor_index
                            else reviewer.mask(seed)
                        )
                        if (
                            rows[seed].get("Review Status") == "rejected"
                            or not seed_mask.any()
                        ):
                            result.update(
                                status="stopped",
                                message=(
                                    f"Frame {first} needs a nonempty preceding mask "
                                    "to resume."
                                ),
                            )
                            break
                        last = first
                        while last + 1 < len(rows) and last + 1 in pending:
                            last += 1
                        reviewer.propagate(
                            seed,
                            seed,
                            last,
                            "forward",
                            preserve_start=True,
                            seed_mask=seed_mask,
                            on_prediction=written.add,
                        )
                        pending.difference_update(range(first, last + 1))
        except Exception as error:
            # A bad video, tracking loss, or CUDA OOM must not abandon the queue.
            result.update(
                status=(
                    "stopped" if "Tracking lost the tool" in str(error) else "error"
                ),
                message=f"{type(error).__name__}: {error}",
            )
        finally:
            try:
                backend.release()
            except Exception as error:
                result["cleanup_error"] = str(error)
        generated = written - before
        result["new_mask_frames"] = frame_ranges(generated)
        result["regenerated_mask_frames"] = frame_ranges(written & before)
        if (
            make_previews
            and written
            and reviewer is not None
            and anchor_index is not None
        ):
            try:
                result["preview_directory"] = str(
                    reviewer.preview(anchor_index, max(written)).resolve()
                )
            except Exception as error:
                result["preview_error"] = f"{type(error).__name__}: {error}"
        print(
            f"{result['status']}: {result['message']} "
            f"New masks: {result['new_mask_frames']}. "
            f"Regenerated: {result['regenerated_mask_frames']}.",
            flush=True,
        )
        report.append(result)
        atomic_write_text(report_path, json.dumps(report, indent=2))
    print(f"\nBatch report: {report_path.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    run_batch(parse_args(["videos", *sys.argv[1:]]))
