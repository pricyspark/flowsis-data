from pathlib import Path

import pytest

import video_annotation
from annotations import read_csv
from image_review import Candidate
from test_video_annotation import Tracker, mask, parse, prepared, seed_frame
from video_data import VideoLog, file_hash, save_frame_mask


def run_commands(monkeypatch, reviewer, commands):
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    replies = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    return reviewer.run()


@pytest.mark.parametrize(
    "command", ["prop forward 0 3", "prop both 0 3", "prop between 0 3"]
)
def test_undo_propagation_restores_files_rows_and_log(tmp_path, monkeypatch, command):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0)
    if "between" in command:
        seed_frame(tmp_path, video, frames, 3, "accepted")
    rows_before = [row.copy() for row in frames]
    files_before = {path: path.read_bytes() for path in Path("masks").rglob("*.npz")}
    log = VideoLog(Path("video_review.jsonl"))
    decisions_before = log.decisions.copy()
    versions_before = log.accepted_versions.copy()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    assert run_commands(monkeypatch, reviewer, [command, "undo", "q"]) is False
    assert frames == rows_before
    assert read_csv("video_frame_manifest.csv") == rows_before
    assert {
        path: path.read_bytes() for path in Path("masks").rglob("*.npz")
    } == files_before
    restored = VideoLog(Path("video_review.jsonl"))
    assert restored.decisions == decisions_before
    assert restored.accepted_versions == versions_before
    assert reviewer._undo is None


@pytest.mark.parametrize(
    "command", ["accept", "accept 0 3", "a", "r", "absent", "occluded", "v partial"]
)
def test_undo_review_decisions(tmp_path, monkeypatch, command):
    video, frames = prepared(tmp_path, monkeypatch)
    for index in range(4):
        seed_frame(tmp_path, video, frames, index)
    before = [row.copy() for row in frames]
    files = {path: path.read_bytes() for path in Path("masks").rglob("*.npz")}
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    # A saved preview is normally needed for explicit range acceptance.
    if command == "accept 0 3":
        reviewer.inspected = {
            index: file_hash(Path(row["Mask Path"])) for index, row in enumerate(frames)
        }
    assert run_commands(monkeypatch, reviewer, [command, "undo", "q"]) is False
    assert frames == before
    assert {path: path.read_bytes() for path in Path("masks").rglob("*.npz")} == files
    assert set(VideoLog(Path("video_review.jsonl")).decisions.values()) == {"pending"}


@pytest.mark.parametrize("command", ["unanchor 0", "anchor 0", "reopen 0 1"])
def test_undo_anchor_and_reopen_preserves_acceptance(tmp_path, monkeypatch, command):
    video, frames = prepared(tmp_path, monkeypatch)
    for index in (0, 1):
        seed_frame(
            tmp_path, video, frames, index, "accepted", anchor=command != "anchor 0"
        )
    before = [row.copy() for row in frames]
    files = {path: path.read_bytes() for path in Path("masks").rglob("*.npz")}
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    assert run_commands(monkeypatch, reviewer, [command, "undo", "q"]) is False
    assert frames == before
    assert {path: path.read_bytes() for path in Path("masks").rglob("*.npz")} == files
    restored = VideoLog(Path("video_review.jsonl"))
    assert all(restored.acceptance_matches(tmp_path, video, row) for row in frames[:2])


@pytest.mark.parametrize("command", ["click", "click reset", "box", "g 0"])
def test_undo_mask_correction_restores_mask_anchor_and_prompts(
    tmp_path, monkeypatch, command
):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0)
    before = [row.copy() for row in frames]
    files = {path: path.read_bytes() for path in Path("masks").rglob("*.npz")}
    backend = Tracker()
    backend.text = lambda *args: [Candidate(~mask(), 1.0, "tool", "0")]
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--no-refine-crop"), video, frames, backend
    )
    points = [(8.0, 5.0, 1)]
    reviewer.click_history[(0, None)] = points
    monkeypatch.setattr(
        video_annotation, "collect_clicks", lambda *args: [(1.0, 1.0, 1)]
    )
    monkeypatch.setattr(video_annotation, "select_region", lambda *args: (0, 0, 16, 16))
    commands = ["p"] if command == "g 0" else []
    assert run_commands(
        monkeypatch, reviewer, [*commands, command, "undo", "q"]
    ) is False
    assert frames == before
    assert {path: path.read_bytes() for path in Path("masks").rglob("*.npz")} == files
    assert reviewer.click_history[(0, None)] == points


@pytest.mark.parametrize("command", ["jump 2", "pending", "next", "s"])
def test_undo_navigation_returns_to_previous_frame(tmp_path, monkeypatch, command):
    video, frames = prepared(tmp_path, monkeypatch)
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    displayed = []
    replies = iter([command, "undo", "q"])
    monkeypatch.setattr(
        video_annotation, "show_frame", lambda *args: displayed.append(args[2])
    )
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert reviewer.run() is False
    assert "frame 0 @" in displayed[0]
    assert "frame 0 @" not in displayed[1]
    assert "frame 0 @" in displayed[2]
    # Undoing next also restores a previously missing log decision on replay.
    assert VideoLog(Path("video_review.jsonl")).decisions == {}


def test_status_preview_and_failed_commands_preserve_undo_point(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0)
    before = [row.copy() for row in frames]
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos", "--preview-size", "64"), video, frames, Tracker()
    )
    monkeypatch.setattr(video_annotation, "collect_clicks", lambda *args: None)
    commands = ["v partial", "status", "preview 0 0", "unknown", "click", "undo", "q"]
    assert run_commands(monkeypatch, reviewer, commands) is False
    assert frames == before
    assert list(Path("previews").rglob("*.png"))


def test_undo_partial_failed_propagation_restores_every_touched_frame(
    tmp_path, monkeypatch
):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0)
    before = [row.copy() for row in frames]

    class InterruptedTracker(Tracker):
        def track(self, images, anchors, start, stop, reverse):
            yield start, anchors[start].copy()
            yield start + 1, ~mask()
            raise ValueError("Tracking interrupted")

    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, InterruptedTracker()
    )
    assert run_commands(monkeypatch, reviewer, ["prop forward 3", "undo", "q"]) is False
    assert frames == before
    assert not Path(f"masks/videos/{video['Video ID']}/000001.npz").exists()


def test_undo_remains_available_after_completing_video(tmp_path, monkeypatch, capsys):
    video, frames = prepared(tmp_path, monkeypatch)
    for index in range(4):
        seed_frame(tmp_path, video, frames, index)
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    assert run_commands(monkeypatch, reviewer, ["accept", "undo", "q"]) is False
    assert all(row["Review Status"] == "pending" for row in frames)
    assert "Video complete. Use undo" in capsys.readouterr().out


def test_undo_is_single_level_and_releases_backups(tmp_path, monkeypatch, capsys):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0)
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    directories = []

    def reply(_):
        if reviewer._undo is not None:
            directories.append(Path(reviewer._undo.directory.name))
        return next(commands)

    commands = iter(["undo", "v partial", "v severe", "undo", "undo", "q"])
    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    monkeypatch.setattr("builtins.input", reply)
    assert reviewer.run() is False
    assert frames[0]["Visibility"] == "partial"
    assert all(not path.exists() for path in directories)
    assert capsys.readouterr().out.count("Nothing to undo") == 2


@pytest.mark.parametrize("kind", ["legacy", "missing", "changed"])
def test_undo_reacceptance_restores_unverified_acceptance(tmp_path, monkeypatch, kind):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0, "accepted")
    if kind == "missing":
        Path("video_review.jsonl").unlink()
    log = VideoLog(Path("video_review.jsonl"))
    if kind == "changed":
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
        details = {"action": "skip"} if kind == "missing" else {}
        log.append(frames[0]["Frame ID"], "accepted", **details)
    before = log.accepted_versions.copy()
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    assert run_commands(monkeypatch, reviewer, ["accept", "undo", "q"]) is False
    restored = VideoLog(Path("video_review.jsonl"))
    assert restored.accepted_versions == before
    assert not restored.acceptance_matches(tmp_path, video, frames[0])


@pytest.mark.parametrize(
    "command", ["region 1 1 10 10", "view full", "autopreview on", "stay"]
)
def test_undo_restores_view_settings(tmp_path, monkeypatch, command):
    video, frames = prepared(tmp_path, monkeypatch)
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    shown = []
    replies = iter([command, "undo", "q"])
    monkeypatch.setattr(
        video_annotation, "show_frame", lambda *args: shown.append(args)
    )
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert reviewer.run() is False
    assert shown[-1][2] == shown[0][2]
    assert shown[-1][4] == shown[0][4]
    assert reviewer.zoom is True
    assert reviewer.stay is False
    assert reviewer.args.auto_preview is False


def test_undo_text_prompt_restores_candidates_and_prompt(tmp_path, monkeypatch):
    video, frames = prepared(tmp_path, monkeypatch)
    backend = Tracker()
    prompts = []

    def text(image, prompt):
        prompts.append(prompt)
        return [Candidate(mask(), 1.0, prompt, "candidate")]

    backend.text = text
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, backend
    )
    shown = []
    replies = iter(["p", "p different", "undo", "p", "q"])
    monkeypatch.setattr(
        video_annotation, "show_frame", lambda *args: shown.append(args[3])
    )
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert reviewer.run() is False
    assert prompts[0] == prompts[2]
    assert prompts[1] == "different"
    assert shown[1][0] is shown[3][0]


@pytest.mark.parametrize("exit_command", ["q", "next-video"])
def test_leaving_video_cleans_up_undo_backups(tmp_path, monkeypatch, exit_command):
    video, frames = prepared(tmp_path, monkeypatch)
    seed_frame(tmp_path, video, frames, 0)
    reviewer = video_annotation.VideoReviewer(
        parse(monkeypatch, "videos"), video, frames, Tracker()
    )
    directories = []
    replies = iter(["v partial", exit_command])

    def reply(_):
        if reviewer._undo is not None:
            directories.append(Path(reviewer._undo.directory.name))
        return next(replies)

    monkeypatch.setattr(video_annotation, "show_frame", lambda *args: None)
    monkeypatch.setattr("builtins.input", reply)
    assert reviewer.run() is (exit_command == "next-video")
    assert directories and all(not path.exists() for path in directories)
    assert reviewer._undo is None
