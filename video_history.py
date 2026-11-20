"""One-command video undo with lazy file backups outside the dataset."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from image_review import Candidate
from video_data import AcceptedVersion, VideoLog


@dataclass
class VideoViewState:
    current: int | None
    prompt: str
    region: tuple[int, int, int, int] | None
    candidates: list[Candidate] = field(compare=False)
    visited: set[int]
    zoom: bool
    stay: bool
    auto_preview: bool
    click_history: dict[
        tuple[int, tuple[int, int, int, int] | None], list[tuple[float, float, int]]
    ]
    inspected: dict[int, str]
    candidate_ids: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.candidates = list(self.candidates)
        self.candidate_ids = tuple(id(candidate) for candidate in self.candidates)
        self.visited = self.visited.copy()
        self.click_history = {
            key: list(points) for key, points in self.click_history.items()
        }
        self.inspected = self.inspected.copy()


@dataclass
class FrameState:
    row: dict[str, str]
    before: dict[str, str]
    decision: str | None
    version: AcceptedVersion | None


def file_signature(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class VideoCommandUndo:
    def __init__(self, command: str, view: VideoViewState, log: VideoLog) -> None:
        self.command = command
        self.view = view
        self.log = log
        self.rows: dict[str, FrameState] = {}
        self.files: dict[
            Path, tuple[Path | None, tuple[int, int, int, int] | None]
        ] = {}
        self.directory = tempfile.TemporaryDirectory(prefix="flowsis-video-undo-")

    def record(self, row: dict[str, str], paths: tuple[Path, ...]) -> None:
        key = row["Frame ID"]
        if key not in self.rows:
            self.rows[key] = FrameState(
                row,
                row.copy(),
                self.log.decisions.get(key),
                self.log.accepted_versions.get(key),
            )
        for path in paths:
            path = path.resolve()
            if path in self.files:
                continue
            signature = file_signature(path)
            backup = None
            if signature is not None:
                backup = Path(self.directory.name) / str(len(self.files))
                shutil.copyfile(path, backup)
            self.files[path] = backup, signature

    def changed(self, view: VideoViewState) -> bool:
        return (
            view != self.view
            or any(
                state.row != state.before
                or self.log.decisions.get(key) != state.decision
                or self.log.accepted_versions.get(key) != state.version
                for key, state in self.rows.items()
            )
            or any(
                file_signature(path) != signature
                for path, (_, signature) in self.files.items()
            )
        )

    def restore(self) -> VideoViewState:
        for path, (backup, signature) in self.files.items():
            if file_signature(path) == signature:
                continue
            if backup is None:
                path.unlink(missing_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
            try:
                shutil.copyfile(backup, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        for key, state in self.rows.items():
            state.row.clear()
            state.row.update(state.before)
            self.log.restore_decision(key, state.decision, state.version, self.command)
        return self.view

    def close(self) -> None:
        self.directory.cleanup()
