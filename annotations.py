from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from pathlib import Path


IMAGE_ID_FIELD = "Image ID"
CAPTURE_SESSION_FIELD = "Capture Session ID"
IMAGE_OBJECT_FIELDS = (
    IMAGE_ID_FIELD,
    "Instance ID",
    "Tool ID",
    "Tool Class",
    "Tool Name",
    "BBox X",
    "BBox Y",
    "BBox Width",
    "BBox Height",
    "Mask Path",
    "Mask Index",
    "Visibility",
    "Reviewed",
    "Annotation Complete",
    "Mask Valid",
    "Class Valid",
    "Base Valid",
    "Visible",
    "Annotation Confidence",
)
VIDEO_MANIFEST_FIELDS = (
    "Video ID",
    "Filepath",
    "FPS",
    "Frame Count",
    "Width",
    "Height",
    CAPTURE_SESSION_FIELD,
    "Condition",
)
VIDEO_OBJECT_FIELDS = (
    "Video ID",
    "Frame Index",
    "Track ID",
    "Tool ID",
    "Tool Class",
    "Tool Name",
    "BBox X",
    "BBox Y",
    "BBox Width",
    "BBox Height",
    "Mask Path",
    "Mask Index",
    "Visibility",
    "Reviewed",
    "Annotation Complete",
    "Mask Valid",
    "Class Valid",
    "Base Valid",
    "Visible",
    "Annotation Confidence",
    "Preferred Track",
)


def parse_bool(value: object, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value {value!r}.")


def stable_media_id(kind: str, filepath: str) -> str:
    normalized = Path(filepath).as_posix()
    digest = hashlib.sha256(f"{kind}\0{normalized}".encode()).hexdigest()[:16]
    return f"{kind}-{digest}"


def load_taxonomy_aliases(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key).casefold(): str(value) for key, value in raw.items()}


def normalize_tool_name(value: str, aliases: Mapping[str, str]) -> str:
    normalized = " ".join(value.strip().split())
    return aliases.get(normalized.casefold(), normalized)


def canonical_tool_label(
    tool_name: str,
    tool_class: str,
    aliases: Mapping[str, str],
) -> str:
    """Combine a descriptive tool name and its class into one model label."""
    name = normalize_tool_name(tool_name, aliases)
    coarse = " ".join(tool_class.strip().split())
    if not name:
        return coarse
    if not coarse or name.casefold() == coarse.casefold():
        return name
    if name.casefold().endswith(f" {coarse.casefold()}"):
        return name
    return f"{name} {coarse}"


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def assign_capture_sessions(
    rows: list[dict[str, str]],
    *,
    maximum_gap: timedelta = timedelta(hours=2),
) -> list[str]:
    """Suggest capture sessions without changing any source manifest fields."""
    indexed = list(enumerate(rows))
    indexed.sort(
        key=lambda item: (
            item[1].get("Make", ""),
            item[1].get("Model", ""),
            _parse_timestamp(item[1].get("Time Taken", "")) or datetime.max,
            item[0],
        )
    )
    sessions = [""] * len(rows)
    previous_key: tuple[str, str] | None = None
    previous_time: datetime | None = None
    session_number = 0
    for original_index, row in indexed:
        key = (row.get("Make", ""), row.get("Model", ""))
        timestamp = _parse_timestamp(row.get("Time Taken", ""))
        starts_new = (
            previous_key != key
            or timestamp is None
            or previous_time is None
            or timestamp - previous_time > maximum_gap
            or timestamp.date() != previous_time.date()
        )
        if starts_new:
            session_number += 1
        date = timestamp.date().isoformat() if timestamp is not None else "unknown"
        sessions[original_index] = f"session-{date}-{session_number:03d}"
        previous_key = key
        previous_time = timestamp
    return sessions


def extend_image_manifest(
    source: str | Path,
    destination: str | Path | None = None,
) -> Path:
    """Append stable IDs while preserving every existing field and value."""
    source = Path(source)
    destination = source if destination is None else Path(destination)
    with source.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        original_fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not original_fields:
        raise ValueError(f"Image manifest has no header: {source}")
    paths = [row.get("Filepath", "") for row in rows]
    if any(not value for value in paths):
        raise ValueError("Every image-manifest row must contain Filepath.")
    if len(set(paths)) != len(paths):
        raise ValueError("Image-manifest Filepath values must be unique.")

    session_suggestions = assign_capture_sessions(rows)
    for row, session in zip(rows, session_suggestions, strict=True):
        if not row.get(IMAGE_ID_FIELD):
            row[IMAGE_ID_FIELD] = stable_media_id("image", row["Filepath"])
        if not row.get(CAPTURE_SESSION_FIELD):
            row[CAPTURE_SESSION_FIELD] = session
    fields = original_fields + tuple(
        field
        for field in (IMAGE_ID_FIELD, CAPTURE_SESSION_FIELD)
        if field not in original_fields
    )
    lines: list[str] = []
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    lines.append(buffer.getvalue())
    return atomic_write_text(destination, "".join(lines))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, object]],
    fields: Iterable[str],
) -> Path:
    from io import StringIO

    fields = tuple(fields)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return atomic_write_text(path, buffer.getvalue())



def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
