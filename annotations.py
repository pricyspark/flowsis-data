from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path


IMAGE_ID_FIELD = "Image ID"
CAPTURE_SESSION_FIELD = "Capture Session ID"


def image_mask_path(mask_dir: Path, filepath: str) -> Path:
    """Mirror a manifest image path below the mask directory."""
    path = Path(filepath)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Image path must be relative without '..': {filepath}")
    if path.parts and path.parts[0] == "images":
        path = path.relative_to("images")
    return mask_dir / path.with_suffix(".npz")


IMAGE_OBJECT_FIELDS = (
    "Tool ID",
    "Tool Class",
    "Tool Name",
    "Confidence",
    "Sample ID",
    "Filepath",
    IMAGE_ID_FIELD,
    "Instance ID",
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


IMAGE_LEADING_FIELDS = ("Tool IDs", "Sample ID", "Angle", "Filepath", "Lights")


def numeric_sort_key(value: str) -> tuple[int, float, str]:
    """Sort numeric labels numerically, then text, with blanks last."""
    if not value.strip():
        return 2, 0, ""
    try:
        return 0, float(value), value
    except ValueError:
        return 1, 0, value.casefold()


def tool_id_summary(instances: Iterable[Mapping[str, object]]) -> str:
    ids = {str(obj.get("Tool ID", "")).strip() for obj in instances}
    ids.discard("")
    return ";".join(sorted(ids, key=numeric_sort_key))


def write_image_manifests(
    manifest: Path,
    images: Sequence[Mapping[str, str]],
    object_manifest: Path,
    objects: Sequence[Mapping[str, object]],
) -> None:
    """Refresh navigation columns without changing sample or angle assignments."""
    image_by_id = {row[IMAGE_ID_FIELD]: dict(row) for row in images}
    if len(image_by_id) != len(images):
        raise ValueError("Image IDs must be unique.")
    by_image: dict[str, list[dict[str, object]]] = {}
    instance_keys: set[tuple[str, str]] = set()
    for obj in objects:
        image_id = str(obj[IMAGE_ID_FIELD])
        key = (image_id, str(obj["Instance ID"]))
        if key in instance_keys:
            raise ValueError(f"Duplicate instance: {key}")
        instance_keys.add(key)
        if image_id not in image_by_id:
            raise ValueError(f"Instance refers to unknown image: {image_id}")
        image = image_by_id[image_id]
        by_image.setdefault(image_id, []).append({
            **obj,
            "Sample ID": image.get("Sample ID", ""),
            "Filepath": image["Filepath"],
        })
    for image_id, image in image_by_id.items():
        image["Tool IDs"] = tool_id_summary(by_image.get(image_id, []))
    ordered_images = sorted(
        image_by_id.values(),
        key=lambda row: (
            not row.get("Sample ID", "").strip() or not row.get("Angle", "").strip(),
            tuple(numeric_sort_key(tool_id) for tool_id in row["Tool IDs"].split(";")),
        ),
    )
    ordered_objects = [
        obj for image in ordered_images
        for obj in sorted(
            by_image.get(image[IMAGE_ID_FIELD], []),
            key=lambda obj: (
                numeric_sort_key(str(obj.get("Tool ID", ""))),
                str(obj["Instance ID"]),
            ),
        )
    ]
    fields = list(IMAGE_LEADING_FIELDS)
    for image in images:
        fields.extend(field for field in image if field not in fields)
    if not images and manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as file:
            fields.extend(
                field for field in next(csv.reader(file)) if field not in fields
            )
    write_csv(object_manifest, ordered_objects, IMAGE_OBJECT_FIELDS)
    write_csv(manifest, ordered_images, fields)
