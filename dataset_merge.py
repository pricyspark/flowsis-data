"""Three-way dataset merging without choosing between competing edits."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath
from enum import Enum
from typing import cast

CSV_KEYS = {
    "image_manifest.csv": ("Image ID",),
    "image_object_manifest.csv": ("Image ID", "Instance ID"),
    "video_manifest.csv": ("Video ID",),
}


class Missing(Enum):
    VALUE = "missing"


MISSING = Missing.VALUE
type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[
    str, JSONValue
]
type Row = dict[str, str]
type Records = dict[tuple[str, ...], Row]


class Conflict(ValueError):
    """Changes that cannot be combined without a contributor's decision."""


def safe_path(path: str) -> str:
    parsed = PurePosixPath(path)
    if (
        not path
        or parsed.is_absolute()
        or ".." in parsed.parts
        or "\\" in path
        or ":" in path
        or parsed.as_posix() != path
    ):
        raise Conflict(f"Unsafe dataset path: {path!r}")
    return path


def choose[T](base: T, local: T, remote: T, location: str) -> T:
    if local == remote or remote == base:
        return local
    if local == base:
        return remote
    raise Conflict(
        f"{location}: both contributors changed the same value differently. "
        f"Local: {str(local)[:160]!r}; incoming: {str(remote)[:160]!r}."
    )


def merge_mapping[K, V](
    base: dict[K, V],
    local: dict[K, V],
    remote: dict[K, V],
    location: str,
    combine: Callable[[V, V, V, str], V] | None = None,
) -> dict[K, V]:
    result: dict[K, V] = {}
    for key in dict.fromkeys([*remote, *local, *base]):
        b, l, r = (mapping.get(key, MISSING) for mapping in (base, local, remote))
        value: V | Missing
        if (
            combine
            and not isinstance(b, Missing)
            and not isinstance(l, Missing)
            and not isinstance(r, Missing)
        ):
            value = combine(b, l, r, f"{location} / {key}")
        else:
            value = choose(b, l, r, f"{location} / {key}")
        if not isinstance(value, Missing):
            result[key] = value
    return result


def merge_json_value(
    base: JSONValue, local: JSONValue, remote: JSONValue, location: str
) -> JSONValue:
    if isinstance(base, dict) and isinstance(local, dict) and isinstance(remote, dict):
        return merge_mapping(base, local, remote, location, merge_json_value)
    return choose(base, local, remote, location)


def csv_rows(
    content: bytes, keys: tuple[str, ...], location: str
) -> tuple[list[str], Records]:
    if not content.strip():
        return [], {}
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    fields = list(reader.fieldnames or [])
    if not fields or len(set(fields)) != len(fields) or not set(keys) <= set(fields):
        raise Conflict(
            f"{location}: missing or duplicate CSV columns ({', '.join(keys)} required)."
        )
    rows: Records = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise Conflict(f"{location}: malformed CSV row {reader.line_num}.")
        key = tuple(row[field] for field in keys)
        if any(not value.strip() for value in key) or key in rows:
            raise Conflict(f"{location}: blank or duplicate record identity {key}.")
        rows[key] = {field: row[field] for field in fields}
    return fields, rows


def merge_csv(path: str, base: bytes | None, local: bytes, remote: bytes) -> bytes:
    keys = CSV_KEYS[path]
    lf, lr = csv_rows(local, keys, path)
    rf, rr = csv_rows(remote, keys, path)
    bf, br = csv_rows(base, keys, path) if base is not None else ([], {})
    # Column deletion/renaming needs a schema migration, not a silent empty value.
    if not set(bf) <= set(lf) or not set(bf) <= set(rf):
        raise Conflict(f"{path}: a contributor removed or renamed columns.")
    fields = list(dict.fromkeys([*rf, *lf]))
    merged = merge_mapping(
        br, lr, rr, path, lambda b, l, r, where: merge_mapping(b, l, r, where)
    )
    if path == "image_manifest.csv":

        def groups(rows: Records) -> dict[str, set[tuple[str, ...]]]:
            result: dict[str, set[tuple[str, ...]]] = defaultdict(set)
            for key, row in rows.items():
                if row.get("Sample ID", "").strip():
                    result[row["Sample ID"]].add(key)
            return result

        bg, lg, rg = (groups(rows) for rows in (br, lr, rr))
        for sample in lg.keys() & rg.keys():
            if (
                lg[sample] != bg[sample]
                and rg[sample] != bg[sample]
                and lg[sample] != rg[sample]
            ):
                raise Conflict(
                    f"Sample ID {sample}: both contributors changed its image group. Coordinate an unused Sample ID and rerun."
                )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {field: row.get(field, "") for field in fields} for row in merged.values()
    )
    return output.getvalue().encode()


def merge_jsonl(path: str, base: bytes | None, local: bytes, remote: bytes) -> bytes:
    def records(content: bytes | None) -> dict[str, JSONValue]:
        result: dict[str, JSONValue] = {}
        for line in (content or b"").decode("utf-8-sig").splitlines():
            if line.strip():
                record = json.loads(line)
                result[json.dumps(record, sort_keys=True, ensure_ascii=False)] = record
        return result

    b, l, r = (records(content) for content in (base, local, remote))
    # Logs are append-only. Rewriting old events needs an explicit decision.
    if b.keys() - l.keys() or b.keys() - r.keys():
        raise Conflict(f"{path}: existing review events were removed or rewritten.")
    return (
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in (r | l).values()
        )
    ).encode()


def merge_content(path: str, base: bytes | None, local: bytes, remote: bytes) -> bytes:
    try:
        if path in CSV_KEYS:
            return merge_csv(path, base, local, remote)
        if path.endswith(".jsonl"):
            return merge_jsonl(path, base, local, remote)
        if path.endswith(".json"):
            b, l, r = (
                cast(JSONValue, json.loads(content)) if content is not None else {}
                for content in (base, local, remote)
            )
            return (
                json.dumps(
                    merge_json_value(b, l, r, path), indent=2, ensure_ascii=False
                )
                + "\n"
            ).encode()

    except (UnicodeError, json.JSONDecodeError, csv.Error) as error:
        raise Conflict(f"{path}: could not read structured data: {error}") from error
    raise Conflict(
        f"{path}: different file contents were supplied at the same path. Rename a new capture or choose the intended edit, then rerun."
    )


def validate_references(
    files: Mapping[str, bytes | str], read: Callable[[str], bytes]
) -> None:
    """Check identities and references; allow existing incomplete annotation groups."""
    video_data = read("video_manifest.csv") if "video_manifest.csv" in files else b""
    if video_data.strip():
        fields, videos = csv_rows(
            video_data,
            CSV_KEYS["video_manifest.csv"],
            "video_manifest.csv",
        )
        if "Filepath" not in fields:
            raise Conflict("Video manifest is missing its Filepath column.")
        video_paths = [safe_path(row["Filepath"]) for row in videos.values()]
        if len(set(video_paths)) != len(video_paths) or any(
            path not in files for path in video_paths
        ):
            raise Conflict("Video manifest contains duplicate or missing video files.")
    if "image_manifest.csv" not in files:
        return
    fields, images = csv_rows(
        read("image_manifest.csv"), CSV_KEYS["image_manifest.csv"], "image_manifest.csv"
    )
    if not {"Filepath", "Sample ID", "Tool IDs"} <= set(fields):
        raise Conflict(
            "Image manifest is missing Filepath, Sample ID, or Tool IDs columns."
        )
    paths = set()
    for row in images.values():
        path = safe_path(row["Filepath"])
        if path in paths or path not in files:
            raise Conflict(f"Image manifest: duplicate or missing image file {path}.")
        paths.add(path)
    if "image_object_manifest.csv" not in files:
        raise Conflict("Image object manifest is missing.")
    fields, objects = csv_rows(
        read("image_object_manifest.csv"),
        CSV_KEYS["image_object_manifest.csv"],
        "image_object_manifest.csv",
    )
    if not {"Filepath", "Sample ID", "Tool ID"} <= set(fields):
        raise Conflict(
            "Object manifest is missing Filepath, Sample ID, or Tool ID columns."
        )
    by_image: dict[str, list[Row]] = defaultdict(list)
    for (image_id, _), row in objects.items():
        image = images.get((image_id,))
        if image is None:
            raise Conflict(f"Object refers to missing Image ID {image_id}.")
        for field in ("Sample ID", "Filepath"):
            if row.get(field, "") != image.get(field, ""):
                raise Conflict(
                    f"Image {image_id}: object {field} disagrees with the image manifest."
                )
        mask = row.get("Mask Path", "").strip()
        if mask and safe_path(mask) not in files:
            raise Conflict(f"Image {image_id}: mask file {mask} is missing.")
        by_image[image_id].append(row)
    for (image_id,), row in images.items():
        instances = by_image[image_id]
        tools = {obj["Tool ID"] for obj in instances}
        declared = set(filter(None, row.get("Tool IDs", "").split(";")))
        if not instances or tools != declared or len(tools) != len(instances):
            raise Conflict(
                f"Image {image_id}: missing/duplicate tools or inconsistent object declarations."
            )
