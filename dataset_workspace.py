"""Local materialization state, selection rules, and recoverable file updates."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
from typing import NotRequired, TypedDict, cast

from dataset_merge import safe_path
from dataset_storage import DatasetStore, SyncError, checksum

METADATA_FILES = {
    "image_manifest.csv",
    "image_object_manifest.csv",
    "image_review.jsonl",
    "image_orientation_corrections.json",
    "visual_review_log.json",
    "video_manifest.csv",
    "video_frame_manifest.csv",
    "video_review.jsonl",
    "video_manifest.numbers.json",
}
PRESETS = {
    "metadata": (),
    "images": ("images", "masks/images", "image_review_state"),
    "videos": ("videos", "frames", "masks/videos"),
    "raw-images": ("raw-images",),
    "raw-videos": ("raw-videos",),
    "all": ("",),
}
STANDARD_DIRS = (
    "raw-images",
    "images",
    "masks/images",
    "image_review_state",
    "raw-videos",
    "videos",
    "frames/videos",
    "masks/videos/anchors",
    "previews/videos",
)


class FileRecord(TypedDict):
    md5: str
    stat: list[int]


class Selection(TypedDict):
    path: str
    include: bool


class Operation(TypedDict):
    path: str
    before: str | None
    after: str | None


class WorkspaceState(TypedDict):
    version: int
    pointer: str
    materialized: dict[str, FileRecord]
    selections: list[Selection]
    deletions: dict[str, str]
    git_revision: NotRequired[str]
    remote_revision: NotRequired[str]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def normalize_target(value: str) -> str:
    value = value.rstrip("/")
    if value == "data":
        return ""
    if value.startswith("data/"):
        value = value[5:]
    return safe_path(value)


def within(name: str, prefix: str) -> bool:
    return not prefix or name == prefix or name.startswith(prefix + "/")


def data_path(root: Path, name: str) -> Path:
    path = root / "data" / safe_path(name)
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise SyncError(f"Dataset symlinks are not supported: {parent}")
    return path


def load_state(root: Path, pointer: bytes) -> WorkspaceState:
    path = root / ".data-sync/workspace.json"
    if not path.exists():
        return {
            "version": 1,
            "pointer": pointer.decode(),
            "materialized": {},
            "selections": [],
            "deletions": {},
        }
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or not isinstance(value.get("pointer"), str)
    ):
        raise SyncError(
            "Unsupported .data-sync/workspace.json; preserve it for recovery."
        )
    for key in ("materialized", "deletions"):
        if not isinstance(value.get(key), dict):
            raise SyncError(f"Invalid workspace state: {key}")
        for name in value[key]:
            safe_path(name)
    for key in ("git_revision", "remote_revision"):
        if key in value and (
            not isinstance(value[key], str)
            or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value[key])
        ):
            raise SyncError(f"Invalid workspace revision: {key}")
    if not isinstance(value.get("selections"), list):
        raise SyncError("Invalid workspace selections")
    for rule in value["selections"]:
        if (
            not isinstance(rule, dict)
            or not isinstance(rule.get("path"), str)
            or not isinstance(rule.get("include"), bool)
        ):
            raise SyncError("Invalid workspace selection rule")
        if rule["path"]:
            safe_path(rule["path"])
    for record in value["materialized"].values():
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("md5"), str)
            or not isinstance(record.get("stat"), list)
        ):
            raise SyncError("Invalid materialized-file record")
        if (
            not re.fullmatch(r"[0-9a-f]{32}", record["md5"])
            or len(record["stat"]) != 5
            or not all(type(item) is int for item in record["stat"])
        ):
            raise SyncError("Invalid materialized-file fingerprint")
    if not all(
        isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{32}", digest)
        for digest in value["deletions"].values()
    ):
        raise SyncError("Invalid pending deletion fingerprint")
    return cast(WorkspaceState, value)


def save_state(root: Path, state: WorkspaceState) -> None:
    write_json(root / ".data-sync/workspace.json", state)


def signature(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def scan(store: DatasetStore, previous: dict[str, FileRecord]) -> dict[str, FileRecord]:
    data = store.root / "data"
    if data.is_symlink():
        raise SyncError("data/ must not be a symlink")
    if not data.exists():
        return {}
    for path in data.rglob("*"):
        if path.is_symlink():
            raise SyncError(f"Dataset symlinks are not supported: {path}")
    records: dict[str, FileRecord] = {}
    for filename in store.repo.dvcignore.find(store.repo.fs, str(data)):
        path = Path(filename)
        name = path.relative_to(data).as_posix()
        path = data_path(store.root, name)
        before = signature(path)
        old = previous.get(name)
        digest = old["md5"] if old and old["stat"] == before else checksum(path)
        store.put_file(path, digest)
        if signature(path) != before:
            raise SyncError(
                f"File changed while saving it: {name}. Pause annotation and retry."
            )
        records[name] = {"md5": digest, "stat": before}
    return records


def verify_scan(store: DatasetStore, records: dict[str, FileRecord]) -> None:
    data = store.root / "data"
    actual = {}
    if data.exists():
        for filename in store.repo.dvcignore.find(store.repo.fs, str(data)):
            name = Path(filename).relative_to(data).as_posix()
            actual[name] = signature(data_path(store.root, name))
    if actual != {name: record["stat"] for name, record in records.items()}:
        raise SyncError(
            "Your data changed during synchronization. Pause annotation and retry; workspace files were not replaced."
        )


def select(state: WorkspaceState, targets: list[str], preset: str | None) -> None:
    prefixes = [normalize_target(target) for target in targets]
    if preset:
        prefixes.extend(PRESETS[preset])
    for prefix in prefixes:
        state["selections"].append({"path": prefix, "include": True})


def wanted(state: WorkspaceState, name: str) -> bool:
    if name in METADATA_FILES:
        return True
    selected = False
    for rule in state["selections"]:
        if within(name, rule["path"]):
            selected = rule["include"]
    return selected


def skeleton(root: Path, files: dict[str, str]) -> None:
    (root / "data").mkdir(exist_ok=True)
    directories: set[str] = set(STANDARD_DIRS)
    directories.update(
        parent.as_posix()
        for name in files
        for parent in Path(name).parents
        if parent.as_posix() != "."
    )
    for name in sorted(directories):
        if name not in files:
            data_path(root, name).mkdir(parents=True, exist_ok=True)


def stage_apply(
    store: DatasetStore,
    files: dict[str, str],
    observed: dict[str, FileRecord],
    state: WorkspaceState,
    session: Path,
    *,
    evict: set[str] | None = None,
    materialized: set[str] | None = None,
) -> dict[str, object]:
    desired = (
        materialized
        if materialized is not None
        else {name for name in files if name in observed or wanted(state, name)}
    )
    if evict:
        desired -= evict
    operations: list[Operation] = []
    for name in sorted(desired | observed.keys()):
        after = files.get(name) if name in desired else None
        before = observed.get(name, {}).get("md5")
        if after == before:
            continue
        operations.append({"path": name, "before": before, "after": after})
    store.prefetch(
        {
            operation["after"]
            for operation in operations
            if operation["after"] is not None
        }
    )
    for operation in operations:
        if operation["after"]:
            staged = session / "staged" / operation["path"]
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(store.get(operation["after"]), staged)
    operations.sort(
        key=lambda operation: (
            operation["after"] is not None,
            -len(Path(operation["path"]).parts),
        )
    )
    # Stat signatures are recorded only after the staged files have been installed.
    return {
        "session": str(session),
        "operations": operations,
        "files": files,
        "materialized": sorted(desired),
        "observed": observed,
        "state": state,
        "evict": bool(evict),
    }


def finish_apply(root: Path, plan: dict[str, object]) -> None:
    """Idempotent file transaction; the caller handles the Git portion first."""
    session = Path(cast(str, plan["session"]))
    if not session.resolve().is_relative_to((root / ".data-sync/sessions").resolve()):
        raise SyncError("Recovery session is outside .data-sync/sessions")
    files = cast(dict[str, str], plan["files"])
    for operation in cast(list[dict[str, str | None]], plan["operations"]):
        name = operation["path"]
        assert name is not None
        target = data_path(root, name)
        before, after = operation["before"], operation["after"]
        if target.is_dir():
            if before is not None:
                raise SyncError(f"Recovery would overwrite a directory: {name}")
            target.rmdir()
        actual = checksum(target) if target.is_file() else None
        if actual == after:
            continue
        backup = session / "data-before-sync" / name
        if actual != before and not (
            actual is None and backup.is_file() and checksum(backup) == before
        ):
            raise SyncError(
                f"Recovery would overwrite a new edit to {name}; preserve it and resolve manually."
            )
        if target.exists():
            if target.is_dir():
                raise SyncError(f"File/directory collision during recovery: {name}")
            backup.parent.mkdir(parents=True, exist_ok=True)
            target.replace(backup)
        if after:
            staged = session / "staged" / name
            if not staged.is_file() or checksum(staged) != after:
                raise SyncError(f"Recovery staging file is missing or corrupt: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(target)
    state = cast(WorkspaceState, plan["state"])
    observed = cast(dict[str, FileRecord], plan["observed"])
    updated = {
        operation["path"]
        for operation in cast(list[dict[str, str | None]], plan["operations"])
    }
    records: dict[str, FileRecord] = {}
    for name in cast(list[str], plan["materialized"]):
        target = data_path(root, name)
        stat = signature(target)
        old = observed.get(name)
        digest = (
            files[name]
            if name in updated
            else old["md5"] if old and old["stat"] == stat else checksum(target)
        )
        records[name] = {"md5": digest, "stat": stat}
    state["materialized"] = records
    save_state(root, state)
    skeleton(root, files)
    write_json(session / "complete.json", {"complete": True})
    (root / ".data-sync/apply.pending").unlink()
    if plan.get("evict"):
        shutil.rmtree(session / "data-before-sync", ignore_errors=True)
