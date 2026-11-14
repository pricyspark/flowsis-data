"""Local inspection, explicit removals/eviction, and recovery cleanup."""

from __future__ import annotations

from pathlib import Path
import json
import shutil

from dataset_storage import SyncError, cache_file, checksum, open_store
from dataset_workspace import (
    METADATA_FILES,
    finish_apply,
    load_state,
    normalize_target,
    scan,
    skeleton,
    stage_apply,
    verify_scan,
    within,
    write_json,
)
import sync_data


def status(root: Path) -> None:
    sync_data.check_pending(root)
    with open_store(root) as store:
        state, observed, local = sync_data.local_snapshot(store, validate=False)
        tracked = store.index((root / "data.dvc").read_bytes())
        skeleton(root, local)
        print(
            json.dumps(
                {
                    "tracked_files": len(tracked),
                    "local_files": len(observed),
                    "not_downloaded": len(
                        tracked.keys() - observed.keys() - state["deletions"].keys()
                    ),
                    "added": sorted(local.keys() - tracked.keys()),
                    "modified": sorted(
                        name
                        for name in local.keys() & tracked.keys()
                        if local[name] != tracked[name]
                    ),
                    "removed": sorted(tracked.keys() - local.keys()),
                    "selections": state["selections"],
                },
                indent=2,
            )
        )


def remove_or_evict(root: Path, targets: list[str], *, evict: bool) -> None:
    sync_data.check_pending(root)
    prefixes = [normalize_target(target) for target in targets]
    with open_store(root) as store:
        # Explicit commands also resolve files already removed manually.
        state = load_state(root, (root / "data.dvc").read_bytes())
        tracked = store.index(state["pointer"].encode())
        observed = scan(store, state["materialized"])
        names = {
            name
            for name in tracked.keys() | observed.keys()
            if any(within(name, prefix) for prefix in prefixes)
        }
        if not names:
            raise SyncError("No dataset paths match those targets.")
        missing_elsewhere = (
            state["materialized"].keys()
            - observed.keys()
            - state["deletions"].keys()
            - names
        )
        if missing_elsewhere:
            raise SyncError(
                "Other downloaded files disappeared; resolve those with remove/evict first."
            )
        files = {
            **tracked,
            **{name: record["md5"] for name, record in observed.items()},
        }
        if evict:
            if names & METADATA_FILES:
                raise SyncError("The metadata working set cannot be evicted.")
            if any(
                name not in tracked
                or (name in observed and observed[name]["md5"] != tracked[name])
                for name in names
            ):
                raise SyncError(
                    "Eviction would discard unpublished edits. Synchronize them first."
                )
            if store.remote_missing({tracked[name] for name in names}):
                raise SyncError(
                    "Eviction would discard unpublished objects. Upload them first."
                )
            for prefix in prefixes:
                state["selections"].append({"path": prefix, "include": False})
        else:
            for name in names:
                state["deletions"][name] = files.pop(name)
        session = sync_data.new_session(root)
        plan = stage_apply(
            store,
            files,
            observed,
            state,
            session,
            evict=names if evict else None,
            materialized=set(observed) - names,
        )
        verify_scan(store, observed)
        write_json(root / ".data-sync/apply.pending", plan)
        finish_apply(root, plan)
        if evict:
            protected = {record["md5"] for record in state["materialized"].values()}
            if store.cache.resolve().is_relative_to(root.resolve() / ".dvc/cache"):
                for digest in {tracked[name] for name in names} - protected:
                    cache_file(store.cache, digest).unlink(missing_ok=True)
        print(
            f"{'Evicted locally' if evict else 'Marked for dataset deletion'}: {len(names)} files."
        )


def restore_removed(root: Path, targets: list[str]) -> None:
    sync_data.check_pending(root)
    prefixes = [normalize_target(target) for target in targets]
    with open_store(root) as store:
        state = load_state(root, (root / "data.dvc").read_bytes())
        names = {
            name
            for name in state["deletions"]
            if any(within(name, prefix) for prefix in prefixes)
        }
        if not names:
            raise SyncError("No pending dataset removals match those targets.")
        observed = scan(store, state["materialized"])
        if state["materialized"].keys() - observed.keys() - state["deletions"].keys():
            raise SyncError(
                "Other downloaded files disappeared; resolve those before restoring removals."
            )
        files = {
            **store.index(state["pointer"].encode()),
            **{name: record["md5"] for name, record in observed.items()},
        }
        for name in names:
            digest = state["deletions"][name]
            if name in observed and observed[name]["md5"] != digest:
                raise SyncError(
                    f"Restore would overwrite a new edit to {name}; preserve it first."
                )
            files[name] = digest
            del state["deletions"][name]
        session = sync_data.new_session(root)
        plan = stage_apply(
            store, files, observed, state, session, materialized=set(observed) | names
        )
        verify_scan(store, observed)
        write_json(root / ".data-sync/apply.pending", plan)
        finish_apply(root, plan)
        print(f"Restored {len(names)} pending removals; nothing was published.")


def cleanup(root: Path, name: str) -> None:
    sync_data.check_pending(root)
    if Path(name).name != name or name in {".", ".."}:
        raise SyncError("Supply a session name from .data-sync/sessions, not a path.")
    session = root / ".data-sync/sessions" / name
    if not (session / "complete.json").exists() and not any(
        session.glob("attempt-*/complete.json")
    ):
        raise SyncError("Only completed sessions can be cleaned up.")
    with open_store(root) as store:
        backups = [
            path
            for path in session.rglob("*")
            if path.is_file() and "data-before-sync" in path.parts
        ]
        if store.remote_missing({checksum(path) for path in backups}):
            raise SyncError(
                "Recovery files include unpublished content; keep this session."
            )
    shutil.rmtree(session)
    print(f"Removed completed recovery session {name}.")
