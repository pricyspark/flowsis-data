"""Metadata-first DVC synchronization, selective pulls, and explicit deletions."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from dvc.utils.serialize import load_yaml
from filelock import FileLock, Timeout

from dataset_merge import (
    CSV_KEYS,
    Conflict,
    MISSING,
    Missing,
    choose,
    merge_content,
    validate_references,
)
from dataset_storage import (
    DatasetStore,
    SyncError,
    cache_file,
    open_store,
    pointer_hash,
    read_index,
)
from dataset_workspace import (
    FileRecord,
    WorkspaceState,
    finish_apply,
    data_path,
    load_state,
    normalize_target,
    save_state,
    scan,
    select,
    stage_apply,
    verify_scan,
    within,
    write_json,
)

ROOT = Path(__file__).resolve().parent
WRAPPER = ROOT / "dvc_gdrive_sync.py"


def run(
    args: list[str], cwd: Path, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if check and result.returncode:
        raise SyncError(
            f"Command failed: {' '.join(args[:3])}\n{result.stdout.strip()}"
        )
    return result


def git(cwd: Path, *args: str, check: bool = True) -> str:
    return run(["git", *args], cwd, check=check).stdout.strip()


def dvc(cwd: Path, *args: str) -> str:
    command = [sys.executable, str(WRAPPER), *args]
    if args[0] == "push":
        result = subprocess.run(command, cwd=cwd)
        if result.returncode:
            raise SyncError("DVC push failed. Git publication was stopped.")
        return ""
    return run(command, cwd).stdout.strip()


def index(pointer: Path, cache: Path) -> dict[str, str]:
    path = cache_file(cache, pointer_hash(pointer.read_bytes()))
    if not path.is_file():
        raise SyncError(f"Dataset index is missing: {path.name}")
    return read_index(path)


def merge_dataset(
    base: dict[str, str],
    local: dict[str, str],
    remote: dict[str, str],
    cache: Path,
    store: DatasetStore | None = None,
    *,
    validate: bool = True,
) -> dict[str, str | bytes]:
    def read(digest: str) -> bytes:
        return (store.get(digest) if store else cache_file(cache, digest)).read_bytes()

    result: dict[str, str | bytes] = {}
    for path in sorted(base.keys() | local.keys() | remote.keys()):
        b, l, r = (mapping.get(path, MISSING) for mapping in (base, local, remote))
        try:
            value = choose(b, l, r, path)
        except Conflict:
            if isinstance(l, str) and isinstance(r, str):
                if path not in CSV_KEYS and Path(path).suffix not in {
                    ".json",
                    ".jsonl",
                }:
                    raise Conflict(
                        f"{path}: different file contents were supplied at the same path. Choose the intended edit or rename the new capture."
                    ) from None
                value = merge_content(
                    path, read(b) if isinstance(b, str) else None, read(l), read(r)
                )
            else:
                raise Conflict(
                    f"{path}: a file was deleted on one side and edited on the other."
                ) from None
        if not isinstance(value, Missing):
            result[path] = value
    for path in result:
        if any(parent.as_posix() in result for parent in Path(path).parents):
            raise Conflict(
                f"{path}: another contributor used a parent directory as a file."
            )

    def contents(path: str) -> bytes:
        value = result[path]
        return value if isinstance(value, bytes) else read(value)

    if validate:
        validate_references(result, contents)
    return result


def cached_results(
    store: DatasetStore, files: dict[str, str | bytes]
) -> dict[str, str]:
    return {
        path: store.put_bytes(value) if isinstance(value, bytes) else value
        for path, value in files.items()
    }


def check_workspace(root: Path) -> None:
    if git(root, "diff", "--cached", "--name-only"):
        raise SyncError(
            "Files are already staged in Git. Commit or unstage them before synchronizing."
        )
    if any(
        path != "data.dvc" for path in git(root, "diff", "--name-only").splitlines()
    ):
        raise SyncError(
            "There are uncommitted code/documentation changes. Commit or save those separately before synchronizing."
        )
    for state in (
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "rebase-merge",
        "rebase-apply",
    ):
        path = Path(git(root, "rev-parse", "--git-path", state))
        if not path.is_absolute():
            path = root / path
        if path.exists():
            raise SyncError(
                "An earlier Git merge/rebase is unfinished. Finish or abort it first."
            )


def check_pending(root: Path) -> None:
    if (root / ".data-sync/apply.pending").exists():
        raise SyncError(
            "An earlier operation was interrupted while updating this machine. Run python sync_data.py recover. Do not pull DVC over the current data."
        )


def new_session(root: Path) -> Path:
    path = (
        root
        / ".data-sync/sessions"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    path.mkdir(parents=True)
    return path


def local_snapshot(
    store: DatasetStore,
    *,
    validate: bool = True,
) -> tuple[WorkspaceState, dict[str, FileRecord], dict[str, str]]:
    root = store.root
    pointer = (root / "data.dvc").read_bytes()
    managed = (root / ".data-sync/workspace.json").exists()
    state = load_state(root, pointer)
    adopting_revision = "git_revision" not in state
    if adopting_revision:
        state["git_revision"] = git(root, "rev-parse", "HEAD")
    base = store.index(state["pointer"].encode())
    current = store.index(pointer)
    observed = scan(store, state["materialized"])
    missing = state["materialized"].keys() - observed.keys() - state["deletions"].keys()
    if missing:
        raise SyncError(
            f"Previously downloaded files disappeared, starting with {sorted(missing)[0]}. Use 'remove' for dataset deletion, 'evict' for local removal, or restore the files."
        )
    local = {**base, **{name: record["md5"] for name, record in observed.items()}}
    for name in state["deletions"]:
        if name in observed:
            raise SyncError(
                f"{name} is marked for removal but exists again. Resolve the pending deletion first."
            )
        local.pop(name, None)
    merge_base = base
    if managed and pointer != state["pointer"].encode():
        head_pointer = git(root, "show", "HEAD:data.dvc").encode() + b"\n"
        if pointer != head_pointer and (
            base.keys() - current.keys() - state["deletions"].keys()
        ):
            raise SyncError(
                "An external DVC snapshot dropped tracked paths. Do not run 'dvc add data' on a partial checkout; restore its pointer and use explicit remove commands."
            )
        if pointer == head_pointer:
            assert "git_revision" in state
            public_revision = state["git_revision"]
            remote_revision = state.get("remote_revision")
            if (
                remote_revision
                and run(
                    ["git", "merge-base", "--is-ancestor", remote_revision, "HEAD"],
                    root,
                    check=False,
                ).returncode
                == 0
            ):
                public_revision = remote_revision
            merge_base = store.index(
                git(root, "show", f"{public_revision}:data.dvc").encode() + b"\n"
            )
    local = cached_results(
        store,
        merge_dataset(
            merge_base, local, current, store.cache, store, validate=validate
        ),
    )
    if not managed:
        state["materialized"] = observed
    if not managed or adopting_revision:
        save_state(root, state)
    return state, observed, local


def prepare_clone(
    root: Path, work: Path, head: str, remote_url: str, cache: Path
) -> None:
    git(root, "clone", "--shared", "--no-checkout", str(root), str(work))
    git(work, "checkout", "-b", "dataset-sync", head)
    git(work, "remote", "set-url", "origin", remote_url)
    for key in ("user.name", "user.email"):
        value = git(root, "config", "--get", key, check=False)
        if value:
            git(work, "config", key, value)
    source = root / ".dvc/config.local"
    if source.exists():
        shutil.copy2(source, work / ".dvc/config.local")
    for key, value in (
        ("cache.dir", str(cache)),
        ("core.autostage", "false"),
        ("core.site_cache_dir", str(root / ".data-sync/dvc-state")),
    ):
        dvc(work, "config", "--local", key, value)


def upload_snapshots(work: Path, incoming: str, store: DatasetStore) -> None:
    """Publish all newly referenced objects without fetching missing media."""
    pointer = work / "data.dvc"
    final = pointer.read_bytes()
    snapshots = {final}
    for revision in git(work, "rev-list", "HEAD", f"^{incoming}").splitlines():
        snapshots.add(git(work, "show", f"{revision}:data.dvc").encode() + b"\n")
    try:
        for content in snapshots:
            files = store.index(content)
            absent = store.remote_missing({pointer_hash(content), *files.values()})
            if any(not cache_file(store.cache, digest).is_file() for digest in absent):
                raise SyncError(
                    "Some published objects are unavailable locally and remotely. Ask their contributor to run python dvc_gdrive_sync.py push. Git publication was stopped."
                )
            pointer.write_bytes(content)
            dvc(work, "push", "data.dvc")
            store.verify_remote(files, content)
    finally:
        pointer.write_bytes(final)


def recover(root: Path) -> None:
    pending = root / ".data-sync/apply.pending"
    if not pending.exists():
        print("No interrupted update needs recovery.")
        return
    try:
        plan = json.loads(pending.read_text())
    except json.JSONDecodeError as error:
        raise SyncError(
            "This is a legacy recovery journal. Preserve it for manual inspection."
        ) from error
    if "git_before" in plan:
        check_workspace(root)
        head = git(root, "rev-parse", "HEAD")
        if head not in {plan["git_before"], plan["git_after"]}:
            raise SyncError(
                "Git changed since interruption; recovery needs manual review."
            )
        head_pointer = git(root, "show", "HEAD:data.dvc").encode() + b"\n"
        if (root / "data.dvc").read_bytes() not in {
            plan["original_pointer"].encode(),
            plan["pointer"].encode(),
            head_pointer,
        }:
            raise SyncError(
                "data.dvc changed since interruption; preserve it before recovering."
            )
        if head != plan["git_after"]:
            (root / "data.dvc").write_bytes(head_pointer)
            git(root, "merge", "--ff-only", plan["git_after"])
        (root / "data.dvc").write_text(plan["pointer"])
    finish_apply(root, plan)


def sync(
    root: Path,
    message: str | None,
    retries: int = 3,
    *,
    publish: bool = True,
    targets: list[str] | None = None,
    preset: str | None = None,
) -> None:
    check_pending(root)
    check_workspace(root)
    branch = git(root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    remote = git(root, "config", "--get", f"branch.{branch}.remote", check=False)
    remote_ref = git(root, "config", "--get", f"branch.{branch}.merge", check=False)
    if (
        not branch
        or not remote
        or remote == "."
        or not remote_ref.startswith("refs/heads/")
    ):
        raise SyncError("Use a normal branch with a configured remote tracking branch.")
    head = git(root, "rev-parse", "HEAD")
    original_pointer = (root / "data.dvc").read_bytes()
    session = new_session(root)
    print(f"Recovery session: {session.relative_to(root)}", flush=True)
    with open_store(root) as store:
        state, observed, local = local_snapshot(store)
        select(state, targets or [], preset)
        (session / "local.dvc").write_bytes(store.pointer(local, original_pointer))

        def verify_local() -> None:
            check_workspace(root)
            if (
                git(root, "rev-parse", "HEAD") != head
                or (root / "data.dvc").read_bytes() != original_pointer
            ):
                raise SyncError(
                    "Git changed during synchronization. Your data is preserved; rerun."
                )
            verify_scan(store, observed)

        for attempt in range(1, retries + 1):
            git(root, "fetch", remote, remote_ref)
            incoming = git(root, "rev-parse", "FETCH_HEAD")
            ancestors = git(root, "merge-base", "--all", head, incoming).splitlines()
            if len(ancestors) != 1:
                raise SyncError(
                    "The branches have an ambiguous common ancestor; review Git history first."
                )
            incoming_pointer = (
                git(root, "show", f"{incoming}:data.dvc").encode() + b"\n"
            )
            base = store.index(
                git(root, "show", f"{ancestors[0]}:data.dvc").encode() + b"\n"
            )
            theirs = store.index(incoming_pointer)
            remote_revision = state.get("remote_revision")
            if (
                remote_revision
                and run(
                    ["git", "merge-base", "--is-ancestor", remote_revision, incoming],
                    root,
                    check=False,
                ).returncode
                == 0
                and run(
                    [
                        "git",
                        "merge-base",
                        "--is-ancestor",
                        ancestors[0],
                        remote_revision,
                    ],
                    root,
                    check=False,
                ).returncode
                == 0
            ):
                base = store.index(
                    git(root, "show", f"{remote_revision}:data.dvc").encode() + b"\n"
                )
            store.verify_remote(theirs, incoming_pointer)
            combined = cached_results(
                store, merge_dataset(base, local, theirs, store.cache, store)
            )
            for target in targets or []:
                prefix = normalize_target(target)
                if (
                    prefix
                    and not any(within(name, prefix) for name in combined)
                    and not data_path(root, prefix).is_dir()
                ):
                    raise SyncError(f"Requested dataset path is not tracked: {target}")
            pointer = store.pointer(combined, incoming_pointer)
            next_state = deepcopy(state)
            next_state["pointer"] = pointer.decode()
            if publish:
                next_state["deletions"] = {}
            attempt_dir = session / f"attempt-{attempt}"
            attempt_dir.mkdir()
            plan = stage_apply(store, combined, observed, next_state, attempt_dir)
            fast_forward = git(root, "merge-base", head, incoming) == head
            after = incoming if fast_forward else head
            work = attempt_dir / "git"
            try:
                if publish:
                    prepare_clone(
                        root,
                        work,
                        head,
                        git(root, "remote", "get-url", "--push", remote),
                        store.cache,
                    )
                    if fast_forward:
                        git(work, "reset", "--hard", incoming)
                    elif head != incoming:
                        result = run(
                            ["git", "merge", "--no-ff", "--no-commit", incoming],
                            work,
                            check=False,
                        )
                        conflicts = git(
                            work, "diff", "--name-only", "--diff-filter=U"
                        ).splitlines()
                        if any(path != "data.dvc" for path in conflicts):
                            raise Conflict(
                                "Code changes also conflict: " + ", ".join(conflicts)
                            )
                        if result.returncode and not conflicts:
                            raise SyncError(result.stdout)
                    (work / "data.dvc").write_bytes(pointer)
                    git(work, "add", "--", "data.dvc")
                    if (
                        git(work, "diff", "--cached", "--name-only")
                        or (work / ".git/MERGE_HEAD").exists()
                    ):
                        if message is None:
                            if not sys.stdin.isatty():
                                raise SyncError(
                                    'Supply a message with -m "Name: dataset changes".'
                                )
                            message = input("Describe your changes: ").strip()
                        if not message:
                            raise SyncError("A nonempty commit message is required.")
                        git(work, "commit", "-m", f"[data-sync] {message}")
                    verify_local()
                    upload_snapshots(work, incoming, store)
                    verify_local()
                    after = git(work, "rev-parse", "HEAD")
                    if after != incoming:
                        pushed = run(
                            ["git", "push", "origin", f"HEAD:{remote_ref}"],
                            work,
                            check=False,
                        )
                        if pushed.returncode:
                            git(root, "fetch", remote, remote_ref)
                            if (
                                git(root, "rev-parse", "FETCH_HEAD") != incoming
                                and attempt < retries
                            ):
                                print(
                                    "Another contributor published first; merging and retrying...",
                                    flush=True,
                                )
                                continue
                            raise SyncError(
                                "Git publication failed; your data is preserved.\n"
                                + pushed.stdout
                            )
                    git(root, "fetch", str(work), "HEAD")
                else:
                    git(root, "fetch", remote, remote_ref)
                    if git(root, "rev-parse", "FETCH_HEAD") != incoming:
                        continue
                verify_local()
                next_state["git_revision"] = after
                next_state["remote_revision"] = after if publish else incoming
                plan.update(
                    {
                        "git_before": head,
                        "git_after": after,
                        "pointer": pointer.decode(),
                        "original_pointer": original_pointer.decode(),
                    }
                )
                write_json(root / ".data-sync/apply.pending", plan)
                recover(root)
                print(
                    (
                        "Synchronization complete."
                        if publish
                        else "Pull complete; nothing was uploaded or committed."
                    ),
                    flush=True,
                )
                return
            finally:
                (work / ".dvc/config.local").unlink(missing_ok=True)
    raise SyncError(
        "Contributors kept publishing; your data is preserved. Retry shortly."
    )


def main() -> int:
    from dataset_commands import cleanup, remove_or_evict, restore_removed, status
    from dataset_workspace import PRESETS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="sync",
        choices=(
            "sync",
            "pull",
            "status",
            "remove",
            "evict",
            "restore",
            "recover",
            "cleanup",
        ),
    )
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--preset", choices=tuple(PRESETS))
    parser.add_argument(
        "-m", "--message", help="commit message; prefixed with [data-sync]"
    )
    args = parser.parse_intermixed_args()
    try:
        (ROOT / ".data-sync").mkdir(exist_ok=True)
        with FileLock(ROOT / ".data-sync/sync.lock", timeout=0):
            if args.command in {"sync", "pull"}:
                sync(
                    ROOT,
                    args.message,
                    publish=args.command == "sync",
                    targets=args.targets,
                    preset=args.preset,
                )
            elif args.command == "status":
                status(ROOT)
            elif args.command in {"remove", "evict"}:
                if not args.targets:
                    parser.error("remove/evict requires at least one dataset path")
                remove_or_evict(ROOT, args.targets, evict=args.command == "evict")
            elif args.command == "recover":
                recover(ROOT)
            elif args.command == "restore":
                if not args.targets:
                    parser.error("restore requires at least one pending-removal path")
                restore_removed(ROOT, args.targets)
            else:
                if len(args.targets) != 1:
                    parser.error("cleanup requires one completed session name")
                cleanup(ROOT, args.targets[0])
    except Timeout:
        print(
            "Another dataset operation is running. Wait for it to finish.",
            file=sys.stderr,
        )
        return 1
    except (SyncError, Conflict, OSError, ValueError, KeyboardInterrupt) as error:
        print(
            f"Synchronization stopped: {error}\nYour local data and recovery files are preserved.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
