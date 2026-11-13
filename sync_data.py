"""Synchronize the dataset safely: python sync_data.py [-m 'What changed']."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from dvc.utils.serialize import load_yaml
from filelock import FileLock, Timeout

from dataset_merge import (
    Missing,
    MISSING,
    Conflict,
    choose,
    merge_content,
    safe_path,
    validate_references,
)

ROOT = Path(__file__).resolve().parent
WRAPPER = ROOT / "dvc_gdrive_sync.py"


class SyncError(RuntimeError):
    pass


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
    if args[0] in {"fetch", "push"}:
        # Show download progress and OAuth login links while the command runs.
        result = subprocess.run(command, cwd=cwd)
        if result.returncode:
            raise SyncError(f"DVC {args[0]} failed. See the command output above.")
        return ""
    return run(command, cwd).stdout.strip()


def cache_file(cache: Path, digest: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}(?:\.dir)?", digest):
        raise SyncError(f"Unsupported DVC object hash: {digest}")
    return cache / "files/md5" / digest[:2] / digest[2:]


def index(pointer: Path, cache: Path) -> dict[str, str]:
    outputs = load_yaml(pointer).get("outs", [])
    if len(outputs) != 1 or outputs[0].get("path") != "data":
        raise SyncError("Expected data.dvc to track the single data/ directory.")
    digest = outputs[0].get("md5", "")
    if not digest.endswith(".dir"):
        raise SyncError("Expected an MD5 directory snapshot in data.dvc.")
    path = cache_file(cache, digest)
    if not path.is_file():
        raise SyncError(
            f"Dataset index {digest} is missing. Ask the contributor who published it to run python dvc_gdrive_sync.py push on their original machine."
        )
    result = {}
    for row in json.loads(path.read_text()):
        relative = safe_path(row["relpath"])
        cache_file(cache, row["md5"])
        if relative in result:
            raise SyncError(f"Repeated dataset path: {relative}")
        result[relative] = row["md5"]
    return result


def snapshot(root: Path, destination: Path) -> None:
    """Cache working data, saving its pointer without changing Git's working copy."""
    pointer = root / "data.dvc"
    original = pointer.read_bytes()
    config = root / ".dvc/config.local"
    original_config = config.read_bytes() if config.exists() else None
    try:
        # A contributor's autostage setting must not alter the real Git index.
        dvc(root, "config", "--local", "core.autostage", "false")
        dvc(root, "add", "data")
        destination.write_bytes(pointer.read_bytes())
    finally:
        pointer.write_bytes(original)
        if original_config is None:
            config.unlink(missing_ok=True)
        else:
            config.write_bytes(original_config)


def merge_dataset(
    base: dict[str, str], local: dict[str, str], remote: dict[str, str], cache: Path
) -> dict[str, str | bytes]:
    result: dict[str, str | bytes] = {}
    missing = MISSING
    for path in sorted(base.keys() | local.keys() | remote.keys()):
        b, l, r = (mapping.get(path, missing) for mapping in (base, local, remote))
        value: str | bytes | Missing
        try:
            value = choose(b, l, r, path)
        except Conflict:
            if isinstance(l, str) and isinstance(r, str):
                value = merge_content(
                    path,
                    cache_file(cache, b).read_bytes() if isinstance(b, str) else None,
                    cache_file(cache, l).read_bytes(),
                    cache_file(cache, r).read_bytes(),
                )
            else:
                raise Conflict(
                    f"{path}: a file was deleted on one side and edited on the other."
                ) from None
        if not isinstance(value, Missing):
            result[path] = value
    # Reject file/directory collisions before materializing anything.
    for path in result:
        for parent in Path(path).parents:
            if parent.as_posix() in result:
                raise Conflict(
                    f"{path}: another contributor used a parent directory as a file."
                )

    def read(path: str) -> bytes:
        value = result[path]
        return (
            value if isinstance(value, bytes) else cache_file(cache, value).read_bytes()
        )

    validate_references(result, read)
    return result


def materialize(files: dict[str, str | bytes], cache: Path, directory: Path) -> None:
    directory.mkdir()
    for name, value in files.items():
        target = directory / safe_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            target.write_bytes(value)
        else:
            # Copy rather than hardlink: editing the result must not mutate the cache.
            shutil.copyfile(cache_file(cache, value), target)


def check_workspace(root: Path) -> None:
    if git(root, "diff", "--cached", "--name-only"):
        raise SyncError(
            "Files are already staged in Git. Commit or unstage them before synchronizing; this command publishes dataset changes only."
        )
    changes = git(root, "diff", "--name-only").splitlines()
    if any(path != "data.dvc" for path in changes):
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
                "An earlier Git merge/rebase is unfinished. Finish or abort it first; no dataset changes were made."
            )


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
    dvc(work, "config", "--local", "cache.dir", str(cache))


def fetch_snapshot(
    work: Path, pointer: bytes, destination: Path, cache: Path
) -> dict[str, str]:
    (work / "data.dvc").write_bytes(pointer)
    destination.write_bytes(pointer)
    try:
        output = dvc(work, "fetch", "data.dvc")
    except SyncError as error:
        raise SyncError(
            f"Could not retrieve a published dataset. Its contributor may need to upload it using python dvc_gdrive_sync.py push.\n{error}"
        ) from error
    if output:
        print(output, flush=True)
    files = index(destination, cache)
    absent = [
        name
        for name, digest in files.items()
        if not cache_file(cache, digest).is_file()
    ]
    if absent:
        raise SyncError(
            f"Published dataset files are unavailable, starting with {absent[0]}. Ask its contributor to run python dvc_gdrive_sync.py push on their original machine."
        )
    return files


def verify_local(
    root: Path,
    head: str,
    pointer: bytes,
    local: dict[str, str],
    cache: Path,
    session: Path,
) -> None:
    check_workspace(root)
    if (
        git(root, "rev-parse", "HEAD") != head
        or (root / "data.dvc").read_bytes() != pointer
    ):
        raise SyncError(
            "Git changed during synchronization. Your data is preserved; finish that operation and rerun."
        )
    if (root / "data").is_dir():
        snapshot(root, session / "recheck.dvc")
        if index(session / "recheck.dvc", cache) != local:
            raise SyncError(
                "Your data changed during synchronization. Stop annotation/collection and rerun; the working dataset has not been replaced."
            )


def upload_snapshots(work: Path, incoming: str) -> None:
    """Upload every newly published pointer, including existing local commits."""
    pointer = work / "data.dvc"
    final = pointer.read_bytes()
    snapshots = {final}
    for revision in git(work, "rev-list", "HEAD", f"^{incoming}").splitlines():
        snapshots.add(git(work, "show", f"{revision}:data.dvc").encode() + b"\n")
    cache = Path(dvc(work, "cache", "dir")).resolve()
    try:
        for content in sorted(snapshots):
            pointer.write_bytes(content)
            if not cache_file(cache, load_yaml(pointer)["outs"][0]["md5"]).is_file():
                dvc(work, "fetch", "data.dvc")
            index(pointer, cache)
            dvc(work, "push", "data.dvc")
            cloud = dvc(work, "status", "--cloud", "--json", "data.dvc")
            if json.loads(cloud):
                raise SyncError(
                    "Remote verification found missing dataset objects. Git publication was stopped."
                )
    finally:
        pointer.write_bytes(final)


def sync(root: Path, message: str | None, retries: int = 3) -> None:
    pending_apply = root / ".data-sync/apply.pending"
    if pending_apply.exists():
        raise SyncError(
            "An earlier synchronization published successfully but was interrupted while updating this machine. "
            f"Ask the maintainer to finish recovery using {pending_apply.read_text().strip()}. "
            "Do not pull DVC over the current data or delete recovery files."
        )
    check_workspace(root)
    branch = git(root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    if not branch:
        raise SyncError(
            "Switch to your normal branch before synchronizing (usually main)."
        )
    remote = git(root, "config", "--get", f"branch.{branch}.remote", check=False)
    remote_ref = git(root, "config", "--get", f"branch.{branch}.merge", check=False)
    if not remote or remote == "." or not remote_ref.startswith("refs/heads/"):
        raise SyncError(
            "This branch needs a Git remote tracking branch. Ask the maintainer to configure it."
        )
    remote_url = git(root, "remote", "get-url", "--push", remote)
    head = git(root, "rev-parse", "HEAD")
    cache = Path(dvc(root, "cache", "dir")).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session = root / ".data-sync" / timestamp
    session.mkdir(parents=True)
    local_pointer = session / "local.dvc"
    original_pointer = (root / "data.dvc").read_bytes()
    (session / "original.dvc").write_bytes(original_pointer)
    (session / "state.json").write_text(
        json.dumps({"head": head, "branch": branch, "remote_ref": remote_ref}, indent=2)
        + "\n"
    )
    (session / "README.txt").write_text(
        "Dataset synchronization recovery\n\n"
        "Before publication, your original working data stays in data/.\n"
        "local.dvc preserves its cached snapshot; original.dvc preserves the original pointer.\n"
        "base.dvc and incoming.dvc identify the snapshots used for comparison.\n"
        "After success, data-before-sync/ contains your previous working directory.\n"
        "Do not delete recovery files or run DVC cache garbage collection while recovering.\n"
        "For an ambiguous edit, correct/rename the reported file in data/ and rerun.\n"
        "If unsure, share the error and this directory with the project maintainer.\n"
        "No force-push or manual hash editing is needed.\n"
    )
    print(f"Recovery files: {session.relative_to(root)}", flush=True)
    if (root / "data").is_dir():
        print("Saving your current dataset before downloading changes...", flush=True)
        snapshot(root, local_pointer)
    else:
        local_pointer.write_bytes(original_pointer)
    for attempt in range(1, retries + 1):
        print(f"Checking incoming changes (attempt {attempt}/{retries})...", flush=True)
        git(root, "fetch", remote, remote_ref)
        incoming = git(root, "rev-parse", "FETCH_HEAD")
        ancestors = git(root, "merge-base", "--all", head, incoming).splitlines()
        if len(ancestors) != 1:
            raise SyncError(
                "The branches have an ambiguous common ancestor. Ask the maintainer to review the Git history."
            )
        work = session / f"attempt-{attempt}"
        try:
            prepare_clone(root, work, head, remote_url, cache)
            local = fetch_snapshot(
                work, local_pointer.read_bytes(), session / "local.dvc", cache
            )
            base = fetch_snapshot(
                work,
                git(root, "show", f"{ancestors[0]}:data.dvc").encode() + b"\n",
                session / "base.dvc",
                cache,
            )
            incoming_pointer = (
                git(root, "show", f"{incoming}:data.dvc").encode() + b"\n"
            )
            theirs = fetch_snapshot(
                work, incoming_pointer, session / "incoming.dvc", cache
            )
            print("Combining files and manifest records...", flush=True)
            combined = merge_dataset(base, local, theirs, cache)
            if (
                head == incoming
                and local == theirs
                and original_pointer == incoming_pointer
                and (root / "data").is_dir()
            ):
                verify_local(root, head, original_pointer, local, cache, session)
                upload_snapshots(work, incoming)
                verify_local(root, head, original_pointer, local, cache, session)
                git(root, "fetch", remote, remote_ref)
                if git(root, "rev-parse", "FETCH_HEAD") != incoming:
                    print("New incoming changes arrived. Checking again...", flush=True)
                    continue
                print(
                    "Your dataset is already synchronized. No new commit or data copy was needed.",
                    flush=True,
                )
                return
            # Restore the clone before merging code/history; all dataset changes are separate.
            git(work, "checkout", "HEAD", "--", "data.dvc")
            if git(root, "merge-base", head, incoming) == head:
                git(work, "reset", "--hard", incoming)
            else:
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
                        "Code changes also conflict: "
                        + ", ".join(conflicts)
                        + ". Ask the maintainer to resolve these separately."
                    )
                if result.returncode and not conflicts:
                    raise SyncError(result.stdout)
            (work / "data.dvc").write_bytes(incoming_pointer)
            materialize(combined, cache, work / "data")
            dvc(work, "add", "data")
            git(work, "add", "--", "data.dvc")
            pending = (
                bool(git(work, "diff", "--cached", "--name-only"))
                or (work / ".git/MERGE_HEAD").exists()
            )
            if pending:
                if message is None:
                    if not sys.stdin.isatty():
                        raise SyncError(
                            'Supply a commit message with -m "Name: tools/samples added or corrected".'
                        )
                    message = input(
                        "Describe your changes (name, tools/samples): "
                    ).strip()
                    if not message:
                        raise SyncError(
                            "No commit message entered. Your original data is unchanged; rerun when ready."
                        )
                git(work, "commit", "-m", f"[data-sync] {message}")
            verify_local(root, head, original_pointer, local, cache, session)
            print(
                "Uploading complete datasets before publishing their Git pointers...",
                flush=True,
            )
            upload_snapshots(work, incoming)
            verify_local(root, head, original_pointer, local, cache, session)
            pushed = run(
                ["git", "push", "origin", f"HEAD:{remote_ref}"], work, check=False
            )
            if pushed.returncode:
                git(root, "fetch", remote, remote_ref)
                if (
                    git(root, "rev-parse", "FETCH_HEAD") != incoming
                    and attempt < retries
                ):
                    print(
                        "Another contributor published first. Combining their changes and retrying...",
                        flush=True,
                    )
                    shutil.rmtree(work / "data")
                    continue
                raise SyncError(
                    "Git publication failed; your data is preserved. Check connectivity/access and rerun.\n"
                    + pushed.stdout
                )
            verify_local(root, head, original_pointer, local, cache, session)
            # The candidate includes the original local HEAD, so this is always a fast-forward.
            git(root, "fetch", str(work), "HEAD")
            # An interrupted directory replacement must not be mistaken for user deletions.
            pending_apply.write_text(str(session) + "\n")
            (root / "data.dvc").write_bytes(
                git(root, "show", "HEAD:data.dvc").encode() + b"\n"
            )
            git(root, "merge", "--ff-only", "FETCH_HEAD")
            if (root / "data").exists():
                (root / "data").rename(session / "data-before-sync")
            (work / "data").rename(root / "data")
            pending_apply.unlink()
            git(root, "fetch", remote, remote_ref)
            print(
                "Synchronization complete. Your data and Git commit are published.",
                flush=True,
            )
            print(
                f"Your previous data remains in {session.relative_to(root)}/data-before-sync (if present).",
                flush=True,
            )
            return
        finally:
            # Do not leave copies of local OAuth credentials in recovery checkouts.
            (work / ".dvc/config.local").unlink(missing_ok=True)
    raise SyncError(
        "Other contributors kept publishing. Your data is preserved; rerun shortly."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine local and incoming dataset changes, then upload and publish safely."
    )
    parser.add_argument(
        "-m",
        "--message",
        help="commit message; prompted when needed; prefixed with [data-sync]",
    )
    args = parser.parse_args()
    try:
        directory = ROOT / ".data-sync"
        directory.mkdir(exist_ok=True)
        with FileLock(directory / "sync.lock", timeout=0):
            sync(ROOT, args.message)
    except Timeout:
        print(
            "Another dataset synchronization is already running. Wait for it to finish.",
            file=sys.stderr,
        )
        return 1
    except (SyncError, Conflict, OSError, ValueError, KeyboardInterrupt) as error:
        print(
            f"\nSynchronization stopped: {error}\nYour local data and recovery snapshots have been preserved. Resolve the reported issue and rerun python sync_data.py. Do not delete .data-sync/ while recovering work.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
