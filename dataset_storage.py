"""DVC object/index operations without materializing entire datasets.

Keep the small DVC-internal integration here; directory serialization is delegated
to DVC's Tree implementation rather than duplicated in the sync workflow.
"""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import tempfile
from collections.abc import Iterator, Mapping

import yaml
from dvc.repo import Repo
from dvc_data.hashfile.db import HashFileDB
from dvc_data.hashfile.hash_info import HashInfo
from dvc_data.hashfile.tree import Tree
from dvc_objects.fs.local import LocalFileSystem

from dataset_merge import Conflict, safe_path
from dvc_gdrive_sync import bounded_drive_queries


class SyncError(RuntimeError):
    pass


def cache_file(cache: Path, digest: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}(?:\.dir)?", digest):
        raise SyncError(f"Unsupported DVC object hash: {digest}")
    return cache / "files/md5" / digest[:2] / digest[2:]


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def pointer_hash(pointer: bytes) -> str:
    document = yaml.safe_load(pointer)
    if not isinstance(document, dict):
        raise SyncError("Invalid data.dvc pointer")
    outputs = document.get("outs")
    if (
        not isinstance(outputs, list)
        or len(outputs) != 1
        or not isinstance(outputs[0], dict)
        or outputs[0].get("path") != "data"
    ):
        raise SyncError("Expected data.dvc to track the single data/ directory.")
    digest = outputs[0].get("md5", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{32}\.dir", digest):
        raise SyncError("Expected an MD5 directory snapshot in data.dvc.")
    return digest


def read_index(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in json.loads(path.read_text()):
        name = safe_path(row["relpath"])
        digest = row["md5"]
        if not re.fullmatch(r"[0-9a-f]{32}", digest) or name in result:
            raise SyncError(f"Invalid or repeated dataset entry: {name}")
        result[name] = digest
    for name in result:
        if any(parent.as_posix() in result for parent in Path(name).parents):
            raise Conflict(f"File/directory collision: {name}")
    return result


class DatasetStore:
    def __init__(self, root: Path, repo: Repo):
        self.root = root
        self.repo = repo
        self.local = repo.cache.local
        self.cache = Path(self.local.path).parent.parent
        self._remote: HashFileDB | None = None

    @property
    def remote(self) -> HashFileDB:
        if self._remote is None:
            self._remote = self.repo.cloud.get_remote_odb()
        return self._remote

    def get(self, digest: str) -> Path:
        target = cache_file(self.cache, digest)
        if target.is_file():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            download = Path(temporary) / "object"
            try:
                self.remote.fs.get_file(self.remote.oid_to_path(digest), str(download))
            except Exception as error:
                raise SyncError(
                    f"DVC object {digest} is unavailable. Ask its contributor to run "
                    "python dvc_gdrive_sync.py push on their original machine. "
                    f"Storage error: {error}"
                ) from error
            if checksum(download) != digest.removesuffix(".dir"):
                raise SyncError(f"Downloaded DVC object failed checksum: {digest}")
            download.replace(target)
            target.chmod(0o444)
        return target

    def index(self, pointer: bytes) -> dict[str, str]:
        return read_index(self.get(pointer_hash(pointer)))

    def prefetch(self, digests: set[str]) -> None:
        missing = [
            digest for digest in digests if not cache_file(self.cache, digest).is_file()
        ]
        if not missing:
            return
        # Initialize authentication once before starting transfer workers.
        _ = self.remote
        with ThreadPoolExecutor(max_workers=4) as executor:
            for number, _ in enumerate(executor.map(self.get, missing), 1):
                if number % 100 == 0:
                    print(
                        f"Downloaded {number}/{len(missing)} selected objects",
                        flush=True,
                    )

    def put_file(self, path: Path, digest: str | None = None) -> str:
        digest = digest or checksum(path)
        target = cache_file(self.cache, digest)
        if digest.endswith(".dir"):
            raise SyncError("File content cannot use a directory hash")
        existed = target.exists()
        if existed:
            return digest
        self.local.add(str(path), LocalFileSystem(), digest)
        if not existed and checksum(target) != digest:
            target.unlink()
            raise SyncError(f"File changed while caching: {path}")
        return digest

    def put_bytes(self, content: bytes) -> str:
        digest = hashlib.md5(content).hexdigest()
        if not cache_file(self.cache, digest).exists():
            with tempfile.TemporaryDirectory(dir=self.root / ".data-sync") as directory:
                path = Path(directory) / "merged"
                path.write_bytes(content)
                self.put_file(path, digest)
        return digest

    def pointer(self, files: Mapping[str, str], previous: bytes | None = None) -> bytes:
        tree = Tree()
        for name, digest in sorted(files.items()):
            tree.add(tuple(safe_path(name).split("/")), None, HashInfo("md5", digest))
        tree.digest()
        assert tree.hash_info and tree.hash_info.value
        digest = tree.hash_info.value
        self.local.add(tree.path, tree.fs, digest)
        if previous is not None and pointer_hash(previous) == digest:
            return previous
        # Size is optional in DVC pointers. Obtaining the total for a partial
        # checkout would otherwise require inspecting every remote media object.
        text = yaml.safe_dump(
            {
                "outs": [
                    {"md5": digest, "nfiles": len(files), "hash": "md5", "path": "data"}
                ]
            },
            sort_keys=False,
        )
        assert isinstance(text, str)
        return text.encode()

    def remote_missing(self, digests: set[str]) -> set[str]:
        if not digests:
            return set()
        try:
            return digests - set(self.remote.oids_exist(digests, jobs=4))
        except Exception as error:
            raise SyncError(f"Could not verify DVC remote objects: {error}") from error

    def verify_remote(self, files: Mapping[str, str], pointer: bytes) -> None:
        missing = self.remote_missing({pointer_hash(pointer), *files.values()})
        if missing:
            raise SyncError(
                f"Published dataset objects are missing remotely ({len(missing)}). "
                "Ask their contributor to run python dvc_gdrive_sync.py push "
                "on their original machine. Git publication was stopped."
            )


@contextmanager
def open_store(root: Path) -> Iterator[DatasetStore]:
    (root / ".data-sync").mkdir(exist_ok=True)
    with bounded_drive_queries(), Repo(
        str(root),
        config={"core": {"site_cache_dir": str(root / ".data-sync/dvc-state")}},
    ) as repo:
        yield DatasetStore(root, repo)
