"""Run DVC with bounded Google Drive listing queries.

Use like the DVC CLI: python dvc_gdrive_sync.py push
"""

import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from itertools import batched

from dvc.cli import main as dvc_main
from pydrive2.files import GoogleDriveFile
from pydrive2.fs.spec import GDriveFileSystem


@contextmanager
def bounded_drive_queries() -> Iterator[None]:
    original = GDriveFileSystem._gdrive_list_ids

    def list_parent_batches(
        self: GDriveFileSystem, query_ids: Iterable[str]
    ) -> Iterator[GoogleDriveFile]:
        # PyDrive2 otherwise combines every parent folder into one query.
        # Drive rejects the large query for this remote with HTTP 400.
        for parent_ids in batched(query_ids, 32):
            yield from original(self, parent_ids)

    setattr(GDriveFileSystem, "_gdrive_list_ids", list_parent_batches)
    try:
        yield
    finally:
        setattr(GDriveFileSystem, "_gdrive_list_ids", original)


def main() -> int:
    with bounded_drive_queries():
        return dvc_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
