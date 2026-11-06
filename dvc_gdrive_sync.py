"""Run DVC with bounded Google Drive listing queries.

Use like the DVC CLI: python dvc_gdrive_sync.py push
"""

import sys
from collections.abc import Iterable, Iterator
from itertools import batched

from dvc.cli import main as dvc_main
from pydrive2.files import GoogleDriveFile
from pydrive2.fs.spec import GDriveFileSystem


def main() -> int:
    original = GDriveFileSystem._gdrive_list_ids

    def list_parent_batches(
        self: GDriveFileSystem, query_ids: Iterable[str]
    ) -> Iterator[GoogleDriveFile]:
        # PyDrive2 otherwise combines every parent folder into one query.
        # Drive rejects the large query for this remote with HTTP 400.
        for parent_ids in batched(query_ids, 32):
            yield from original(self, parent_ids)

    GDriveFileSystem._gdrive_list_ids = list_parent_batches
    try:
        return dvc_main(sys.argv[1:])
    finally:
        GDriveFileSystem._gdrive_list_ids = original


if __name__ == "__main__":
    raise SystemExit(main())
