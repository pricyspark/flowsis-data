"""Run DVC with bounded Google Drive listing queries.

Use like the DVC CLI: python dvc_gdrive_sync.py push
"""

import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from itertools import batched

from dvc.cli import main as dvc_main
from pydrive2.auth import AuthenticationError, GoogleAuth
from pydrive2.files import GoogleDriveFile
from pydrive2.fs.spec import GDriveFileSystem


@contextmanager
def drive_authentication() -> Iterator[None]:
    original = GoogleAuth.LocalWebserverAuth

    def local_auth(
        self: GoogleAuth,
        host_name: str = "127.0.0.1",
        port_numbers: list[int] | None = None,
        launch_browser: bool = True,
        bind_addr: str | None = None,
    ) -> None:
        # Other local services often occupy PyDrive2's two default ports.
        ports = (
            port_numbers
            if port_numbers is not None
            else [8080, 8090, *range(8100, 8111)]
        )
        try:
            return original(
                self,
                host_name=host_name,
                port_numbers=ports,
                launch_browser=launch_browser,
                bind_addr=bind_addr,
            )
        except AuthenticationError as error:
            if str(error):
                raise
            raise AuthenticationError(
                "Could not start the Google OAuth callback server on "
                f"{bind_addr or host_name} "
                f"using ports {ports}. Free one of these ports and rerun "
                "python sync_data.py pull. "
                "Keep the command running during browser login."
            ) from error

    setattr(GoogleAuth, "LocalWebserverAuth", local_auth)
    try:
        yield
    finally:
        setattr(GoogleAuth, "LocalWebserverAuth", original)


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
    with bounded_drive_queries(), drive_authentication():
        return dvc_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
