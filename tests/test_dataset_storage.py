from unittest.mock import Mock

import pytest
from pydrive2.fs.spec import GDriveAuthError

from dataset_storage import DatasetStore, SyncError


@pytest.mark.parametrize(
    "error, expected, missing",
    [
        (FileNotFoundError("missing object"), "is unavailable", True),
        (GDriveAuthError("Failed to authenticate GDrive"), "authentication", False),
        (PermissionError("access denied"), "permissions", False),
        (ConnectionError("offline"), "connectivity", False),
    ],
)
def test_download_errors_only_request_contributor_upload_for_missing_objects(
    tmp_path, error, expected, missing
):
    repo = Mock()
    repo.cache.local.path = str(tmp_path / "cache/files/md5")
    repo.cloud.get_remote_odb.return_value.fs.get_file.side_effect = error
    store = DatasetStore(tmp_path, repo)
    digest = "a" * 32

    with pytest.raises(SyncError, match=expected) as failure:
        store.get(digest)

    assert ("contributor" in str(failure.value)) is missing
    assert failure.value.__cause__ is error
    assert not (store.cache / "files/md5" / digest[:2] / digest[2:]).exists()


@pytest.mark.parametrize("operation", ["get", "remote_missing"])
def test_remote_authentication_preserves_callback_error_detail(tmp_path, operation):
    repo = Mock()
    repo.cache.local.path = str(tmp_path / "cache/files/md5")
    error = GDriveAuthError("Failed to authenticate GDrive")
    error.__cause__ = RuntimeError("Could not start the Google OAuth callback server")
    remote = repo.cloud.get_remote_odb.return_value
    remote.fs.get_file.side_effect = error
    remote.oids_exist.side_effect = error
    store = DatasetStore(tmp_path, repo)
    digest = "a" * 32

    with pytest.raises(
        SyncError, match="Could not start the Google OAuth callback server"
    ):
        if operation == "get":
            store.get(digest)
        else:
            store.remote_missing({digest})
