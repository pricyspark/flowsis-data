import sys

import pytest

import dvc_gdrive_sync


@pytest.mark.parametrize("count", [0, 1, 32, 33, 254])
def test_drive_listing_batches_preserve_all_results_and_restore_method(
    monkeypatch, count
):
    calls = []
    parents = {f"parent-{index}": f"folder-{index}" for index in range(count)}

    def original(self, ids):
        calls.append(tuple(ids))
        return iter(ids)

    def command(args):
        assert args == ["status", "--cloud"]
        actual = list(
            dvc_gdrive_sync.GDriveFileSystem._gdrive_list_ids(None, parents)
        )
        assert actual == list(parents)
        return 0

    monkeypatch.setattr(dvc_gdrive_sync.GDriveFileSystem, "_gdrive_list_ids", original)
    monkeypatch.setattr(dvc_gdrive_sync, "dvc_main", command)
    monkeypatch.setattr(sys, "argv", ["dvc_gdrive_sync.py", "status", "--cloud"])

    assert dvc_gdrive_sync.main() == 0
    assert all(1 <= len(batch) <= 32 for batch in calls)
    assert len(calls) == (count + 31) // 32
    assert dvc_gdrive_sync.GDriveFileSystem._gdrive_list_ids is original


def test_drive_listing_restores_method_when_command_fails(monkeypatch):
    original = dvc_gdrive_sync.GDriveFileSystem._gdrive_list_ids

    def command(args):
        raise RuntimeError("Command failed")

    monkeypatch.setattr(dvc_gdrive_sync, "dvc_main", command)
    with pytest.raises(RuntimeError, match="Command failed"):
        dvc_gdrive_sync.main()
    assert dvc_gdrive_sync.GDriveFileSystem._gdrive_list_ids is original
