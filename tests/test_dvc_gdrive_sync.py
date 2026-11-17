import sys
from unittest.mock import Mock

import pydrive2.auth
import pytest

import dvc_gdrive_sync


def test_drive_authentication_uses_loopback_and_fallback_ports(monkeypatch):
    calls = []

    def original(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(dvc_gdrive_sync.GoogleAuth, "LocalWebserverAuth", original)
    with dvc_gdrive_sync.drive_authentication():
        dvc_gdrive_sync.GoogleAuth.LocalWebserverAuth(None)
        dvc_gdrive_sync.GoogleAuth.LocalWebserverAuth(
            None, host_name="localhost", port_numbers=[9000], launch_browser=False
        )
    assert calls[0] == {
        "host_name": "127.0.0.1",
        "port_numbers": [8080, 8090, *range(8100, 8111)],
        "launch_browser": True,
        "bind_addr": None,
    }
    assert calls[1]["host_name"] == "localhost"
    assert calls[1]["port_numbers"] == [9000]
    assert calls[1]["launch_browser"] is False
    assert dvc_gdrive_sync.GoogleAuth.LocalWebserverAuth is original


def test_drive_authentication_completes_callback_when_default_ports_are_busy(
    monkeypatch,
):
    monkeypatch.delenv("GDRIVE_NON_INTERACTIVE", raising=False)
    addresses = []
    server = Mock(query_params={"code": "test-code"})

    def start_server(address, handler):
        addresses.append(address)
        if address[1] in {8080, 8090}:
            raise OSError("Address already in use")
        return server

    monkeypatch.setattr(pydrive2.auth, "ClientRedirectServer", start_server)
    auth = Mock(settings={}, credentials=None)
    auth.GetAuthUrl.return_value = "https://accounts.google.com/test"
    auth.Auth.side_effect = lambda code: setattr(auth, "credentials", Mock())
    with dvc_gdrive_sync.drive_authentication():
        dvc_gdrive_sync.GoogleAuth.LocalWebserverAuth(auth, launch_browser=False)

    assert addresses == [("127.0.0.1", port) for port in (8080, 8090, 8100)]
    assert auth.flow.redirect_uri == "http://127.0.0.1:8100/"
    server.handle_request.assert_called_once()
    auth.Auth.assert_called_once_with("test-code")


@pytest.mark.parametrize("message", ["", "User rejected authentication"])
def test_drive_authentication_explains_bind_failure_and_restores_method(
    monkeypatch, message
):
    def original(self, **kwargs):
        raise dvc_gdrive_sync.AuthenticationError(message)

    monkeypatch.setattr(dvc_gdrive_sync.GoogleAuth, "LocalWebserverAuth", original)
    expected = message or "Could not start the Google OAuth callback server"
    with pytest.raises(dvc_gdrive_sync.AuthenticationError, match=expected):
        with dvc_gdrive_sync.drive_authentication():
            dvc_gdrive_sync.GoogleAuth.LocalWebserverAuth(None)
    assert dvc_gdrive_sync.GoogleAuth.LocalWebserverAuth is original


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
        actual = list(dvc_gdrive_sync.GDriveFileSystem._gdrive_list_ids(None, parents))
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
