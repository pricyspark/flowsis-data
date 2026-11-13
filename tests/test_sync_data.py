from pathlib import Path
import subprocess

import pytest

import sync_data


def call(path, *args):
    result = subprocess.run(args, cwd=path, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def git(path, *args):
    return call(path, "git", *args)


def dvc(path, *args):
    return call(path, "python", str(sync_data.WRAPPER), *args)


@pytest.fixture
def repositories(tmp_path, monkeypatch):
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(tmp_path / "dvc-site-cache"))
    remote = tmp_path / "git-remote"
    git(tmp_path, "init", "--bare", str(remote))
    root = tmp_path / "local"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Test Contributor")
    git(root, "config", "user.email", "test@example.com")
    dvc(root, "init")
    dvc(root, "config", "core.site_cache_dir", str(tmp_path / "site-cache"))
    dvc(root, "remote", "add", "-d", "storage", str(tmp_path / "storage"))
    (root / ".gitignore").write_text("/data/\n/.data-sync/\n")
    (root / "data").mkdir()
    (root / "data/base.txt").write_text("base")
    dvc(root, "add", "data")
    dvc(root, "push")
    git(root, "add", ".")
    git(root, "commit", "-m", "Initial dataset")
    git(root, "remote", "add", "origin", str(remote))
    git(root, "push", "-u", "origin", "main")
    git(tmp_path, "--git-dir", str(remote), "symbolic-ref", "HEAD", "refs/heads/main")
    other = tmp_path / "other"
    git(tmp_path, "clone", str(remote), str(other))
    git(other, "config", "user.name", "Other Contributor")
    git(other, "config", "user.email", "other@example.com")
    dvc(other, "pull")
    return root, other, remote


def publish(root, name, content):
    (root / "data" / name).write_text(content)
    dvc(root, "add", "data")
    dvc(root, "push")
    git(root, "add", "data.dvc")
    git(root, "commit", "-m", f"Add {name}")
    git(root, "push")


def test_sync_combines_divergent_commits_and_uncommitted_files(repositories):
    root, other, remote = repositories
    (root / "data/local.txt").write_text("local")
    dvc(root, "add", "data")
    git(root, "add", "data.dvc")
    git(root, "commit", "-m", "Unpublished local dataset")
    local_commit = git(root, "rev-parse", "HEAD")
    unpublished_hash = sync_data.load_yaml(root / "data.dvc")["outs"][0]["md5"]
    (root / "data/extra.txt").write_text("extra")
    publish(other, "incoming.txt", "incoming")
    dvc(root, "config", "--local", "core.autostage", "true")

    sync_data.sync(root, "Combined dataset")

    assert {p.name for p in (root / "data").iterdir()} == {
        "base.txt",
        "local.txt",
        "extra.txt",
        "incoming.txt",
    }
    assert git(root, "rev-parse", "HEAD") == git(remote, "rev-parse", "main")
    assert git(root, "log", "-1", "--format=%s") == "[data-sync] Combined dataset"
    assert (
        git(root, "show", "-s", "--format=%s", local_commit)
        == "Unpublished local dataset"
    )
    assert sync_data.cache_file(remote.parent / "storage", unpublished_hash).is_file()
    assert not git(root, "status", "--porcelain", "--untracked-files=no")
    assert dvc(root, "status", "--json") == "{}"
    assert dvc(root, "status", "--cloud", "--json") == "{}"
    backups = list((root / ".data-sync").glob("*/data-before-sync"))
    assert len(backups) == 1
    assert (backups[0] / "extra.txt").read_text() == "extra"
    # An unchanged rerun should not create another commit.
    head = git(root, "rev-parse", "HEAD")
    sync_data.sync(root, "No changes")
    assert git(root, "rev-parse", "HEAD") == head


def test_same_path_conflict_preserves_original_workspace(repositories):
    root, other, _ = repositories
    head = git(root, "rev-parse", "HEAD")
    pointer = (root / "data.dvc").read_bytes()
    (root / "data/base.txt").write_text("local edit")
    publish(other, "base.txt", "different incoming edit")
    with pytest.raises(sync_data.Conflict, match="different file contents"):
        sync_data.sync(root, "Cannot combine")
    assert (root / "data/base.txt").read_text() == "local edit"
    assert (root / "data.dvc").read_bytes() == pointer
    assert git(root, "rev-parse", "HEAD") == head
    assert not list((root / ".data-sync").glob("*/attempt-*/.dvc/config.local"))


def test_failed_dvc_upload_never_publishes_git(repositories, monkeypatch):
    root, _, remote = repositories
    before = git(remote, "rev-parse", "main")
    (root / "data/local.txt").write_text("local")
    original = sync_data.dvc

    def failing(cwd, *args):
        if args[0] == "push":
            raise sync_data.SyncError("Simulated upload failure")
        return original(cwd, *args)

    monkeypatch.setattr(sync_data, "dvc", failing)
    with pytest.raises(sync_data.SyncError, match="upload failure"):
        sync_data.sync(root, "Not uploaded")
    assert git(remote, "rev-parse", "main") == before
    assert git(root, "rev-parse", "HEAD") == before
    assert (root / "data/local.txt").read_text() == "local"


def test_publication_race_retries_without_force_push(repositories, monkeypatch):
    root, other, remote = repositories
    (root / "data/local.txt").write_text("local")
    original = sync_data.run
    raced = False

    def racing(args, cwd, *, check=True):
        nonlocal raced
        if args[:2] == ["git", "push"] and not raced:
            raced = True
            publish(other, "racing.txt", "raced")
        return original(args, cwd, check=check)

    monkeypatch.setattr(sync_data, "run", racing)
    sync_data.sync(root, "Handle race")
    assert raced
    assert (root / "data/racing.txt").read_text() == "raced"
    assert (root / "data/local.txt").read_text() == "local"
    assert git(root, "rev-parse", "HEAD") == git(remote, "rev-parse", "main")


def test_missing_incoming_cache_stops_before_publication(repositories):
    root, other, remote = repositories
    before = git(root, "rev-parse", "HEAD")
    (root / "data/local.txt").write_text("local")
    (other / "data/missing.txt").write_text("not uploaded")
    dvc(other, "add", "data")
    git(other, "add", "data.dvc")
    git(other, "commit", "-m", "Pointer without data")
    git(other, "push")
    remote_head = git(remote, "rev-parse", "main")
    with pytest.raises(sync_data.SyncError, match="dvc_gdrive_sync.py push"):
        sync_data.sync(root, "Must stop")
    assert git(remote, "rev-parse", "main") == remote_head
    assert git(root, "rev-parse", "HEAD") == before
    assert (root / "data/local.txt").read_text() == "local"


def test_edit_during_upload_preserves_work_and_stops_git_push(
    repositories, monkeypatch
):
    root, _, remote = repositories
    before = git(remote, "rev-parse", "main")
    (root / "data/local.txt").write_text("before upload")
    original = sync_data.dvc

    def editing(cwd, *args):
        if args[0] == "push":
            (root / "data/local.txt").write_text("edited during upload")
        return original(cwd, *args)

    monkeypatch.setattr(sync_data, "dvc", editing)
    with pytest.raises(
        sync_data.SyncError, match="data changed during synchronization"
    ):
        sync_data.sync(root, "Do not overwrite")
    assert git(remote, "rev-parse", "main") == before
    assert git(root, "rev-parse", "HEAD") == before
    assert (root / "data/local.txt").read_text() == "edited during upload"


def test_missing_working_data_is_restored_without_new_commit(repositories):
    import shutil

    root, _, _ = repositories
    head = git(root, "rev-parse", "HEAD")
    shutil.rmtree(root / "data")
    sync_data.sync(root, "Restore")
    assert (root / "data/base.txt").read_text() == "base"
    assert git(root, "rev-parse", "HEAD") == head


def test_interrupted_final_update_cannot_be_misread_as_local_deletions(
    repositories, monkeypatch
):
    root, _, _ = repositories
    (root / "data/local.txt").write_text("local")
    original = sync_data.git

    def interrupted(cwd, *args, **kwargs):
        result = original(cwd, *args, **kwargs)
        if cwd == root and args[:2] == ("merge", "--ff-only"):
            raise sync_data.SyncError("Simulated interruption after Git update")
        return result

    monkeypatch.setattr(sync_data, "git", interrupted)
    with pytest.raises(sync_data.SyncError, match="Simulated interruption"):
        sync_data.sync(root, "Published before interruption")
    assert (root / ".data-sync/apply.pending").is_file()
    assert (root / "data/local.txt").read_text() == "local"
    monkeypatch.setattr(sync_data, "git", original)
    with pytest.raises(sync_data.SyncError, match="interrupted while updating"):
        sync_data.sync(root, "Must not infer deletions")
