from pathlib import Path
import subprocess
import sys

import pytest

import sync_data


def call(path, *args):
    result = subprocess.run(args, cwd=path, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def git(path, *args):
    return call(path, "git", *args)


def dvc(path, *args):
    return call(path, sys.executable, str(sync_data.WRAPPER), *args)


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
    (root / "data" / name).parent.mkdir(parents=True, exist_ok=True)
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

    sync_data.sync(root, "Combined dataset", preset="all")

    assert {p.name for p in (root / "data").iterdir() if p.is_file()} == {
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
    snapshots = list((root / ".data-sync/sessions").glob("*/local.dvc"))
    assert len(snapshots) == 1
    cache = root / ".dvc/cache"
    assert "extra.txt" in sync_data.index(snapshots[0], cache)
    # An unchanged rerun should not create another commit.
    head = git(root, "rev-parse", "HEAD")
    sync_data.sync(root, "No changes", preset="all")
    assert git(root, "rev-parse", "HEAD") == head


def test_same_path_conflict_preserves_original_workspace(repositories):
    root, other, _ = repositories
    head = git(root, "rev-parse", "HEAD")
    pointer = (root / "data.dvc").read_bytes()
    (root / "data/base.txt").write_text("local edit")
    publish(other, "base.txt", "different incoming edit")
    with pytest.raises(sync_data.Conflict, match="different file contents"):
        sync_data.sync(root, "Cannot combine", preset="all")
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
        sync_data.sync(root, "Not uploaded", preset="all")
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
    sync_data.sync(root, "Handle race", preset="all")
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
        sync_data.sync(root, "Must stop", preset="all")
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
        sync_data.sync(root, "Do not overwrite", preset="all")
    assert git(remote, "rev-parse", "main") == before
    assert git(root, "rev-parse", "HEAD") == before
    assert (root / "data/local.txt").read_text() == "edited during upload"


def test_missing_working_data_is_restored_without_new_commit(repositories):
    import shutil

    root, _, _ = repositories
    head = git(root, "rev-parse", "HEAD")
    shutil.rmtree(root / "data")
    sync_data.sync(root, "Restore", preset="all")
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
        sync_data.sync(root, "Published before interruption", preset="all")
    assert (root / ".data-sync/apply.pending").is_file()
    assert (root / "data/local.txt").read_text() == "local"
    monkeypatch.setattr(sync_data, "git", original)
    with pytest.raises(sync_data.SyncError, match="interrupted while updating"):
        sync_data.sync(root, "Must not infer deletions", preset="all")
    sync_data.recover(root)
    assert not (root / ".data-sync/apply.pending").exists()
    assert (root / "data/local.txt").read_text() == "local"


def sparse_dataset(repositories):
    import shutil

    root, other, remote = repositories
    for name, content in {
        "videos/45/capture.mp4": "video-content" * 100,
        "images/45/photo.jpg": "image-content",
        "raw-images/45/photo.HEIC": "raw-content",
        "video_manifest.csv": "Video ID,Filepath\nvideo-1,videos/45/capture.mp4\n",
        "image_orientation_corrections.json": "{}\n",
    }.items():
        path = other / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    dvc(other, "add", "data")
    dvc(other, "push")
    git(other, "add", "data.dvc")
    git(other, "commit", "-m", "Add images and videos")
    git(other, "push")
    shutil.rmtree(root / "data")
    shutil.rmtree(root / ".dvc/cache")
    return root, other, remote


def test_metadata_pull_and_contribution_never_download_videos(
    repositories, monkeypatch
):
    import json
    from dataset_storage import DatasetStore

    root, other, remote = sparse_dataset(repositories)
    source = sync_data.index(other / "data.dvc", other / ".dvc/cache")
    video_hash = source["videos/45/capture.mp4"]
    original_get = DatasetStore.get
    requested = []

    def get(self, digest):
        requested.append(digest)
        return original_get(self, digest)

    monkeypatch.setattr(DatasetStore, "get", get)
    sync_data.sync(root, None, publish=False)
    assert (root / "data/video_manifest.csv").is_file()
    assert (root / "data/images/45").is_dir()
    assert not (root / "data/images/45/photo.jpg").exists()
    assert not sync_data.cache_file(root / ".dvc/cache", video_hash).exists()
    assert git(root, "rev-parse", "HEAD") == git(remote, "rev-parse", "main")
    (root / "data/images/45/new.jpg").write_text("new local capture")
    publish(other, "images/45/incoming.jpg", "new incoming capture")
    sync_data.sync(root, "Add image from sparse checkout")
    files = sync_data.index(root / "data.dvc", root / ".dvc/cache")
    assert files["videos/45/capture.mp4"] == video_hash
    assert "images/45/new.jpg" in files and "images/45/incoming.jpg" in files
    assert not (root / "data/images/45/incoming.jpg").exists()
    assert video_hash not in requested
    assert not sync_data.cache_file(root / ".dvc/cache", video_hash).exists()
    assert not json.loads((root / ".data-sync/workspace.json").read_text())["deletions"]


def test_image_preset_is_persistent_and_excludes_raw_and_videos(repositories):
    root, other, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False, preset="images")
    assert (root / "data/images/45/photo.jpg").is_file()
    assert not (root / "data/raw-images/45/photo.HEIC").exists()
    assert not (root / "data/videos/45/capture.mp4").exists()
    publish(other, "images/45/later.jpg", "later capture")
    sync_data.sync(root, None, publish=False)
    assert (root / "data/images/45/later.jpg").read_text() == "later capture"


def test_eviction_reclaims_cache_and_does_not_delete_or_redownload(repositories):
    from dataset_commands import remove_or_evict

    root, other, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False, preset="videos")
    before = sync_data.index(root / "data.dvc", root / ".dvc/cache")
    digest = before["videos/45/capture.mp4"]
    remove_or_evict(root, ["data/videos/45"], evict=True)
    assert not (root / "data/videos/45/capture.mp4").exists()
    assert not sync_data.cache_file(root / ".dvc/cache", digest).exists()
    sync_data.sync(root, "Keep video remote")
    assert sync_data.index(root / "data.dvc", root / ".dvc/cache") == before
    assert not (root / "data/videos/45/capture.mp4").exists()
    assert not sync_data.cache_file(root / ".dvc/cache", digest).exists()
    sync_data.sync(root, None, publish=False, targets=["data/videos/45/capture.mp4"])
    assert (root / "data/videos/45/capture.mp4").is_file()


def test_unexpected_missing_file_requires_explicit_remove(repositories):
    from dataset_commands import remove_or_evict

    root, _, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False, preset="images")
    (root / "data/images/45/photo.jpg").unlink()
    with pytest.raises(sync_data.SyncError, match="disappeared"):
        sync_data.sync(root, "Do not infer deletion")
    remove_or_evict(root, ["data/images/45/photo.jpg"], evict=False)
    sync_data.sync(root, "Explicitly remove image")
    assert "images/45/photo.jpg" not in sync_data.index(
        root / "data.dvc", root / ".dvc/cache"
    )
    assert "videos/45/capture.mp4" in sync_data.index(
        root / "data.dvc", root / ".dvc/cache"
    )


def test_eviction_refuses_unpublished_edits(repositories):
    from dataset_commands import remove_or_evict

    root, _, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False, preset="images")
    image = root / "data/images/45/photo.jpg"
    image.write_text("unpublished edit")
    with pytest.raises(sync_data.SyncError, match="unpublished edits"):
        remove_or_evict(root, ["data/images/45"], evict=True)
    assert image.read_text() == "unpublished edit"


def test_sparse_remove_conflicts_with_remote_edit(repositories):
    from dataset_commands import remove_or_evict

    root, other, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False)
    remove_or_evict(root, ["data/images/45/photo.jpg"], evict=False)
    publish(other, "images/45/photo.jpg", "different incoming edit")
    with pytest.raises(sync_data.Conflict, match="deleted on one side"):
        sync_data.sync(root, "Do not discard edit")


def test_git_pull_before_sparse_sync_preserves_local_edits(repositories):
    root, other, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False, preset="images")
    (root / "data/images/45/photo.jpg").write_text("local edit")
    publish(other, "images/45/incoming.jpg", "incoming")
    git(root, "pull", "--ff-only")
    sync_data.sync(root, "Preserve local edit")
    assert (root / "data/images/45/photo.jpg").read_text() == "local edit"
    assert (root / "data/images/45/incoming.jpg").read_text() == "incoming"


def test_interrupted_file_apply_can_be_resumed(repositories, monkeypatch):
    root, other, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False, preset="images")
    publish(other, "images/45/photo.jpg", "updated image")
    original = Path.replace
    interrupted = False

    def replace(self, target):
        nonlocal interrupted
        if (
            not interrupted
            and "staged" in self.parts
            and str(target).endswith("data/images/45/photo.jpg")
        ):
            interrupted = True
            raise OSError("Simulated interrupted file replacement")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="Simulated interrupted"):
        sync_data.sync(root, None, publish=False)
    monkeypatch.setattr(Path, "replace", original)
    sync_data.recover(root)
    assert (root / "data/images/45/photo.jpg").read_text() == "updated image"
    assert not (root / ".data-sync/apply.pending").exists()


def test_recovery_preserves_new_edits_after_interruption(repositories, monkeypatch):
    root, other, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False, preset="images")
    publish(other, "images/45/photo.jpg", "incoming version")
    original = Path.replace

    def replace(self, target):
        if "staged" in self.parts and str(target).endswith("data/images/45/photo.jpg"):
            raise OSError("Interrupted")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="Interrupted"):
        sync_data.sync(root, None, publish=False)
    monkeypatch.setattr(Path, "replace", original)
    target = root / "data/images/45/photo.jpg"
    target.write_text("new edit during recovery")
    with pytest.raises(sync_data.SyncError, match="overwrite a new edit"):
        sync_data.recover(root)
    assert target.read_text() == "new edit during recovery"


def test_metadata_cannot_be_evicted_and_target_paths_are_safe(repositories):
    from dataset_commands import remove_or_evict

    root, _, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False)
    with pytest.raises(sync_data.SyncError, match="metadata working set"):
        remove_or_evict(root, ["data/video_manifest.csv"], evict=True)
    with pytest.raises(sync_data.Conflict, match="Unsafe"):
        remove_or_evict(root, ["../outside"], evict=False)


def test_symlinked_dataset_paths_are_refused(repositories):
    root, _, remote = repositories
    outside = remote.parent / "outside.txt"
    outside.write_text("outside")
    (root / "data/link.txt").symlink_to(outside)
    with pytest.raises(sync_data.SyncError, match="symlinks"):
        sync_data.sync(root, "Do not follow links")
    assert outside.read_text() == "outside"


def test_restore_cancels_pending_deletion_without_publishing(repositories):
    import json
    from dataset_commands import remove_or_evict, restore_removed

    root, _, remote = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False, preset="images")
    head = git(remote, "rev-parse", "main")
    image = root / "data/images/45/photo.jpg"
    remove_or_evict(root, ["data/images/45/photo.jpg"], evict=False)
    assert not image.exists()
    restore_removed(root, ["data/images/45/photo.jpg"])
    assert image.read_text() == "image-content"
    assert not json.loads((root / ".data-sync/workspace.json").read_text())["deletions"]
    assert git(remote, "rev-parse", "main") == head


def test_status_shows_pending_removal_before_references_are_fixed(repositories, capsys):
    from dataset_commands import remove_or_evict, status

    root, _, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False)
    remove_or_evict(root, ["data/videos/45/capture.mp4"], evict=False)
    status(root)
    assert '"videos/45/capture.mp4"' in capsys.readouterr().out
    with pytest.raises(sync_data.Conflict, match="missing video files"):
        sync_data.sync(root, "Refuse dangling reference")


def test_git_pull_after_readonly_merge_keeps_unpublished_addition(repositories):
    root, other, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False)
    local = root / "data/images/45/local.jpg"
    local.write_text("unpublished capture")
    sync_data.sync(root, None, publish=False)
    publish(other, "images/45/later.jpg", "incoming capture")
    git(root, "checkout", "HEAD", "--", "data.dvc")
    git(root, "pull", "--ff-only")
    sync_data.sync(root, "Keep unpublished capture")
    files = sync_data.index(root / "data.dvc", root / ".dvc/cache")
    assert "images/45/local.jpg" in files and "images/45/later.jpg" in files
    assert local.read_text() == "unpublished capture"


def test_divergent_pull_then_remote_metadata_edit_uses_last_remote_baseline(
    repositories,
):
    import json

    root, other, _ = sparse_dataset(repositories)
    sync_data.sync(root, None, publish=False)
    (root / "code.txt").write_text("local code")
    git(root, "add", "code.txt")
    git(root, "commit", "-m", "Unpublished code")
    (root / "data/images/45/local.jpg").write_text("unpublished capture")
    publish(other, "image_orientation_corrections.json", '{"45/photo.jpg": 90}\n')
    sync_data.sync(root, None, publish=False)
    publish(other, "image_orientation_corrections.json", '{"45/photo.jpg": 180}\n')
    sync_data.sync(root, "Combine latest metadata")
    assert json.loads(
        (root / "data/image_orientation_corrections.json").read_text()
    ) == {"45/photo.jpg": 180}
    assert (root / "data/images/45/local.jpg").read_text() == "unpublished capture"


@pytest.mark.parametrize(
    "arguments,publish,preset,targets",
    [
        (["-m", "Share captures"], True, None, []),
        (
            ["pull", "--preset", "images", "data/images/45"],
            False,
            "images",
            ["data/images/45"],
        ),
        (
            ["sync", "data/masks/images/45", "-m", "Share masks"],
            True,
            None,
            ["data/masks/images/45"],
        ),
    ],
)
def test_cli_accepts_presets_paths_and_options_in_either_order(
    tmp_path, monkeypatch, arguments, publish, preset, targets
):
    calls = []
    monkeypatch.setattr(sync_data, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["sync_data.py", *arguments])
    monkeypatch.setattr(
        sync_data, "sync", lambda root, message, **kwargs: calls.append(kwargs)
    )
    assert sync_data.main() == 0
    assert calls == [{"publish": publish, "targets": targets, "preset": preset}]


def test_pull_rejects_unknown_target_without_advancing_git(repositories):
    root, _, _ = sparse_dataset(repositories)
    head = git(root, "rev-parse", "HEAD")
    with pytest.raises(sync_data.SyncError, match="not tracked"):
        sync_data.sync(root, None, publish=False, targets=["data/images/unknown.jpg"])
    assert git(root, "rev-parse", "HEAD") == head


def test_sync_repairs_published_pointer_from_local_cache(repositories):
    root, _, remote = repositories
    capture = root / "data/videos/local.mp4"
    capture.parent.mkdir(parents=True)
    capture.write_text("locally cached video")
    dvc(root, "add", "data")
    files = sync_data.index(root / "data.dvc", root / ".dvc/cache")
    video_hash = files["videos/local.mp4"]
    git(root, "add", "data.dvc")
    git(root, "commit", "-m", "Pointer pushed before data")
    git(root, "push")
    head = git(root, "rev-parse", "HEAD")
    storage = remote.parent / "storage"
    assert not sync_data.cache_file(storage, video_hash).exists()

    sync_data.sync(root, None)

    assert sync_data.cache_file(storage, video_hash).is_file()
    assert git(root, "rev-parse", "HEAD") == head
    assert git(remote, "rev-parse", "main") == head
    assert capture.read_text() == "locally cached video"


def test_repair_still_requires_remote_verification_after_upload(
    repositories, monkeypatch
):
    root, _, remote = repositories
    (root / "data/local.txt").write_text("locally cached data")
    dvc(root, "add", "data")
    git(root, "add", "data.dvc")
    git(root, "commit", "-m", "Pointer pushed before data")
    git(root, "push")
    head = git(remote, "rev-parse", "main")
    original = sync_data.dvc
    monkeypatch.setattr(
        sync_data,
        "dvc",
        lambda cwd, *args: "" if args[0] == "push" else original(cwd, *args),
    )

    with pytest.raises(sync_data.SyncError, match="missing remotely"):
        sync_data.sync(root, None)

    assert git(remote, "rev-parse", "main") == head
    assert (root / "data/local.txt").read_text() == "locally cached data"


def test_repair_uploads_incoming_version_even_if_final_dataset_removes_file(
    repositories,
):
    root, _, remote = repositories
    capture = root / "data/removed.txt"
    capture.write_text("historical version must remain recoverable")
    dvc(root, "add", "data")
    digest = sync_data.index(root / "data.dvc", root / ".dvc/cache")["removed.txt"]
    git(root, "add", "data.dvc")
    git(root, "commit", "-m", "Pointer pushed before data")
    git(root, "push")
    capture.unlink()
    dvc(root, "add", "data")

    sync_data.sync(root, "Remove obsolete file")

    assert "removed.txt" not in sync_data.index(root / "data.dvc", root / ".dvc/cache")
    assert sync_data.cache_file(remote.parent / "storage", digest).is_file()
