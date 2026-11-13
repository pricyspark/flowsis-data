import json
import pytest

import annotate
from annotations import VIDEO_MANIFEST_FIELDS, write_csv
from video_annotation import list_videos
from video_data import select_videos, video_numbers


def test_video_numbers_survive_reordering_removal_and_new_videos(tmp_path):
    manifest = tmp_path / "video_manifest.csv"
    a, b, c, d = [{"Video ID": value} for value in ("a", "b", "c", "d")]
    assert video_numbers(manifest, [a, b]) == {"a": 1, "b": 2}
    assert video_numbers(manifest, [b, a]) == {"a": 1, "b": 2}
    assert video_numbers(manifest, [a, c]) == {"a": 1, "b": 2, "c": 3}
    assert video_numbers(manifest, [b, d, a]) == {"a": 1, "b": 2, "c": 3, "d": 4}
    assert json.loads(manifest.with_suffix(".numbers.json").read_text())["c"] == 3


@pytest.mark.parametrize(
    "saved", [{"a": 1, "b": 1}, {"a": True}, {"a": 0}, {"a": "1"}, []]
)
def test_invalid_number_registry_is_not_silently_reassigned(tmp_path, saved):
    manifest = tmp_path / "video_manifest.csv"
    registry = manifest.with_suffix(".numbers.json")
    contents = json.dumps(saved)
    registry.write_text(contents)
    with pytest.raises(ValueError):
        video_numbers(manifest, [{"Video ID": "a"}])
    assert registry.read_text() == contents


def test_numeric_and_hash_selection_use_same_video(tmp_path):
    args = annotate.parse_args(["videos", "--data-dir", str(tmp_path), "--video", "2"])
    videos = [{"Video ID": "a"}, {"Video ID": "b"}]
    assert select_videos(args, videos) == [(2, videos[1])]
    legacy = annotate.parse_args(
        ["videos", "--data-dir", str(tmp_path), "--video-id", "b"]
    )
    assert select_videos(legacy, videos) == [(2, videos[1])]
    args.video = 3
    with pytest.raises(ValueError, match="Unknown video number 3"):
        select_videos(args, videos)


def test_numeric_and_hash_selectors_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        annotate.parse_args(["videos", "--video", "1", "--video-id", "a"])


def test_listing_shows_permanent_numbers_and_does_not_rewrite_manifest(
    tmp_path, capsys
):
    manifest = tmp_path / "video_manifest.csv"
    videos = [
        {
            "Video ID": "a", "Tool ID": "12",
            "Filepath": "videos/12/first.mp4", "Frame Count": "10",
        },
        {
            "Video ID": "b", "Tool ID": "12",
            "Filepath": "videos/12/second.mp4", "Frame Count": "20",
        },
    ]
    write_csv(manifest, videos, VIDEO_MANIFEST_FIELDS)
    before = manifest.read_bytes()
    list_videos(annotate.parse_args(["list-videos", "--data-dir", str(tmp_path)]))
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split() == ["1", "12", "0/10", "videos/12/first.mp4"]
    assert lines[2].split() == ["2", "12", "0/20", "videos/12/second.mp4"]
    assert manifest.read_bytes() == before
