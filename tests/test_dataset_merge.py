import csv
import io
import json

import pytest

from dataset_merge import Conflict, choose, merge_content
from sync_data import merge_dataset


def csv_bytes(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(
        out, fieldnames=["Image ID", "Sample ID", "Angle", "Filepath"]
    )
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def image(identity, sample="1", angle="0"):
    return {
        "Image ID": identity,
        "Sample ID": sample,
        "Angle": angle,
        "Filepath": f"images/{identity}.jpg",
    }


def test_manifest_preserves_local_correction_and_remote_addition():
    base = [image("a")]
    local = [image("a", angle="180")]
    incoming = [image("a"), image("b", sample="2")]
    result = merge_content(
        "image_manifest.csv", csv_bytes(base), csv_bytes(local), csv_bytes(incoming)
    )
    rows = list(csv.DictReader(io.StringIO(result.decode())))
    assert rows == [image("a", angle="180"), image("b", sample="2")]


def test_conflicting_field_reports_record_and_both_values():
    with pytest.raises(Conflict, match="Angle.*90.*180"):
        merge_content(
            "image_manifest.csv",
            csv_bytes([image("a")]),
            csv_bytes([image("a", angle="90")]),
            csv_bytes([image("a", angle="180")]),
        )


def test_independent_sample_id_collision_is_not_silently_combined():
    with pytest.raises(Conflict, match="Sample ID 2"):
        merge_content(
            "image_manifest.csv",
            csv_bytes([image("a")]),
            csv_bytes([image("a"), image("b", "2")]),
            csv_bytes([image("a"), image("c", "2")]),
        )


def test_identical_additions_and_unilateral_deletion():
    missing = object()
    assert choose(missing, "same", "same", "image") == "same"
    assert choose("old", missing, "old", "image") is missing
    with pytest.raises(Conflict):
        choose("old", missing, "edited", "image")


def test_review_log_combines_events_without_duplicating_old_event():
    base = b'{"key":"a","action":"q"}\n'
    local = base + base
    incoming = base + b'{"key":"b","action":"q"}\n'
    result = merge_content("image_review.jsonl", base, local, incoming)
    assert [json.loads(line)["key"] for line in result.splitlines()] == ["a", "b"]


def test_review_rewrite_requires_decision():
    with pytest.raises(Conflict, match="removed or rewritten"):
        merge_content(
            "image_review.jsonl",
            b'{"key":"a"}\n',
            b'{"key":"b"}\n',
            b'{"key":"a"}\n{"key":"c"}\n',
        )


def test_duplicate_csv_identity_is_rejected():
    with pytest.raises(Conflict, match="duplicate record"):
        merge_content(
            "image_manifest.csv",
            csv_bytes([]),
            csv_bytes([image("a"), image("a")]),
            csv_bytes([image("b")]),
        )


def test_binary_conflict_and_file_directory_collision(tmp_path):
    with pytest.raises(Conflict, match="different file contents"):
        merge_content("images/a.jpg", b"old", b"local", b"incoming")
    with pytest.raises(Conflict, match="parent directory"):
        merge_dataset(
            {}, {"images/a": "a" * 32}, {"images/a/b.jpg": "b" * 32}, tmp_path
        )


def test_json_merges_distinct_orientation_corrections():
    result = merge_content(
        "image_orientation_corrections.json",
        b'{"a":0}',
        b'{"a":90}',
        b'{"a":0,"b":180}',
    )
    assert json.loads(result) == {"a": 90, "b": 180}


def test_empty_video_manifest_placeholder_is_supported():
    from dataset_merge import validate_references

    validate_references({"video_manifest.csv": b""}, lambda path: b"")
    content = b"Video ID,Filepath\nv1,videos/1.mp4\n"
    result = merge_content(
        "video_manifest.csv", b"", content, b"Video ID,Filepath\nv2,videos/2.mp4\n"
    )
    assert set(
        row["Video ID"] for row in csv.DictReader(io.StringIO(result.decode()))
    ) == {"v1", "v2"}
