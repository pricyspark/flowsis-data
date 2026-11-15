"""Leakage and export regressions for two-stage holdouts."""

import copy
import json
import random

import pytest
from datasets import load_from_disk

import build_dataset as builder
from dataset_splits import split_tool_samples


def row(tool, sample, category=0, *, video="", session=None):
    return {
        "image_id": f"image-{sample}-{tool}",
        "sample_id": sample,
        "video_id": video,
        "capture_session_id": session or f"session-{sample}",
        "tool_ids": [tool],
        "base_eligible": True,
        "source_type": "video" if video else "image",
        "sequence_id": video,
        "timestamp_seconds": 0.0,
        "mask_bundle_path": "",
        "image_path": "/tmp/example.png",
        "video_path": "",
        "frame_index": 0,
        "width": 16,
        "height": 16,
        "objects": [
            {
                "tool_id": tool,
                "category": category,
                "class_valid": True,
                "mask_valid": True,
                "visible": True,
                "base_valid": True,
                "instance_id": tool,
                "track_id": tool,
                "area": 4.0,
                "bbox": [0, 0, 2, 2],
                "tool_name": "example",
                "tool_class": "example",
                "mask_path": "",
                "mask_index": 0,
                "annotation_confidence": 1.0,
                "visibility": "clear",
                "preferred_track": False,
            }
        ],
    }


@pytest.fixture
def rows():
    return [
        row(str(tool), f"capture-{tool}-{capture}", tool % 2)
        for tool in range(12)
        for capture in range(4)
    ]


def test_two_tests_are_distinct_and_known_tools_are_supervised(rows):
    assignments, report = split_tool_samples(rows, unseen_tool_ids=["0"])
    tools = {
        name: {
            tool
            for r, s in zip(rows, assignments)
            if s == name
            for tool in r["tool_ids"]
        }
        for name in set(assignments)
    }
    assert set(tools) == {"train", "validation", "test_known", "test_unseen"}
    assert "0" not in tools["train"] | tools["validation"] | tools["test_known"]
    assert tools["test_known"] <= tools["train"]
    assert tools["validation"] <= tools["train"]
    assert report["unseen_tool_ids"] == ["0"]
    assert report["splits"]["test_known"]["capture_groups"] == 5
    assert {
        o["category"]
        for r, s in zip(rows, assignments)
        if s == "train"
        for o in r["objects"]
    } == {0, 1}


def test_mixed_scene_quarantines_entire_sample_and_connected_video(rows):
    mixed = row("0", "mixed", video="shared-video")
    mixed["tool_ids"].append("1")
    mixed["objects"].append(copy.deepcopy(row("1", "unused", 1)["objects"][0]))
    rows += [mixed, row("1", "mixed", 1), row("1", "another", 1, video="shared-video")]
    assignments, report = split_tool_samples(rows, unseen_tool_ids=["0"])
    assert assignments[-3:] == ["test_unseen"] * 3
    assert report["known_companion_tool_ids_in_unseen_test"] == ["1"]


def test_session_grouping_is_transitive(rows):
    rows[4]["capture_session_id"] = rows[0]["capture_session_id"]
    assignments, _ = split_tool_samples(rows, unseen_tool_ids=["0"], group_by="session")
    assert assignments[4] == "test_unseen"
    assignments, _ = split_tool_samples(rows, unseen_tool_ids=["0"], group_by="sample")
    assert assignments[4] != "test_unseen"


def test_seeded_selection_and_assignments_ignore_row_order(rows):
    _, original = split_tool_samples(rows, seed=17)
    random.Random(99).shuffle(rows)
    _, reordered = split_tool_samples(rows, seed=17)
    assert reordered == original


def test_singleton_class_stays_in_training(rows):
    rows += [row("rare", "rare-capture", 2)]
    assignments, report = split_tool_samples(rows)
    assert assignments[-1] == "train"
    assert "rare" not in report["unseen_tool_ids"]
    with pytest.raises(ValueError, match="coverage"):
        split_tool_samples(rows, unseen_tool_ids=["rare"])


def test_quarantine_cannot_make_a_companion_accidentally_unseen(rows):
    mixed = row("0", "mixed")
    mixed["tool_ids"].append("companion")
    mixed["objects"].append(row("companion", "unused")["objects"][0])
    rows.append(mixed)
    with pytest.raises(ValueError, match="coverage"):
        split_tool_samples(rows, unseen_tool_ids=["0"])


def test_missing_group_metadata_fails(rows):
    rows[0]["sample_id"] = ""
    with pytest.raises(ValueError, match="Sample ID"):
        split_tool_samples(rows)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unseen_tool_size": 0},
        {"unseen_tool_size": float("nan")},
        {"validation_size": 0.6, "test_size": 0.4},
        {"test_size": -1},
        {"unseen_tool_ids": ["unknown"]},
    ],
)
def test_invalid_requests_fail(rows, kwargs):
    with pytest.raises(ValueError):
        split_tool_samples(rows, **kwargs)


def test_insufficient_independent_captures_fails():
    rows = [row(str(tool), f"sample-{tool}") for tool in range(5)]
    with pytest.raises(ValueError, match="capture groups"):
        split_tool_samples(rows, unseen_tool_ids=["0"])


def test_export_round_trip_and_manifest(rows, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "audit_masks", lambda *a, **kw: {"overlaps": []})
    monkeypatch.setattr(
        builder,
        "build_rows",
        lambda *a, **kw: (copy.deepcopy(rows), ["class_a", "class_b"], {}),
    )
    output = tmp_path / "export"
    builder.build_dataset(
        tmp_path,
        output,
        split_by="tool-sample",
        unseen_tool_ids=["0"],
        validation_size=0.1,
        test_size=0.1,
    )
    loaded = load_from_disk(output)
    assert set(loaded) == {"train", "validation", "test_known", "test_unseen"}
    report = json.loads((output / "build_report.json").read_text())
    manifest = json.loads((output / "split_assignments.json").read_text())
    assert report["physical_tool_holdout"]["unseen_tool_ids"] == ["0"]
    assert report["augmentation_recommendation"]["rotation_degrees"] == [-22.5, 22.5]
    assert len(manifest["assignments"]) == len(rows)
    for split, subset in loaded.items():
        assert all(manifest["assignments"][r["image_id"]] == split for r in subset)


def test_section_splitting_is_rejected_before_reading_assets(tmp_path):
    with pytest.raises(ValueError, match="whole videos"):
        builder.build_dataset(
            tmp_path, tmp_path / "export", split_by="tool-sample", video_sections=3
        )


def test_holdouts_have_video_and_multi_tool_groups_when_available(rows):
    for index, item in enumerate(rows):
        item["video_id"] = f"video-{index}"
        companion = copy.deepcopy(item["objects"][0])
        companion["instance_id"] += "-second"
        item["objects"].append(companion)
    assignments, report = split_tool_samples(rows, unseen_tool_ids=["0"])
    assert {"video", "multi_tool"} <= set(report["enforced_capture_conditions"])
    for split in ("train", "validation", "test_known"):
        part = [item for item, name in zip(rows, assignments) if name == split]
        assert any(item["video_id"] for item in part)
        assert any(len(item["objects"]) > 1 for item in part)
