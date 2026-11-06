"""Dataset integration checks."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from datasets import DatasetDict, load_from_disk
from PIL import Image

import build_dataset as builder
from build_dataset import audit_masks, build_dataset, build_rows, main, split_rows
from masks import save_mask_bundle


@pytest.fixture
def source(tmp_path):
    images, objects, decisions = [], [], []
    for index in range(6):
        image_id = f"image-{index}"
        Image.new("RGB", (12, 10)).save(tmp_path / f"{index}.png")
        masks = np.zeros((2, 10, 12), dtype=bool)
        masks[0, 1:4, 2:5] = True
        masks[1, 5:9, 7:10] = True
        masks[1, 6, 8] = False  # Pixel area differs from bounding-box area.
        save_mask_bundle(tmp_path / f"masks/{index}.npz", masks, ["a", "b"])
        images.append(
            {
                "Image ID": image_id,
                "Filepath": f"{index}.png",
                "Sample ID": str(index),
                "Capture Session ID": f"session-{index}",
            }
        )
        for mask_index, identity in ((1, "b"), (0, "a")):
            objects.append(
                {
                    "Image ID": image_id,
                    "Instance ID": identity,
                    "Tool ID": f"{index}-{identity}",
                    "Tool Class": "scissors",
                    "Tool Name": "iris",
                    "Mask Path": f"masks/{index}.npz",
                    "Mask Index": mask_index,
                    "Annotation Confidence": 1,
                    "BBox X": 7 if mask_index else 2,
                    "BBox Y": 5 if mask_index else 1,
                    "BBox Width": 3,
                    "BBox Height": 4 if mask_index else 3,
                    **dict.fromkeys(
                        (
                            "Reviewed",
                            "Annotation Complete",
                            "Mask Valid",
                            "Class Valid",
                            "Base Valid",
                            "Visible",
                        ),
                        "TRUE",
                    ),
                }
            )
        decisions.append(json.dumps({"key": image_id, "decision": "accepted"}))
    for name, rows in (
        ("image_manifest.csv", images),
        ("image_object_manifest.csv", objects),
    ):
        with (tmp_path / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (tmp_path / "image_review.jsonl").write_text("\n".join(decisions) + "\n")
    return tmp_path


def test_round_trip_keeps_all_instances_and_indices(source):
    result = build_dataset(source, source / "arrow")
    loaded = load_from_disk(source / "arrow")
    assert sum(map(len, result.values())) == 6
    assert loaded["train"].features["objects"].feature["category"].names == [
        "iris scissors"
    ]
    for subset in loaded.values():
        for row in subset:
            assert "physical_tool_id" not in row
            assert "modified" not in row
            assert [obj["mask_index"] for obj in row["objects"]] == [1, 0]
            assert [obj["instance_id"] for obj in row["objects"]] == ["b", "a"]
            assert all(
                "physical_tool_id" not in obj and "modified" not in obj
                for obj in row["objects"]
            )
            assert {obj["tool_id"] for obj in row["objects"]} == set(row["tool_ids"])
            assert len(row["tool_ids"]) == 2
            assert row["objects"][0]["area"] == 11
            assert not Path(row["image_path"]).is_absolute()
            assert all(
                not Path(obj["mask_path"]).is_absolute() for obj in row["objects"]
            )
    report = json.loads((source / "arrow/build_report.json").read_text())
    assert report["split_by"] == "tool"
    assert report["split_grouping"] == ["physical Tool ID", "Sample ID"]
    assert report["mask_audit"]["overlaps"] == []
    assert report["mask_audit"]["multi_instance_bundles"] == 6
    with pytest.raises(FileExistsError):
        build_dataset(source, source / "arrow")


def test_cli_defaults_use_data_and_export_outside_it(source, monkeypatch):
    dataset = source / "data"
    dataset.mkdir()
    for path in list(source.iterdir()):
        if path != dataset:
            path.rename(dataset / path.name)
    monkeypatch.chdir(source)
    monkeypatch.setattr(builder, "__file__", str(source / "build_dataset.py"))
    monkeypatch.setattr(sys, "argv", ["build_dataset.py"])

    main()

    output = source / "exports/segmentation-dataset"
    loaded = load_from_disk(output)
    assert sum(map(len, loaded.values())) == 6
    for subset in loaded.values():
        for row in subset:
            assert (output / row["image_path"]).is_file()
            assert all((output / obj["mask_path"]).is_file() for obj in row["objects"])
    assert not (dataset / "segmentation-dataset").exists()


def test_review_latest_decision_wins(source):
    with (source / "image_review.jsonl").open("a") as stream:
        stream.write(json.dumps({"key": "image-0", "decision": "in_progress"}) + "\n")
    rows, names, skipped = build_rows(source)
    assert len(rows) == 5
    assert names == ["iris scissors"]
    assert skipped == {"not_accepted": 1}


@pytest.mark.parametrize("split_by", ["tool", "sample"])
def test_missing_sample_ids_are_only_excluded_in_sample_mode(source, split_by):
    path = source / "image_manifest.csv"
    with path.open() as stream:
        images = list(csv.DictReader(stream))
    images[0]["Sample ID"] = ""
    images[1]["Sample ID"] = "   "
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(images[0]))
        writer.writeheader()
        writer.writerows(images)
    rows, _, skipped = build_rows(source, split_by=split_by)
    expected_indices = range(2, 6) if split_by == "sample" else range(6)
    assert [row["image_id"] for row in rows] == [f"image-{i}" for i in expected_indices]
    assert skipped == ({"missing_sample_id": 2} if split_by == "sample" else {})
    result = build_dataset(source, source / "arrow", split_by=split_by)
    assert sum(map(len, result.values())) == len(rows)
    report = json.loads((source / "arrow/build_report.json").read_text())
    assert report["skipped"] == skipped
    assert (source / "masks/0.npz").is_file()
    assert (source / "masks/1.npz").is_file()


def test_split_keeps_connected_tool_and_sample_ids_together(source):
    rows, _, _ = build_rows(source)
    rows[0]["sample_id"] = ""
    rows[1]["sample_id"] = ""
    rows[1]["tool_ids"] = rows[0]["tool_ids"]
    rows[2]["tool_ids"] = rows[1]["tool_ids"]
    rows[3]["sample_id"] = rows[2]["sample_id"]
    sizes = {"train": 0.7, "validation": 0.15, "test": 0.15}
    splits = split_rows(rows, sizes, seed=42)
    assert len(set(splits[:4])) == 1
    assert set(splits) == set(sizes)
    assert splits == split_rows(rows, sizes, seed=42)
    rows[4]["tool_ids"] = rows[0]["tool_ids"]
    rows[5]["tool_ids"] = rows[0]["tool_ids"]
    assert split_rows(rows, sizes, seed=42) == ["train"] * len(rows)


def test_tool_split_does_not_merge_unrelated_images_with_blank_sample_ids(source):
    rows, _, _ = build_rows(source)
    for row in rows:
        row["sample_id"] = ""
    splits = split_rows(rows, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=42)
    assert set(splits) == {"train", "validation", "test"}


def test_split_allows_shared_capture_sessions(source):
    rows, _, _ = build_rows(source)
    for row in rows:
        row["capture_session_id"] = "shared-session"
    splits = split_rows(rows, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=42)
    assert set(splits) == {"train", "validation", "test"}


def test_sample_split_keeps_views_together_but_allows_shared_tools(source):
    rows, _, _ = build_rows(source)
    for index, row in enumerate(rows):
        row["sample_id"] = str(index // 2)
        row["tool_ids"] = ["12", "18"] if index < 2 else ["12"]
    sizes = {"train": 0.7, "validation": 0.15, "test": 0.15}
    assert split_rows(rows, sizes, seed=42) == ["train"] * len(rows)
    splits = split_rows(rows, sizes, seed=42, split_by="sample")
    assert set(splits) == set(sizes)
    assert all(splits[index] == splits[index + 1] for index in (0, 2, 4))
    assert splits == split_rows(rows, sizes, seed=42, split_by="sample")


def test_sample_split_requires_ids_for_every_row(source):
    rows, _, _ = build_rows(source)
    rows[0]["sample_id"] = ""
    with pytest.raises(ValueError, match="requires a Sample ID"):
        split_rows(rows, {"train": 1}, seed=42, split_by="sample")


@pytest.mark.parametrize("seed", range(10))
def test_split_reserves_rare_classes_and_never_evaluates_untrained_labels(source, seed):
    rows, _, _ = build_rows(source)
    # A singleton class, a two-sample class, and a common class. One group
    # contains two labels, and its second view must move with the first.
    for index, row in enumerate(rows):
        for obj in row["objects"]:
            obj["category"] = 0 if index == 0 else 1 if index < 3 else 2
    rows[0]["objects"][1]["category"] = 2
    rows[5]["sample_id"] = rows[0]["sample_id"]
    splits = split_rows(
        rows, {"train": 0.4, "validation": 0.3, "test": 0.3}, seed, split_by="sample"
    )
    assert splits[0] == splits[5] == "train"
    train_labels = {
        obj["category"]
        for row, split in zip(rows, splits)
        if split == "train"
        for obj in row["objects"]
    }
    assert train_labels == {0, 1, 2}
    assert set(splits) == {"train", "validation", "test"}


def test_training_coverage_can_exhaust_groups(source):
    rows, _, _ = build_rows(source)
    for index, row in enumerate(rows):
        for obj in row["objects"]:
            obj["category"] = index
    splits = split_rows(
        rows, {"train": 0.7, "validation": 0.15, "test": 0.15}, 42, split_by="sample"
    )
    assert splits == ["train"] * len(rows)


def test_label_metadata_is_preserved_and_empty_holdouts_are_reported(source):
    result = build_dataset(source, source / "arrow", label_fields=("tool_class",))
    report = json.loads((source / "arrow/build_report.json").read_text())
    assert report["label_fields"] == ["tool_class"]
    assert report["classes"] == ["scissors"]
    assert report["classes_absent_from_train"] == []
    assert all(
        obj["tool_name"] == "iris" and obj["tool_class"] == "scissors"
        for split in result.values()
        for row in split
        for obj in row["objects"]
    )
    # One physical connected group means both requested holdouts stay empty.
    path = source / "image_object_manifest.csv"
    objects = list(csv.DictReader(path.open()))
    for obj in objects:
        obj["Tool ID"] = obj["Instance ID"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(objects[0]))
        writer.writeheader()
        writer.writerows(objects)
    result = build_dataset(source, source / "single-group")
    loaded = load_from_disk(source / "single-group")
    assert len(loaded["train"]) == 6
    assert set(loaded) == {"train"}
    report = json.loads((source / "single-group/build_report.json").read_text())
    assert report["empty_splits"] == ["validation", "test"]
    assert report["splits"]["validation"]["classes_absent"] == ["iris scissors"]


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (("tool_class",), "scissors"),
        (("tool_name",), "iris"),
        (("tool_class", "tool_name"), "scissors iris"),
        (("tool_name", "tool_class"), "iris scissors"),
    ],
)
def test_ordered_label_fields(source, fields, expected):
    rows, names, _ = build_rows(source, label_fields=fields)
    assert names == [expected]
    assert all(obj["category"] == 0 for row in rows for obj in row["objects"])


def test_coarse_labels_merge_names_without_discarding_metadata(source):
    path = source / "image_object_manifest.csv"
    objects = list(csv.DictReader(path.open()))
    for obj in objects:
        if obj["Instance ID"] == "b":
            obj["Tool Name"] = "metzenbaum scissors"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(objects[0]))
        writer.writeheader()
        writer.writerows(objects)
    _, specific_names, _ = build_rows(source)
    assert specific_names == ["iris scissors", "metzenbaum scissors"]
    rows, broad_names, _ = build_rows(source, label_fields=("tool_class",))
    assert broad_names == ["scissors"]
    assert {obj["tool_name"] for row in rows for obj in row["objects"]} == {
        "iris",
        "metzenbaum scissors",
    }
    assert all(obj["category"] == 0 for row in rows for obj in row["objects"])


@pytest.mark.parametrize("fields", [(), ("unknown",), ("tool_class", "tool_class")])
def test_invalid_label_fields_are_rejected(source, fields):
    with pytest.raises(ValueError, match="label_fields"):
        build_rows(source, label_fields=fields)


def test_split_requires_training_fraction(source):
    rows, _, _ = build_rows(source)
    with pytest.raises(ValueError, match="positive training fraction"):
        split_rows(rows, {"train": 0, "validation": 0.5, "test": 0.5}, 42)


def test_audit_reports_overlaps_and_build_refuses_them(source):
    path = source / "masks/0.npz"
    # Simulate an archive written before disjointness was enforced by the saver.
    with np.load(path) as archive:
        data = {key: archive[key] for key in archive.files}
    data["packed_masks"][1] = data["packed_masks"][0]
    np.savez_compressed(path, **data)
    audit = audit_masks(source / "masks")
    assert audit["overlaps"][0]["pixels"] == 9
    assert audit["overlaps"][0]["pairs"] == [{"instances": ["a", "b"], "pixels": 9}]
    with pytest.raises(ValueError, match="Overlapping masks"):
        build_dataset(source, source / "arrow")
    assert not (source / "arrow").exists()
    assert (source / "arrow.mask-audit.json").is_file()


def test_builder_rejects_wrong_mask_identity(source):
    path = source / "image_object_manifest.csv"
    path.write_text(path.read_text().replace("image-0,b,", "image-0,wrong,", 1))
    with pytest.raises(ValueError, match="Mask identity"):
        build_rows(source)


def test_accepted_background_requires_empty_bundle(source):
    path = source / "image_object_manifest.csv"
    objects = list(csv.DictReader(path.open()))
    objects = [obj for obj in objects if obj["Image ID"] != "image-0"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(objects[0]))
        writer.writeheader()
        writer.writerows(objects)
    # Empty accepted images use the annotation workflow's canonical mask path.
    empty_path = source / "masks/images/0.npz"
    with pytest.raises(FileNotFoundError):
        build_rows(source)
    save_mask_bundle(empty_path, np.zeros((0, 10, 12), dtype=bool), [])
    rows, _, _ = build_rows(source)
    assert rows[0]["objects"] == []
    assert rows[0]["tool_ids"] == []
    result = build_dataset(source, source / "arrow")
    assert sum(not row["objects"] for subset in result.values() for row in subset) == 1


def test_duplicate_tool_id_within_image_is_rejected(source):
    path = source / "image_object_manifest.csv"
    with path.open() as stream:
        objects = list(csv.DictReader(stream))
    objects[1]["Tool ID"] = objects[0]["Tool ID"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(objects[0]))
        writer.writeheader()
        writer.writerows(objects)
    with pytest.raises(ValueError, match="Duplicate Tool ID"):
        build_rows(source)


def test_force_cli_replaces_existing_dataset_and_removes_stale_files(
    source, monkeypatch
):
    output = source / "arrow"
    build_dataset(source, output)
    (output / "stale-file.txt").write_text("old export")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dataset.py",
            "--root",
            str(source),
            "--output-path",
            str(output),
            "--validation-size",
            "0",
            "--test-size",
            "0",
            "--force",
            "--split-by",
            "sample",
            "--label-fields",
            "tool_class",
        ],
    )
    main()
    loaded = load_from_disk(output)
    assert set(loaded) == {"train"}
    assert len(loaded["train"]) == 6
    assert not (output / "validation").exists()
    assert not (output / "test").exists()
    assert not (output / "stale-file.txt").exists()
    report = json.loads((output / "build_report.json").read_text())
    assert report["requested_fractions"] == {"train": 1, "validation": 0, "test": 0}
    assert report["split_by"] == "sample"
    assert report["split_grouping"] == ["Sample ID"]
    assert report["label_fields"] == ["tool_class"]
    assert report["classes"] == ["scissors"]
    assert not list(source.glob(".arrow-*"))


def test_force_preserves_existing_dataset_when_save_fails(source, monkeypatch):
    output = source / "arrow"
    build_dataset(source, output)
    original_report = (output / "build_report.json").read_bytes()

    def fail_save(self, path):
        path.mkdir()
        (path / "partial-write.txt").write_text("incomplete")
        raise OSError("simulated write failure")

    monkeypatch.setattr(DatasetDict, "save_to_disk", fail_save)
    with pytest.raises(OSError, match="simulated write failure"):
        build_dataset(source, output, force=True)
    assert (output / "build_report.json").read_bytes() == original_report
    assert sum(map(len, load_from_disk(output).values())) == 6
    assert not list(source.glob(".arrow-*"))


def test_force_refuses_to_overwrite_source_directories(source):
    for output in (source, source / "masks"):
        with pytest.raises(ValueError, match="source asset directories"):
            build_dataset(source, output, force=True)
    assert (source / "image_manifest.csv").is_file()
    assert (source / "masks/0.npz").is_file()
